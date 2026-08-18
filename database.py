"""
El-Peak — Data Layer v12.1 (P0 hardening)
PostgreSQL multi-user data layer with improved safety,
payments, quiz sessions, OAuth codes, and rate limiting.
P0 changes: IP-based auth rate limiting table, safer pool default,
advisory-locked schema init (multi-worker safe).
"""
import os
import json
import time
import secrets
import datetime as dt
import logging
from typing import Any
import asyncpg

logger = logging.getLogger("el-peak.database")

DATABASE_URL = os.getenv("DATABASE_URL", "")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    created_at TEXT NOT NULL,
    tier TEXT DEFAULT 'free',
    ai_questions_used INTEGER DEFAULT 0,
    pdf_uploads_used INTEGER DEFAULT 0,
    usage_reset_date TEXT,
    username TEXT,
    language TEXT DEFAULT 'en',
    user_type TEXT DEFAULT 'general',
    grade TEXT,
    auth_provider TEXT DEFAULT 'local',
    subscription_status TEXT DEFAULT 'inactive',
    subscription_expires_at TEXT,
    last_login_at TEXT
);
CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    subject TEXT DEFAULT 'General',
    score REAL DEFAULT 50.0,
    attempts INTEGER DEFAULT 0,
    updated_at TEXT,
    UNIQUE(user_id, name)
);
CREATE TABLE IF NOT EXISTS attempts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    correct INTEGER NOT NULL,
    total INTEGER NOT NULL,
    pct REAL NOT NULL,
    source TEXT DEFAULT 'static',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    topic TEXT,
    deadline TEXT NOT NULL,
    done INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS payments (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tier TEXT NOT NULL,
    amount_cents INTEGER DEFAULT 0,
    currency TEXT DEFAULT 'EGP',
    merchant_order_id TEXT UNIQUE,
    paymob_order_id TEXT,
    status TEXT DEFAULT 'pending',
    raw_payload TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS quiz_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    source TEXT NOT NULL,
    difficulty TEXT,
    questions JSONB NOT NULL,
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS oauth_codes (
    code TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rate_limits (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    last_request_epoch DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS ip_rate_limits (
    key TEXT PRIMARY KEY,
    last_request_epoch DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_events (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    amount INTEGER NOT NULL DEFAULT 1,
    tier TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_created ON attempts(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_topics_user ON topics(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_payments_user ON payments(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_quiz_sessions_user ON quiz_sessions(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_events_user ON usage_events(user_id, created_at);
"""

_pool = None

async def get_pool():
    global _pool
    if _pool is not None:
        return _pool
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL is missing. Set it in Railway or .env.")
        raise RuntimeError("Missing DATABASE_URL")
    clean_url = db_url.split("?")[0]
    is_local = "localhost" in clean_url or "127.0.0.1" in clean_url
    min_size = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
    # P0: default lowered from 10 to 5. With UVICORN_WORKERS=2 that is
    # 2 x 5 = 10 connections max, safe on small Railway Postgres plans.
    max_size = int(os.getenv("DB_POOL_MAX_SIZE", "5"))
    ssl_mode = False if is_local else "require"
    host_part = clean_url.split("@")[1] if "@" in clean_url else clean_url
    logger.info("Connecting to PostgreSQL at: %s", host_part)
    _pool = await asyncpg.create_pool(
        dsn=clean_url,
        min_size=min_size,
        max_size=max_size,
        ssl=ssl_mode,
        command_timeout=60,
        statement_cache_size=0
    )
    return _pool

async def close_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

async def ping():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")

def now_str() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

def today_utc() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()

def parse_dt(s: str) -> dt.datetime:
    try:
        clean_s = str(s).replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(clean_s)
        if d.tzinfo is not None:
            return d.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return d
    except Exception:
        return now_utc() + dt.timedelta(days=7)

async def init_db():
    pool = await get_pool()
    async with pool.acquire() as conn:
        # P0: with multiple workers booting simultaneously, concurrent
        # CREATE TABLE IF NOT EXISTS can race. Serialize schema init with
        # a session-level advisory lock.
        await conn.execute("SELECT pg_advisory_lock(7331)")
        try:
            statements = SCHEMA.strip().split(";")
            for stmt in statements:
                if stmt.strip():
                    await conn.execute(stmt.strip())
            # Compatibility upgrades for older databases
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS tier TEXT DEFAULT 'free'")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS ai_questions_used INTEGER DEFAULT 0")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS pdf_uploads_used INTEGER DEFAULT 0")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS usage_reset_date TEXT")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS username TEXT")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS language TEXT DEFAULT 'en'")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS user_type TEXT DEFAULT 'general'")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS grade TEXT")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider TEXT DEFAULT 'local'")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_status TEXT DEFAULT 'inactive'")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_expires_at TEXT")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TEXT")
            await conn.execute("ALTER TABLE topics ADD COLUMN IF NOT EXISTS user_id INTEGER DEFAULT 1")
            await conn.execute("ALTER TABLE topics ADD COLUMN IF NOT EXISTS subject TEXT DEFAULT 'General'")
            await conn.execute("ALTER TABLE topics ADD COLUMN IF NOT EXISTS score REAL DEFAULT 50.0")
            await conn.execute("ALTER TABLE topics ADD COLUMN IF NOT EXISTS attempts INTEGER DEFAULT 0")
            await conn.execute("ALTER TABLE topics ADD COLUMN IF NOT EXISTS updated_at TEXT")
            await conn.execute("ALTER TABLE attempts ADD COLUMN IF NOT EXISTS user_id INTEGER DEFAULT 1")
            await conn.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS user_id INTEGER DEFAULT 1")
        finally:
            await conn.execute("SELECT pg_advisory_unlock(7331)")

-----------------------------
AUTH / USERS
-----------------------------
async def create_user(
    email: str,
    hashed_pw: str,
    user_type: str = "general",
    grade: str = None,
    username: str = None,
    language: str = "en",
    provider: str = "local"
) -> dict:
    pool = await get_pool()
    username = username or email.split("@")[0]
    async with pool.acquire() as conn:
        try:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO users(
                        email, password, created_at, user_type, grade,
                        username, language, auth_provider
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id, email
                    """,
                    email.lower().strip(),
                    hashed_pw,
                    now_str(),
                    user_type,
                    grade,
                    username,
                    language,
                    provider
                )
                return dict(row) if row else None
        except asyncpg.UniqueViolationError:
            return None

async def get_user_by_email(email: str) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE lower(email)=lower($1)",
            email.strip()
        )
        return dict(row) if row else None

async def get_user_full(user_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, email, username, language, user_type, grade,
                tier, auth_provider, subscription_status,
                subscription_expires_at, created_at, last_login_at
            FROM users
            WHERE id=$1
            """,
            user_id
        )
        return dict(row) if row else None

async def get_user_by_id(user_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, email, username, language, user_type, grade,
                tier, subscription_status, subscription_expires_at
            FROM users
            WHERE id=$1
            """,
            user_id
        )
        return dict(row) if row else None

async def update_last_login(user_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET last_login_at=$2 WHERE id=$1",
            user_id,
            now_str()
        )

async def update_user_settings(
    user_id: int,
    username: str,
    language: str,
    user_type: str,
    grade: str | None
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                UPDATE users
                SET username=$1, language=$2, user_type=$3, grade=$4
                WHERE id=$5
                """,
                username,
                language,
                user_type,
                grade,
                user_id
            )

async def update_user_password(user_id: int, hashed_pw: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE users SET password=$1 WHERE id=$2",
                hashed_pw,
                user_id
            )

-----------------------------
OAUTH CODES
-----------------------------
async def create_oauth_code(user_id: int) -> str:
    code = secrets.token_urlsafe(32)
    expires = now_utc() + dt.timedelta(minutes=5)
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO oauth_codes(code, user_id, expires_at, created_at)
            VALUES ($1, $2, $3, $4)
            """,
            code,
            user_id,
            expires.isoformat(timespec="seconds"),
            now_str()
        )
    return code

async def consume_oauth_code(code: str) -> int | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            DELETE FROM oauth_codes
            WHERE code=$1 AND expires_at > $2
            RETURNING user_id
            """,
            code,
            now_str()
        )
        return row["user_id"] if row else None

-----------------------------
TOPICS
-----------------------------
async def ensure_topic(user_id: int, name: str, subject: str = "General") -> dict:
    name = name.strip()
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM topics WHERE user_id=$1 AND lower(name)=lower($2)",
                user_id,
                name
            )
            if row:
                return dict(row)
            try:
                row = await conn.fetchrow(
                    """
                    INSERT INTO topics(user_id, name, subject, score, attempts, updated_at)
                    VALUES ($1, $2, $3, 50.0, 0, $4)
                    RETURNING *
                    """,
                    user_id,
                    name,
                    subject,
                    now_str()
                )
                return dict(row) if row else None
            except asyncpg.UniqueViolationError:
                row = await conn.fetchrow(
                    "SELECT * FROM topics WHERE user_id=$1 AND lower(name)=lower($2)",
                    user_id,
                    name
                )
                return dict(row) if row else None

async def get_topics(user_id: int) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM topics WHERE user_id=$1 ORDER BY name",
            user_id
        )
        return [dict(r) for r in rows]

-----------------------------
ATTEMPTS
-----------------------------
async def record_attempt(
    user_id: int,
    topic_name: str,
    correct: int,
    total: int,
    source: str
) -> dict:
    topic = await ensure_topic(user_id, topic_name)
    if not topic:
        raise RuntimeError("Could not create or find topic.")
    pct = round(correct / total * 100, 1) if total else 0.0
    old = float(topic["score"] or 50.0)
    if int(topic["attempts"] or 0) == 0:
        new = pct
    else:
        new = round(0.55 * old + 0.45 * pct, 1)
    new = max(0.0, min(100.0, new))
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE topics SET score=$1, attempts=attempts+1, updated_at=$2 WHERE id=$3",
                new,
                now_str(),
                topic["id"]
            )
            await conn.execute(
                """
                INSERT INTO attempts(user_id, topic_id, correct, total, pct, source, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                user_id,
                topic["id"],
                correct,
                total,
                pct,
                source,
                now_str()
            )
    return {
        "name": topic["name"],
        "old": old,
        "new": new,
        "delta": round(new - old, 1),
        "pct": pct
    }

async def recent_attempts(user_id: int, limit: int = 20) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT a.pct, a.source, a.created_at, t.name AS topic
            FROM attempts a
            JOIN topics t ON t.id=a.topic_id
            WHERE a.user_id=$1
            ORDER BY a.created_at DESC
            LIMIT $2
            """,
            user_id,
            limit
        )
        return [dict(r) for r in rows]

async def heatmap(user_id: int, days: int = 84) -> list[dict]:
    start = today_utc() - dt.timedelta(days=days - 1)
    counts = {start + dt.timedelta(days=i): 0 for i in range(days)}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT SUBSTR(created_at, 1, 10) AS d, COUNT(*) AS n
            FROM attempts
            WHERE user_id=$1 AND created_at >= $2
            GROUP BY d
            """,
            user_id,
            start.isoformat()
        )
        for row in rows:
            key = dt.date.fromisoformat(row["d"])
            if key in counts:
                counts[key] = int(row["n"])
    return [
        {
            "date": k.isoformat(),
            "count": v
        }
        for k, v in counts.items()
    ]

-----------------------------
STATS
-----------------------------
async def get_stats(user_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        topics = [dict(r) for r in await conn.fetch(
            "SELECT * FROM topics WHERE user_id=$1",
            user_id
        )]
        total_attempts = sum(int(t["attempts"] or 0) for t in topics)
        if topics:
            w_sum = sum((1 + int(t["attempts"] or 0)) for t in topics)
            w_avg = sum(float(t["score"] or 0) * (1 + int(t["attempts"] or 0)) for t in topics) / w_sum
        else:
            w_avg = 0.0
        week_ago = (now_utc() - dt.timedelta(days=7)).isoformat(timespec="seconds")
        attempts_7d = await conn.fetchval(
            "SELECT COUNT(*) FROM attempts WHERE user_id=$1 AND created_at >= $2",
            user_id,
            week_ago
        )
        avg_pct_val = await conn.fetchval(
            "SELECT AVG(pct) FROM attempts WHERE user_id=$1",
            user_id
        )
        avg_pct = float(avg_pct_val) if avg_pct_val is not None else 0.0
        peak = int(round(min(100, max(0, 0.75 * w_avg + 0.25 * min(1.0, attempts_7d / 8) * 100))))
        best = max(topics, key=lambda t: t["score"]) if topics else None
        weak = [t for t in topics if float(t["score"] or 0) < 60]
        days_set = {
            r["d"]
            for r in await conn.fetch(
                "SELECT DISTINCT SUBSTR(created_at, 1, 10) AS d FROM attempts WHERE user_id=$1",
                user_id
            )
        }
        streak = 0
        day = today_utc()
        if day.isoformat() not in days_set:
            day -= dt.timedelta(days=1)
        while day.isoformat() in days_set:
            streak += 1
            day -= dt.timedelta(days=1)
        return {
            "peak": peak,
            "weighted_mastery": round(w_avg, 1),
            "attempts": total_attempts,
            "attempts_7d": int(attempts_7d or 0),
            "avg_pct": round(avg_pct, 1),
            "streak": streak,
            "topic_count": len(topics),
            "weak_count": len(weak),
            "best_topic": best["name"] if best else None
        }

-----------------------------
TASKS / SCHEDULER
-----------------------------
async def add_task(user_id: int, title: str, topic: str | None, deadline: str) -> dict:
    if topic:
        await ensure_topic(user_id, topic)
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO tasks(user_id, title, topic, deadline, created_at)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                user_id,
                title.strip(),
                topic or None,
                deadline,
                now_str()
            )
            return dict(row) if row else None

async def toggle_task(user_id: int, task_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE tasks SET done = 1 - done WHERE id=$1 AND user_id=$2",
                task_id,
                user_id
            )
            row = await conn.fetchrow(
                "SELECT * FROM tasks WHERE id=$1 AND user_id=$2",
                task_id,
                user_id
            )
            return dict(row) if row else {}

async def delete_task(user_id: int, task_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM tasks WHERE id=$1 AND user_id=$2",
                task_id,
                user_id
            )

async def get_schedule(user_id: int) -> dict:
    now = now_utc()
    pool = await get_pool()
    async with pool.acquire() as conn:
        tasks = [dict(r) for r in await conn.fetch(
            "SELECT * FROM tasks WHERE user_id=$1",
            user_id
        )]
        topic_rows = [dict(r) for r in await conn.fetch(
            "SELECT * FROM topics WHERE user_id=$1",
            user_id
        )]
    topics = {t["name"].lower(): t for t in topic_rows}
    queue = []
    cleared = []
    for t in tasks:
        dl = parse_dt(t["deadline"])
        hours = (dl - now).total_seconds() / 3600
        topic = topics.get((t.get("topic") or "").lower())
        t["topic_score"] = topic["score"] if topic else None
        if t["done"]:
            t["tier"] = "DONE"
            t["priority"] = 0
            cleared.append(t)
            continue
        if hours <= 0:
            deadline_score = 100.0
        else:
            deadline_score = max(0.0, 100 - hours * (100 / 168))
        weakness = 100 - float(topic["score"] if topic else 55)
        t["priority"] = int(round(0.62 * deadline_score + 0.38 * weakness))
        if hours < 0:
            t["tier"] = "OVERDUE"
        elif hours < 24:
            t["tier"] = "CRITICAL"
        elif hours < 72:
            t["tier"] = "HIGH"
        elif hours < 168:
            t["tier"] = "THIS WEEK"
        else:
            t["tier"] = "PLANNED"
        queue.append(t)
    queue.sort(key=lambda t: t["priority"], reverse=True)
    cleared.sort(key=lambda t: t["deadline"], reverse=True)
    busy = {(t.get("topic") or "").lower() for t in tasks if not t["done"]}
    suggestions = [
        {
            "topic": tp["name"],
            "score": tp["score"]
        }
        for tp in sorted(topic_rows, key=lambda x: float(x["score"] or 0))
        if float(tp["score"] or 0) < 60 and tp["name"].lower() not in busy
    ][:4]
    return {
        "queue": queue,
        "cleared": cleared,
        "suggestions": suggestions
    }

-----------------------------
USAGE / PAYWALL
-----------------------------
async def get_user_usage(user_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id,
                tier,
                ai_questions_used,
                pdf_uploads_used,
                usage_reset_date,
                subscription_status,
                subscription_expires_at
            FROM users
            WHERE id=$1
            """,
            user_id
        )
        return dict(row) if row else None

async def reset_usage_if_needed(user_id: int):
    current_month = now_utc().strftime("%Y-%m")
    reset_date = today_utc().isoformat()
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                UPDATE users
                SET
                    ai_questions_used = 0,
                    pdf_uploads_used = 0,
                    usage_reset_date = $2
                WHERE id=$1
                  AND (
                    usage_reset_date IS NULL
                    OR SUBSTR(usage_reset_date, 1, 7) <> $3
                  )
                """,
                user_id,
                reset_date,
                current_month
            )

async def increment_usage(user_id: int, ai_questions: int = 0, pdf_uploads: int = 0):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await reset_usage_if_needed(user_id)
            if ai_questions > 0:
                await conn.execute(
                    "UPDATE users SET ai_questions_used = ai_questions_used + $1 WHERE id=$2",
                    ai_questions,
                    user_id
                )
            if pdf_uploads > 0:
                await conn.execute(
                    "UPDATE users SET pdf_uploads_used = pdf_uploads_used + $1 WHERE id=$2",
                    pdf_uploads,
                    user_id
                )

async def upgrade_user_tier(user_id: int, tier: str):
    tier = tier.strip().lower()
    if tier == "free":
        subscription_status = "inactive"
        expires = None
    else:
        subscription_status = "active"
        expires = (now_utc() + dt.timedelta(days=30)).isoformat(timespec="seconds")
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                UPDATE users
                SET tier=$1,
                    subscription_status=$2,
                    subscription_expires_at=$3
                WHERE id=$4
                """,
                tier,
                subscription_status,
                expires,
                user_id
            )

-----------------------------
PAYMENTS
-----------------------------
async def create_payment(
    user_id: int,
    tier: str,
    amount_cents: int,
    merchant_order_id: str,
    status: str = "pending"
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO payments(
                user_id, tier, amount_cents, merchant_order_id, status, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (merchant_order_id) DO NOTHING
            """,
            user_id,
            tier,
            amount_cents,
            merchant_order_id,
            status,
            now_str()
        )

async def update_payment_paymob_order(merchant_order_id: str, paymob_order_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE payments
            SET paymob_order_id=$2, updated_at=$3
            WHERE merchant_order_id=$1
            """,
            merchant_order_id,
            paymob_order_id,
            now_str()
        )

async def update_payment_status(merchant_order_id: str, status: str, raw_payload: str | None = None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE payments
            SET status=$2, raw_payload=COALESCE($3, raw_payload), updated_at=$4
            WHERE merchant_order_id=$1
            """,
            merchant_order_id,
            status,
            raw_payload,
            now_str()
        )

async def finalize_payment(
    user_id: int,
    tier: str,
    merchant_order_id: str,
    amount_cents: Any = None,
    raw_payload: str | None = None
) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            existing = await conn.fetchrow(
                "SELECT * FROM payments WHERE merchant_order_id=$1",
                merchant_order_id
            )
            if existing and existing["status"] == "paid":
                return False
            if existing:
                await conn.execute(
                    """
                    UPDATE payments
                    SET status='paid',
                        raw_payload=COALESCE($2, raw_payload),
                        amount_cents=COALESCE($3, amount_cents),
                        updated_at=$4
                    WHERE merchant_order_id=$1
                    """,
                    merchant_order_id,
                    raw_payload,
                    int(amount_cents) if amount_cents is not None else None,
                    now_str()
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO payments(
                        user_id, tier, amount_cents, merchant_order_id,
                        status, raw_payload, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, 'paid', $5, $6, $7)
                    ON CONFLICT (merchant_order_id) DO NOTHING
                    """,
                    user_id,
                    tier,
                    int(amount_cents) if amount_cents is not None else 0,
                    merchant_order_id,
                    raw_payload,
                    now_str(),
                    now_str()
                )
    await upgrade_user_tier(user_id, tier)
    return True

async def get_payments(user_id: int, limit: int = 20) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, tier, amount_cents, currency, status, merchant_order_id, created_at
            FROM payments
            WHERE user_id=$1
            ORDER BY id DESC
            LIMIT $2
            """,
            user_id,
            limit
        )
        return [dict(r) for r in rows]

-----------------------------
RATE LIMITING
-----------------------------
async def check_ai_rate_limit(user_id: int, seconds: int) -> tuple[bool, int]:
    now = time.time()
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT last_request_epoch FROM rate_limits WHERE user_id=$1",
                user_id
            )
            if row:
                elapsed = now - float(row["last_request_epoch"])
                if elapsed < seconds:
                    return False, int(seconds - elapsed) + 1
            await conn.execute(
                """
                INSERT INTO rate_limits(user_id, last_request_epoch)
                VALUES ($1, $2)
                ON CONFLICT (user_id)
                DO UPDATE SET last_request_epoch=$2
                """,
                user_id,
                now
            )
    return True, 0

async def check_ip_rate_limit(key: str, seconds: int) -> tuple[bool, int]:
    """
    P0: same epoch-interval mechanism as check_ai_rate_limit, but keyed by
    an arbitrary string (client IP) for anonymous endpoints (login/signup).
    DB-backed, so it is correct across workers and (later) instances.
    """
    now = time.time()
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT last_request_epoch FROM ip_rate_limits WHERE key=$1",
                key
            )
            if row:
                elapsed = now - float(row["last_request_epoch"])
                if elapsed < seconds:
                    return False, int(seconds - elapsed) + 1
            await conn.execute(
                """
                INSERT INTO ip_rate_limits(key, last_request_epoch)
                VALUES ($1, $2)
                ON CONFLICT (key)
                DO UPDATE SET last_request_epoch=$2
                """,
                key,
                now
            )
    return True, 0

-----------------------------
USAGE EVENTS
-----------------------------
async def log_usage_event(user_id: int, kind: str, amount: int, tier: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO usage_events(user_id, kind, amount, tier, created_at)
            VALUES ($1, $2, $3, $4, $5)
            """,
            user_id,
            kind,
            amount,
            tier,
            now_str()
        )

-----------------------------
QUIZ SESSIONS
-----------------------------
async def create_quiz_session(
    user_id: int,
    topic: str,
    source: str,
    difficulty: str,
    questions: list
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO quiz_sessions(user_id, topic, source, difficulty, questions, status, created_at)
            VALUES ($1, $2, $3, $4, $5, 'active', $6)
            RETURNING id
            """,
            user_id,
            topic,
            source,
            difficulty,
            json.dumps(questions, ensure_ascii=False),
            now_str()
        )
        return row["id"] if row else None

async def get_quiz_sessions(user_id: int, limit: int = 20) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, topic, source, difficulty, status, created_at, completed_at
            FROM quiz_sessions
            WHERE user_id=$1
            ORDER BY id DESC
            LIMIT $2
            """,
            user_id,
            limit
        )
        return [dict(r) for r in rows]

async def get_quiz_session(user_id: int, session_id: int) -> dict | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM quiz_sessions
            WHERE user_id=$1 AND id=$2
            """,
            user_id,
            session_id
        )
        if not row:
            return None
        item = dict(row)
        if isinstance(item.get("questions"), str):
            try:
                item["questions"] = json.loads(item["questions"])
            except Exception:
                item["questions"] = []
        return item

async def complete_quiz_session(user_id: int, session_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE quiz_sessions
            SET status='completed', completed_at=$3
            WHERE user_id=$1 AND id=$2
            """,
            user_id,
            session_id,
            now_str()
        )

-----------------------------
CLEANUP
-----------------------------
async def cleanup_old_pdf_attempts():
    """
    Deletes PDF quiz attempts older than 14 days to save database space.
    Also removes expired OAuth codes.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            cutoff = (now_utc() - dt.timedelta(days=14)).isoformat(timespec="seconds")
            await conn.execute(
                "DELETE FROM attempts WHERE source='pdf' AND created_at < $1",
                cutoff
            )
            await conn.execute(
                "DELETE FROM oauth_codes WHERE expires_at < $1",
                now_str()
            )
