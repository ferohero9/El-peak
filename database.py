"""
El-Peak — Data Layer v9 (PostgreSQL Multi-User, Edu-Stages, Rate Limits)
"""
import datetime as dt
import os
import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/el-peak")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id                SERIAL PRIMARY KEY,
    email             TEXT NOT NULL UNIQUE,
    password          TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    tier              TEXT DEFAULT 'free',
    ai_questions_used INTEGER DEFAULT 0,
    pdf_uploads_used  INTEGER DEFAULT 0,
    usage_reset_date  TEXT,
    username          TEXT,
    language          TEXT DEFAULT 'en',
    user_type         TEXT DEFAULT 'general',
    grade             TEXT
);
CREATE TABLE IF NOT EXISTS topics (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    name       TEXT NOT NULL,
    subject    TEXT DEFAULT 'General',
    score      REAL DEFAULT 50.0,
    attempts   INTEGER DEFAULT 0,
    updated_at TEXT,
    UNIQUE(user_id, name)
);
CREATE TABLE IF NOT EXISTS attempts (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    topic_id   INTEGER NOT NULL REFERENCES topics(id),
    correct    INTEGER NOT NULL,
    total      INTEGER NOT NULL,
    pct        REAL NOT NULL,
    source     TEXT DEFAULT 'static',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    title      TEXT NOT NULL,
    topic      TEXT,
    deadline   TEXT NOT NULL,
    done       INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_attempts_created ON attempts(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_topics_user ON topics(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id);
"""

_pool = None

async def get_pool():
    global _pool
    if _pool is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url or "localhost" in db_url:
            print("❌ FATAL: DATABASE_URL is missing or invalid! Please set it in Railway.")
            raise RuntimeError("Missing DATABASE_URL")
        
        clean_url = db_url.split("?")[0]
        print(f"✅ Connecting to PostgreSQL at: {clean_url.split('@')[1] if '@' in clean_url else clean_url}")
        
        _pool = await asyncpg.create_pool(
            dsn=clean_url,
            min_size=2,
            max_size=15,
            ssl='require'
        )
    return _pool

def now_str() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

def parse_dt(s: str) -> dt.datetime:
    try:
        clean_s = s.replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(clean_s)
        if d.tzinfo is not None:
            return d.astimezone().replace(tzinfo=None)
        return d
    except Exception:
        return dt.datetime.now() + dt.timedelta(days=7)

async def init_db():
    pool = await get_pool()
    async with pool.acquire() as conn:
        statements = SCHEMA.strip().split(';')
        for stmt in statements:
            if stmt.strip():
                await conn.execute(stmt.strip())
        
        # Safely add new columns for older databases
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS tier TEXT DEFAULT 'free'")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS ai_questions_used INTEGER DEFAULT 0")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS pdf_uploads_used INTEGER DEFAULT 0")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS usage_reset_date TEXT")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS username TEXT")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS language TEXT DEFAULT 'en'")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS user_type TEXT DEFAULT 'general'")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS grade TEXT")
        
        await conn.execute("ALTER TABLE topics ADD COLUMN IF NOT EXISTS user_id INTEGER DEFAULT 1")
        await conn.execute("ALTER TABLE attempts ADD COLUMN IF NOT EXISTS user_id INTEGER DEFAULT 1")
        await conn.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS user_id INTEGER DEFAULT 1")

# ---------------------------------------------------------------- AUTH
async def create_user(email: str, hashed_pw: str, user_type: str = "general", grade: str = None) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "INSERT INTO users(email, password, created_at, user_type, grade) VALUES ($1, $2, $3, $4, $5) RETURNING id, email",
                    email, hashed_pw, now_str(), user_type, grade)
                return dict(row) if row else None
        except asyncpg.UniqueViolationError:
            return None

async def get_user_by_email(email: str) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM users WHERE lower(email)=lower($1)", email)
        return dict(row) if row else None

async def get_user_by_id(user_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, email, username, language, user_type, grade FROM users WHERE id=$1", user_id)
        return dict(row) if row else None

# ---------------------------------------------------------------- TOPICS
async def ensure_topic(user_id: int, name: str, subject: str = "General") -> dict:
    name = name.strip()
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM topics WHERE user_id=$1 AND lower(name)=lower($2)", user_id, name)
        if row:
            return dict(row)
        async with conn.transaction():
            row = await conn.fetchrow(
                "INSERT INTO topics(user_id, name, subject, score, attempts, updated_at) VALUES ($1, $2, $3, 50.0, 0, $4) RETURNING *",
                user_id, name, subject, now_str())
            return dict(row) if row else None

async def get_topics(user_id: int) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM topics WHERE user_id=$1 ORDER BY name", user_id)
        return [dict(r) for r in rows]

# ---------------------------------------------------------------- ATTEMPTS
async def record_attempt(user_id: int, topic_name: str, correct: int, total: int, source: str) -> dict:
    topic = await ensure_topic(user_id, topic_name)
    pct = round(correct / total * 100, 1) if total else 0.0
    old = topic["score"]
    new = pct if topic["attempts"] == 0 else round(0.55 * old + 0.45 * pct, 1)
    new = max(0.0, min(100.0, new))
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE topics SET score=$1, attempts=attempts+1, updated_at=$2 WHERE id=$3",
                new, now_str(), topic["id"])
            await conn.execute(
                "INSERT INTO attempts(user_id, topic_id, correct, total, pct, source, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                user_id, topic["id"], correct, total, pct, source, now_str())
    return {"name": topic["name"], "old": old, "new": new, "delta": round(new - old, 1), "pct": pct}

async def recent_attempts(user_id: int, limit: int = 20) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT a.pct, a.source, a.created_at, t.name topic "
            "FROM attempts a JOIN topics t ON t.id=a.topic_id "
            "WHERE a.user_id=$1 ORDER BY a.created_at DESC LIMIT $2", user_id, limit)
        return [dict(r) for r in rows]

async def heatmap(user_id: int, days: int = 84) -> list[dict]:
    start = (dt.datetime.now() - dt.timedelta(days=days - 1)).date()
    counts = {start + dt.timedelta(days=i): 0 for i in range(days)}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT SUBSTR(created_at, 1, 10) d, COUNT(*) n FROM attempts "
            "WHERE user_id=$1 AND created_at >= $2 GROUP BY d", user_id, start.isoformat())
        for row in rows:
            key = dt.date.fromisoformat(row["d"])
            if key in counts:
                counts[key] = row["n"]
    return [{"date": k.isoformat(), "count": v} for k, v in counts.items()]

# ---------------------------------------------------------------- STATS
async def get_stats(user_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        topics = [dict(r) for r in await conn.fetch("SELECT * FROM topics WHERE user_id=$1", user_id)]
        total = sum(t["attempts"] for t in topics)
        if topics:
            w_sum = sum((1 + t["attempts"]) for t in topics)
            w_avg = sum(t["score"] * (1 + t["attempts"]) for t in topics) / w_sum
        else:
            w_avg = 0.0
            
        week_ago = (dt.datetime.now() - dt.timedelta(days=7)).isoformat(timespec="seconds")
        attempts_7d = await conn.fetchval("SELECT COUNT(*) FROM attempts WHERE user_id=$1 AND created_at >= $2", user_id, week_ago)
        
        avg_pct_val = await conn.fetchval("SELECT AVG(pct) FROM attempts WHERE user_id=$1", user_id)
        avg_pct = float(avg_pct_val) if avg_pct_val is not None else 0.0
        
        peak = int(round(min(100, max(0, 0.75 * w_avg + 0.25 * min(1.0, attempts_7d / 8) * 100))))
        best = max(topics, key=lambda t: t["score"]) if topics else None
        weak = [t for t in topics if t["score"] < 60]
        
        days_set = {r["d"] for r in await conn.fetch("SELECT DISTINCT SUBSTR(created_at, 1, 10) d FROM attempts WHERE user_id=$1", user_id)}
        streak, day = 0, dt.date.today()
        if day.isoformat() not in days_set:
            day -= dt.timedelta(days=1)
        while day.isoformat() in days_set:
            streak += 1
            day -= dt.timedelta(days=1)
            
        return {
            "peak": peak, "weighted_mastery": round(w_avg, 1),
            "attempts": total, "attempts_7d": attempts_7d,
            "avg_pct": round(avg_pct, 1), "streak": streak,
            "topic_count": len(topics), "weak_count": len(weak),
            "best_topic": best["name"] if best else None,
        }

# ---------------------------------------------------------------- TASKS
async def add_task(user_id: int, title: str, topic: str | None, deadline: str) -> dict:
    if topic:
        await ensure_topic(user_id, topic)
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "INSERT INTO tasks(user_id, title, topic, deadline, created_at) VALUES ($1, $2, $3, $4, $5) RETURNING *",
                user_id, title.strip(), topic or None, deadline, now_str())
            return dict(row) if row else None

async def toggle_task(user_id: int, task_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("UPDATE tasks SET done = 1 - done WHERE id=$1 AND user_id=$2", task_id, user_id)
            row = await conn.fetchrow("SELECT * FROM tasks WHERE id=$1 AND user_id=$2", task_id, user_id)
            return dict(row) if row else {}

async def delete_task(user_id: int, task_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM tasks WHERE id=$1 AND user_id=$2", task_id, user_id)

async def get_schedule(user_id: int) -> dict:
    now = dt.datetime.now()
    pool = await get_pool()
    async with pool.acquire() as conn:
        tasks = [dict(r) for r in await conn.fetch("SELECT * FROM tasks WHERE user_id=$1", user_id)]
        topics = {t["name"].lower(): t for t in [dict(r) for r in await conn.fetch("SELECT * FROM topics WHERE user_id=$1", user_id)]}
        
        queue, cleared = [], []
        for t in tasks:
            dl = parse_dt(t["deadline"])
            hours = (dl - now).total_seconds() / 3600
            topic = topics.get((t.get("topic") or "").lower())
            t["topic_score"] = topic["score"] if topic else None
            if t["done"]:
                t["tier"], t["priority"] = "DONE", 0
                cleared.append(t)
                continue
            deadline_score = 100.0 if hours <= 0 else max(0.0, 100 - hours * (100 / 168))
            weakness = 100 - (topic["score"] if topic else 55)
            t["priority"] = int(round(0.62 * deadline_score + 0.38 * weakness))
            t["tier"] = ("OVERDUE" if hours < 0 else "CRITICAL" if hours < 24 else "HIGH" if hours < 72 else "THIS WEEK" if hours < 168 else "PLANNED")
            queue.append(t)
        queue.sort(key=lambda t: t["priority"], reverse=True)
        cleared.sort(key=lambda t: t["deadline"], reverse=True)
        
        busy = {(t.get("topic") or "").lower() for t in tasks if not t["done"]}
        suggestions = [
            {"topic": tp["name"], "score": tp["score"]}
            for tp in sorted(topics.values(), key=lambda x: x["score"])
            if tp["score"] < 60 and tp["name"].lower() not in busy
        ][:4]
        return {"queue": queue, "cleared": cleared, "suggestions": suggestions}

# ---------------------------------------------------------------- USAGE & PAYWALL
async def get_user_usage(user_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT tier, ai_questions_used, pdf_uploads_used, usage_reset_date FROM users WHERE id=$1", user_id)
        return dict(row) if row else None

async def increment_usage(user_id: int, ai_questions: int = 0, pdf_uploads: int = 0):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("""
                UPDATE users 
                SET ai_questions_used = CASE WHEN usage_reset_date IS NULL OR TO_CHAR(NOW(), 'YYYY-MM') != SUBSTR(usage_reset_date, 1, 7) THEN 0 ELSE ai_questions_used END,
                    pdf_uploads_used = CASE WHEN usage_reset_date IS NULL OR TO_CHAR(NOW(), 'YYYY-MM') != SUBSTR(usage_reset_date, 1, 7) THEN 0 ELSE pdf_uploads_used END,
                    usage_reset_date = TO_CHAR(NOW(), 'YYYY-MM-DD')
                WHERE id=$1
            """, user_id)
            
            if ai_questions > 0:
                await conn.execute("UPDATE users SET ai_questions_used = ai_questions_used + $1 WHERE id=$2", ai_questions, user_id)
            if pdf_uploads > 0:
                await conn.execute("UPDATE users SET pdf_uploads_used = pdf_uploads_used + $1 WHERE id=$2", pdf_uploads, user_id)

async def upgrade_user_tier(user_id: int, tier: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("UPDATE users SET tier=$1 WHERE id=$2", tier, user_id)

async def cleanup_old_pdf_attempts():
    """Deletes PDF quiz attempts older than 14 days to save database space."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            cutoff = (dt.datetime.now() - dt.timedelta(days=14)).isoformat(timespec="seconds")
            await conn.execute("DELETE FROM attempts WHERE source='pdf' AND created_at < $1", cutoff)

# ---------------------------------------------------------------- SETTINGS
async def update_user_settings(user_id: int, username: str, language: str, user_type: str, grade: str | None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("UPDATE users SET username=$1, language=$2, user_type=$3, grade=$4 WHERE id=$5", username, language, user_type, grade, user_id)

async def update_user_password(user_id: int, hashed_pw: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("UPDATE users SET password=$1 WHERE id=$2", hashed_pw, user_id)
