"""
El-Peak — data layer v4 (Multi-User).
"""
import datetime as dt
from contextlib import asynccontextmanager
from pathlib import Path
import aiosqlite

DB_PATH = Path(__file__).parent / "elpeak.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    email      TEXT NOT NULL UNIQUE,
    password   TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS topics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    name       TEXT NOT NULL,
    subject    TEXT DEFAULT 'General',
    score      REAL DEFAULT 50.0,
    attempts   INTEGER DEFAULT 0,
    updated_at TEXT,
    UNIQUE(user_id, name)
);
CREATE TABLE IF NOT EXISTS attempts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    topic_id   INTEGER NOT NULL REFERENCES topics(id),
    correct    INTEGER NOT NULL,
    total      INTEGER NOT NULL,
    pct        REAL NOT NULL,
    source     TEXT DEFAULT 'static',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    title      TEXT NOT NULL,
    topic      TEXT,
    deadline   TEXT NOT NULL,
    done       INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_attempts_created ON attempts(user_id, created_at);
"""

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

@asynccontextmanager
async def db():
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    try:
        yield conn
    finally:
        await conn.close()

async def init_db():
    async with db() as c:
        # Migration check: If the old 'topics' table lacks 'user_id', wipe and recreate
        cur = await c.execute("PRAGMA table_info(topics)")
        columns = [row[1] for row in await cur.fetchall()]
        if "user_id" not in columns:
            print("⚠️ Old database schema detected. Wiping and rebuilding...")
            await c.executescript("DROP TABLE IF EXISTS topics; DROP TABLE IF EXISTS attempts; DROP TABLE IF EXISTS tasks;")
        
        await c.executescript(SCHEMA)
        await c.commit()

# ---------------------------------------------------------------- AUTH
async def create_user(email: str, hashed_pw: str) -> dict:
    async with db() as c:
        try:
            cur = await c.execute(
                "INSERT INTO users(email, password, created_at) VALUES (?,?,?)",
                (email, hashed_pw, now_str()))
            await c.commit()
            return {"id": cur.lastrowid, "email": email}
        except aiosqlite.IntegrityError:
            return None

async def get_user_by_email(email: str) -> dict:
    async with db() as c:
        cur = await c.execute("SELECT * FROM users WHERE lower(email)=lower(?)", (email,))
        row = await cur.fetchone()
        return dict(row) if row else None

# ---------------------------------------------------------------- TOPICS
async def ensure_topic(user_id: int, name: str, subject: str = "General") -> dict:
    name = name.strip()
    async with db() as c:
        cur = await c.execute("SELECT * FROM topics WHERE user_id=? AND lower(name)=lower(?)", (user_id, name))
        row = await cur.fetchone()
        if row:
            return dict(row)
        await c.execute(
            "INSERT INTO topics(user_id, name, subject, score, attempts, updated_at) VALUES (?,?,?,?,?,?)",
            (user_id, name, subject, 50.0, 0, now_str()))
        await c.commit()
        cur = await c.execute("SELECT * FROM topics WHERE user_id=? AND lower(name)=lower(?)", (user_id, name))
        return dict(await cur.fetchone())

async def get_topics(user_id: int) -> list[dict]:
    async with db() as c:
        cur = await c.execute("SELECT * FROM topics WHERE user_id=? ORDER BY name", (user_id,))
        return [dict(r) for r in await cur.fetchall()]

# ---------------------------------------------------------------- ATTEMPTS
async def record_attempt(user_id: int, topic_name: str, correct: int, total: int, source: str) -> dict:
    topic = await ensure_topic(user_id, topic_name)
    pct = round(correct / total * 100, 1) if total else 0.0
    old = topic["score"]
    new = pct if topic["attempts"] == 0 else round(0.55 * old + 0.45 * pct, 1)
    new = max(0.0, min(100.0, new))
    async with db() as c:
        await c.execute(
            "UPDATE topics SET score=?, attempts=attempts+1, updated_at=? WHERE id=?",
            (new, now_str(), topic["id"]))
        await c.execute(
            "INSERT INTO attempts(user_id, topic_id, correct, total, pct, source, created_at) VALUES (?,?,?,?,?,?,?)",
            (user_id, topic["id"], correct, total, pct, source, now_str()))
        await c.commit()
    return {"name": topic["name"], "old": old, "new": new, "delta": round(new - old, 1), "pct": pct}

async def recent_attempts(user_id: int, limit: int = 20) -> list[dict]:
    async with db() as c:
        cur = await c.execute(
            "SELECT a.pct, a.source, a.created_at, t.name topic "
            "FROM attempts a JOIN topics t ON t.id=a.topic_id "
            "WHERE a.user_id=? ORDER BY a.created_at DESC LIMIT ?", (user_id, limit,))
        return [dict(r) for r in await cur.fetchall()]

async def heatmap(user_id: int, days: int = 84) -> list[dict]:
    start = (dt.datetime.now() - dt.timedelta(days=days - 1)).date()
    counts = {start + dt.timedelta(days=i): 0 for i in range(days)}
    async with db() as c:
        cur = await c.execute(
            "SELECT date(created_at) d, COUNT(*) n FROM attempts "
            "WHERE user_id=? AND created_at >= ? GROUP BY d", (user_id, start.isoformat()))
        async for row in cur:
            key = dt.date.fromisoformat(row["d"])
            if key in counts:
                counts[key] = row["n"]
    return [{"date": k.isoformat(), "count": v} for k, v in counts.items()]

# ---------------------------------------------------------------- STATS
async def get_stats(user_id: int) -> dict:
    async with db() as c:
        cur = await c.execute("SELECT * FROM topics WHERE user_id=?", (user_id,))
        topics = [dict(r) for r in await cur.fetchall()]
        total = sum(t["attempts"] for t in topics)
        if topics:
            w_sum = sum((1 + t["attempts"]) for t in topics)
            w_avg = sum(t["score"] * (1 + t["attempts"]) for t in topics) / w_sum
        else:
            w_avg = 0.0
        week_ago = (dt.datetime.now() - dt.timedelta(days=7)).isoformat(timespec="seconds")
        cur = await c.execute("SELECT COUNT(*) n FROM attempts WHERE user_id=? AND created_at >= ?", (user_id, week_ago))
        attempts_7d = (await cur.fetchone())["n"]
        cur = await c.execute("SELECT AVG(pct) a FROM attempts WHERE user_id=?", (user_id,))
        avg_pct = (await cur.fetchone())["a"] or 0.0
        peak = int(round(min(100, max(0, 0.75 * w_avg + 0.25 * min(1.0, attempts_7d / 8) * 100))))
        best = max(topics, key=lambda t: t["score"]) if topics else None
        weak = [t for t in topics if t["score"] < 60]
        
        cur = await c.execute("SELECT DISTINCT date(created_at) d FROM attempts WHERE user_id=?", (user_id,))
        days = {r["d"] for r in await cur.fetchall()}
        streak, day = 0, dt.date.today()
        if day.isoformat() not in days:
            day -= dt.timedelta(days=1)
        while day.isoformat() in days:
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
    async with db() as c:
        cur = await c.execute(
            "INSERT INTO tasks(user_id, title, topic, deadline, created_at) VALUES (?,?,?,?,?)",
            (user_id, title.strip(), topic or None, deadline, now_str()))
        await c.commit()
        row = await (await c.execute("SELECT * FROM tasks WHERE id=?", (cur.lastrowid,))).fetchone()
        return dict(row)

async def toggle_task(user_id: int, task_id: int) -> dict:
    async with db() as c:
        await c.execute("UPDATE tasks SET done = 1 - done WHERE id=? AND user_id=?", (task_id, user_id))
        await c.commit()
        row = await (await c.execute("SELECT * FROM tasks WHERE id=? AND user_id=?", (task_id, user_id))).fetchone()
        return dict(row) if row else {}

async def delete_task(user_id: int, task_id: int):
    async with db() as c:
        await c.execute("DELETE FROM tasks WHERE id=? AND user_id=?", (task_id, user_id))
        await c.commit()

async def get_schedule(user_id: int) -> dict:
    now = dt.datetime.now()
    async with db() as c:
        tasks = [dict(r) for r in await (await c.execute("SELECT * FROM tasks WHERE user_id=?", (user_id,))).fetchall()]
        topics = {t["name"].lower(): t for t in [dict(r) for r in await (await c.execute("SELECT * FROM topics WHERE user_id=?", (user_id,))).fetchall()]}
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
