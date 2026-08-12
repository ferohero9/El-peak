"""
El-Peak — data layer.
Async SQLite (aiosqlite): schema, seeding, mastery updates,
Peak Score, scheduler ranking, streaks and heatmap aggregation.
"""
import datetime as dt
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent / "elpeak.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    subject    TEXT DEFAULT 'General',
    score      REAL DEFAULT 50.0,        -- mastery 0..100 (EMA-updated)
    attempts   INTEGER DEFAULT 0,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS attempts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id   INTEGER NOT NULL REFERENCES topics(id),
    correct    INTEGER NOT NULL,
    total      INTEGER NOT NULL,
    pct        REAL NOT NULL,
    source     TEXT DEFAULT 'static',    -- 'static' | 'ai'
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    topic      TEXT,
    deadline   TEXT NOT NULL,
    done       INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_attempts_created ON attempts(created_at);
"""


def now_str() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def parse_dt(s: str) -> dt.datetime:
    try:
        # Handle 'Z' (UTC) from JavaScript and convert to local naive time
        clean_s = s.replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(clean_s)
        if d.tzinfo is not None:
            # Convert to local time and remove tzinfo so it matches dt.datetime.now()
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


# ---------------------------------------------------------------- lifecycle
async def init_db():
    async with db() as c:
        await c.executescript(SCHEMA)
        cur = await c.execute("SELECT COUNT(*) n FROM topics")
        if (await cur.fetchone())["n"] == 0:
            await _seed(c)
        await c.commit()


async def _seed(c):
    """First-run demo data so the dashboard is alive immediately."""
    seed_topics = [
        ("Limits & Derivatives", "Mathematics", 46),
        ("Integration", "Mathematics", 38),
        ("Newtonian Mechanics", "Physics", 63),
        ("Python Fundamentals", "Computer Science", 84),
        ("World War II", "History", 57),
        ("Cell Biology", "Biology", 72),
    ]
    ids = []
    for name, subj, score in seed_topics:
        cur = await c.execute(
            "INSERT INTO topics(name, subject, score, attempts, updated_at) "
            "VALUES (?,?,?,?,?)", (name, subj, score, 0, now_str()))
        ids.append(cur.lastrowid)

    # (days_ago, topic_index, pct) — ~3 weeks of activity for the heatmap
    history = [
        (34, 3, 70), (32, 0, 40), (29, 2, 60), (27, 3, 85), (25, 1, 35),
        (22, 4, 55), (20, 3, 90), (18, 0, 50), (16, 5, 70), (13, 2, 65),
        (11, 1, 45), (9, 3, 80), (7, 5, 75), (6, 0, 52), (4, 2, 70),
        (3, 4, 60), (2, 3, 95), (1, 1, 50), (0, 0, 58),
    ]
    for days_ago, ti, pct in history:
        when = (dt.datetime.now() - dt.timedelta(days=days_ago, hours=3))
        await c.execute(
            "INSERT INTO attempts(topic_id, correct, total, pct, source, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (ids[ti], round(pct / 20), 5, pct, "static",
             when.isoformat(timespec="seconds")))
    for ti in set(h[1] for h in history):
        n = sum(1 for h in history if h[1] == ti)
        await c.execute("UPDATE topics SET attempts=? WHERE id=?", (n, ids[ti]))

    now = dt.datetime.now()
    tasks = [
        ("Finish integration problem set", "Integration", now + dt.timedelta(days=1, hours=2)),
        ("Mechanics mock exam", "Newtonian Mechanics", now + dt.timedelta(days=3)),
        ("WWII essay outline", "World War II", now + dt.timedelta(days=6)),
        ("Submit Python mini-project", "Python Fundamentals", now - dt.timedelta(hours=14)),
    ]
    for title, topic, dl in tasks:
        await c.execute(
            "INSERT INTO tasks(title, topic, deadline, created_at) VALUES (?,?,?,?)",
            (title, topic, dl.isoformat(timespec="seconds"), now_str()))


# ---------------------------------------------------------------- topics
async def ensure_topic(name: str, subject: str = "General") -> dict:
    name = name.strip()
    async with db() as c:
        cur = await c.execute("SELECT * FROM topics WHERE lower(name)=lower(?)", (name,))
        row = await cur.fetchone()
        if row:
            return dict(row)
        await c.execute(
            "INSERT INTO topics(name, subject, score, attempts, updated_at) "
            "VALUES (?,?,?,?,?)", (name, subject, 50.0, 0, now_str()))
        await c.commit()
        cur = await c.execute("SELECT * FROM topics WHERE lower(name)=lower(?)", (name,))
        return dict(await cur.fetchone())


async def get_topics() -> list[dict]:
    async with db() as c:
        cur = await c.execute("SELECT * FROM topics ORDER BY name")
        return [dict(r) for r in await cur.fetchall()]


# ---------------------------------------------------------------- attempts
async def record_attempt(topic_name: str, correct: int, total: int, source: str) -> dict:
    """Log a quiz, update topic mastery via exponential moving average."""
    topic = await ensure_topic(topic_name)
    pct = round(correct / total * 100, 1) if total else 0.0
    old = topic["score"]
    new = pct if topic["attempts"] == 0 else round(0.55 * old + 0.45 * pct, 1)
    new = max(0.0, min(100.0, new))
    async with db() as c:
        await c.execute(
            "UPDATE topics SET score=?, attempts=attempts+1, updated_at=? WHERE id=?",
            (new, now_str(), topic["id"]))
        await c.execute(
            "INSERT INTO attempts(topic_id, correct, total, pct, source, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (topic["id"], correct, total, pct, source, now_str()))
        await c.commit()
    return {"name": topic["name"], "old": old, "new": new,
            "delta": round(new - old, 1), "pct": pct}


async def recent_attempts(limit: int = 20) -> list[dict]:
    async with db() as c:
        cur = await c.execute(
            "SELECT a.pct, a.source, a.created_at, t.name topic "
            "FROM attempts a JOIN topics t ON t.id=a.topic_id "
            "ORDER BY a.created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in await cur.fetchall()]


async def heatmap(days: int = 84) -> list[dict]:
    start = (dt.datetime.now() - dt.timedelta(days=days - 1)).date()
    counts = {start + dt.timedelta(days=i): 0 for i in range(days)}
    async with db() as c:
        cur = await c.execute(
            "SELECT date(created_at) d, COUNT(*) n FROM attempts "
            "WHERE created_at >= ? GROUP BY d", (start.isoformat(),))
        async for row in cur:
            key = dt.date.fromisoformat(row["d"])
            if key in counts:
                counts[key] = row["n"]
    return [{"date": k.isoformat(), "count": v} for k, v in counts.items()]


# ---------------------------------------------------------------- stats
async def get_stats() -> dict:
    async with db() as c:
        cur = await c.execute("SELECT * FROM topics")
        topics = [dict(r) for r in await cur.fetchall()]
        total = sum(t["attempts"] for t in topics)
        if topics:
            w_sum = sum((1 + t["attempts"]) for t in topics)
            w_avg = sum(t["score"] * (1 + t["attempts"]) for t in topics) / w_sum
        else:
            w_avg = 0.0
        week_ago = (dt.datetime.now() - dt.timedelta(days=7)).isoformat(timespec="seconds")
        cur = await c.execute("SELECT COUNT(*) n FROM attempts WHERE created_at >= ?", (week_ago,))
        attempts_7d = (await cur.fetchone())["n"]
        cur = await c.execute("SELECT AVG(pct) a FROM attempts")
        avg_pct = (await cur.fetchone())["a"] or 0.0
        peak = int(round(min(100, max(0, 0.75 * w_avg + 0.25 * min(1.0, attempts_7d / 8) * 100))))
        best = max(topics, key=lambda t: t["score"]) if topics else None
        weak = [t for t in topics if t["score"] < 60]
        # streak: consecutive study-days ending today (or yesterday, grace)
        cur = await c.execute("SELECT DISTINCT date(created_at) d FROM attempts")
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


# ---------------------------------------------------------------- tasks
async def add_task(title: str, topic: str | None, deadline: str) -> dict:
    if topic:
        await ensure_topic(topic)
    async with db() as c:
        cur = await c.execute(
            "INSERT INTO tasks(title, topic, deadline, created_at) VALUES (?,?,?,?)",
            (title.strip(), topic or None, deadline, now_str()))
        await c.commit()
        row = await (await c.execute("SELECT * FROM tasks WHERE id=?", (cur.lastrowid,))).fetchone()
        return dict(row)


async def toggle_task(task_id: int) -> dict:
    async with db() as c:
        await c.execute("UPDATE tasks SET done = 1 - done WHERE id=?", (task_id,))
        await c.commit()
        row = await (await c.execute("SELECT * FROM tasks WHERE id=?", (task_id,))).fetchone()
        return dict(row) if row else {}


async def delete_task(task_id: int):
    async with db() as c:
        await c.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        await c.commit()


async def get_schedule() -> dict:
    """Priority queue: 62% deadline urgency + 38% topic weakness."""
    now = dt.datetime.now()
    async with db() as c:
        tasks = [dict(r) for r in await (await c.execute("SELECT * FROM tasks")).fetchall()]
        topics = {t["name"].lower(): t for t in
                  [dict(r) for r in await (await c.execute("SELECT * FROM topics")).fetchall()]}
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
            t["tier"] = ("OVERDUE" if hours < 0 else
                         "CRITICAL" if hours < 24 else
                         "HIGH" if hours < 72 else
                         "THIS WEEK" if hours < 168 else "PLANNED")
            queue.append(t)
        queue.sort(key=lambda t: t["priority"], reverse=True)
        cleared.sort(key=lambda t: t["deadline"], reverse=True)
        # Auto-suggestions: weak topics with no open task
        busy = {(t.get("topic") or "").lower() for t in tasks if not t["done"]}
        suggestions = [
            {"topic": tp["name"], "score": tp["score"]}
            for tp in sorted(topics.values(), key=lambda x: x["score"])
            if tp["score"] < 60 and tp["name"].lower() not in busy
        ][:4]
        return {"queue": queue, "cleared": cleared, "suggestions": suggestions}