"""
El-Peak — FastAPI backend.
REST endpoints + Groq (llama-3.3-70b-versatile) integration with
graceful heuristic/static fallbacks when the AI is unavailable.
"""
import json
import os
import random
import re

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import database as db

try:                                   # optional .env support
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="El-Peak API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# OpenRouter alternative:
#   GROQ_URL   = "https://openrouter.ai/api/v1/chat/completions"
#   GROQ_MODEL = "qwen/qwen-2.5-72b-instruct:free"


@app.on_event("startup")
async def startup():
    await db.init_db()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ================================================================ schemas
class SubmitIn(BaseModel):
    topic_name: str = Field(min_length=1)
    correct: int = Field(ge=0)
    total: int = Field(ge=1)
    source: str = "static"


class AIQuizIn(BaseModel):
    topic: str = Field(min_length=1)
    num_questions: int = Field(default=4, ge=3, le=5)
    difficulty: str = "mixed"


class TaskIn(BaseModel):
    title: str = Field(min_length=1)
    topic: str | None = None
    deadline: str = Field(min_length=10)


# ================================================================ dashboard
@app.get("/api/stats")
async def stats():
    return await db.get_stats()


@app.get("/api/topics")
async def topics():
    return await db.get_topics()


@app.get("/api/recent")
async def recent(limit: int = 20):
    return await db.recent_attempts(limit)


@app.get("/api/heatmap")
async def heatmap():
    return await db.heatmap()


# ================================================================ quiz engine
STATIC_QUIZZES = {
    "Mathematics": [
        {"question": "What is the derivative of x³?",
         "options": ["3x²", "x²", "3x", "x³/3"], "answer_index": 0,
         "explanation": "Power rule: d/dx(xⁿ) = n·xⁿ⁻¹, so 3·x²."},
        {"question": "Evaluate ∫ 2x dx.",
         "options": ["x² + C", "2x² + C", "x + C", "2 + C"], "answer_index": 0,
         "explanation": "Reverse power rule: ∫2x dx = x² + C."},
        {"question": "Solve x² − 5x + 6 = 0.",
         "options": ["x = 2, 3", "x = −2, −3", "x = 1, 6", "x = 0, 5"], "answer_index": 0,
         "explanation": "Factors as (x−2)(x−3) = 0."},
        {"question": "lim(x→0) sin(x)/x = ?",
         "options": ["1", "0", "∞", "undefined"], "answer_index": 0,
         "explanation": "A foundational limit: sin(x)/x → 1 as x → 0."},
        {"question": "Slope of the tangent to y = x² at x = 2?",
         "options": ["4", "2", "8", "1"], "answer_index": 0,
         "explanation": "dy/dx = 2x; at x=2 the slope is 4."},
    ],
    "Physics": [
        {"question": "The SI unit of force is the…",
         "options": ["Newton", "Joule", "Watt", "Pascal"], "answer_index": 0,
         "explanation": "1 N = 1 kg·m/s². Joule is energy, Watt power, Pascal pressure."},
        {"question": "Earth's gravitational acceleration ≈ ?",
         "options": ["9.8 m/s²", "9.8 m/s", "3.0 m/s²", "12.4 m/s²"], "answer_index": 0,
         "explanation": "g ≈ 9.81 m/s² at sea level."},
        {"question": "Kinetic energy is given by…",
         "options": ["½mv²", "mv", "mgh", "ma"], "answer_index": 0,
         "explanation": "KE = ½mv²; mgh is gravitational potential energy."},
        {"question": "Newton's First Law describes…",
         "options": ["Inertia", "Action–reaction", "F = ma", "Universal gravitation"],
         "answer_index": 0,
         "explanation": "Bodies maintain their state of motion unless acted on by a net force."},
        {"question": "Speed of light in vacuum ≈ ?",
         "options": ["3×10⁸ m/s", "3×10⁶ m/s", "3×10¹⁰ m/s", "9.8 m/s"], "answer_index": 0,
         "explanation": "c ≈ 299,792,458 m/s ≈ 3×10⁸ m/s."},
    ],
    "Computer Science": [
        {"question": "Big-O of binary search on a sorted array?",
         "options": ["O(log n)", "O(n)", "O(n log n)", "O(1)"], "answer_index": 0,
         "explanation": "Each step halves the search space → logarithmic."},
        {"question": "Which structure is FIFO?",
         "options": ["Queue", "Stack", "Tree", "Hash map"], "answer_index": 0,
         "explanation": "First-In-First-Out: a queue. A stack is LIFO."},
        {"question": "HTTP 404 means…",
         "options": ["Not Found", "Server Error", "Unauthorized", "Redirect"],
         "answer_index": 0,
         "explanation": "404 = resource not found; 500 = server error; 401/403 = auth."},
        {"question": "Which keyword defines a function in Python?",
         "options": ["def", "func", "fn", "lambda only"], "answer_index": 0,
         "explanation": "`def name():` defines a named function."},
        {"question": "Which SQL statement retrieves data?",
         "options": ["SELECT", "INSERT", "UPDATE", "ALTER"], "answer_index": 0,
         "explanation": "SELECT queries rows; the others modify schema/data."},
    ],
    "History": [
        {"question": "World War II ended in…",
         "options": ["1945", "1944", "1946", "1939"], "answer_index": 0,
         "explanation": "Europe surrendered May 1945; Japan in September 1945."},
        {"question": "The first crewed Moon landing was in…",
         "options": ["1969", "1965", "1972", "1959"], "answer_index": 0,
         "explanation": "Apollo 11, July 20 1969."},
        {"question": "The Renaissance began in…",
         "options": ["Italy", "France", "England", "Spain"], "answer_index": 0,
         "explanation": "It started in Italian city-states like Florence in the 14th c."},
        {"question": "First President of the United States?",
         "options": ["George Washington", "Thomas Jefferson", "John Adams", "Lincoln"],
         "answer_index": 0,
         "explanation": "Washington served 1789–1797."},
        {"question": "The Berlin Wall fell in…",
         "options": ["1989", "1991", "1985", "1993"], "answer_index": 0,
         "explanation": "November 9, 1989."},
    ],
    "Biology": [
        {"question": "The 'powerhouse of the cell' is the…",
         "options": ["Mitochondrion", "Nucleus", "Ribosome", "Golgi body"], "answer_index": 0,
         "explanation": "Mitochondria produce ATP via cellular respiration."},
        {"question": "Plants convert light energy via…",
         "options": ["Photosynthesis", "Respiration", "Fermentation", "Osmosis"],
         "answer_index": 0,
         "explanation": "6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂."},
        {"question": "DNA's structure is a…",
         "options": ["Double helix", "Single strand", "Triple ring", "Sheet"],
         "answer_index": 0,
         "explanation": "Watson & Crick (1953): two antiparallel strands."},
        {"question": "Red blood cells primarily carry…",
         "options": ["Oxygen", "Hormones", "Antibodies", "Glucose"], "answer_index": 0,
         "explanation": "Hemoglobin in RBCs binds and transports O₂."},
        {"question": "The largest organ of the human body is the…",
         "options": ["Skin", "Liver", "Brain", "Lungs"], "answer_index": 0,
         "explanation": "Skin ≈ 1.5–2 m²; the liver is the largest internal organ."},
    ],
}


def _match_static(topic: str, n: int):
    key = topic.strip().lower()
    subject = None
    for name in STATIC_QUIZZES:
        if name.lower() == key or name.lower() in key or key in name.lower():
            subject = name
            break
    if subject is None:
        tw, kw = set(key.split()), None
        for name in STATIC_QUIZZES:
            if tw & set(name.lower().split()):
                kw = name
                break
        subject = kw or random.choice(list(STATIC_QUIZZES))
    bank = STATIC_QUIZZES[subject]
    qs = random.sample(bank, min(n, len(bank)))
    return {"topic": topic, "subject": subject, "questions": qs, "ai": False}


@app.get("/api/quiz/static")
async def quiz_static(topic: str = "Mathematics", n: int = 4):
    return _match_static(topic, max(3, min(5, n)))


@app.post("/api/quiz/ai")
async def quiz_ai(payload: AIQuizIn):
    schema = ('{"questions":[{"question":"...","options":["A","B","C","D"],'
              '"answer_index":0,"explanation":"..."}]}')
    messages = [
        {"role": "system",
         "content": "You are El-Peak's quiz engine. Reply with STRICT JSON only — "
                    "no prose, no markdown — matching exactly this schema: "
                    f"{schema}. Vary the correct answer position."},
        {"role": "user",
         "content": f"Generate exactly {payload.num_questions} {payload.difficulty} "
                    f"multiple-choice questions about '{payload.topic}' for a student. "
                    "Each must have 4 plausible options and a one-sentence explanation."},
    ]
    try:
        raw = await groq_chat(messages, temperature=0.7)
        data = extract_json(raw)
        qs = data.get("questions", data if isinstance(data, list) else [])
        clean = []
        for q in qs[: payload.num_questions]:
            opts = [str(o) for o in q.get("options", [])][:6]
            idx = int(q.get("answer_index", 0))
            if q.get("question") and len(opts) >= 3 and 0 <= idx < len(opts):
                clean.append({"question": str(q["question"]), "options": opts,
                              "answer_index": idx,
                              "explanation": str(q.get("explanation", "—"))})
        if not clean:
            raise ValueError("empty question set")
        return {"topic": payload.topic, "subject": "AI", "questions": clean, "ai": True}
    except Exception:
        fallback = _match_static(payload.topic, payload.num_questions)
        fallback["note"] = "AI unavailable — served from the static bank."
        return fallback


@app.post("/api/quiz/submit")
async def quiz_submit(payload: SubmitIn):
    if payload.correct > payload.total:
        raise HTTPException(422, "correct cannot exceed total")
    result = await db.record_attempt(payload.topic_name, payload.correct,
                                     payload.total, payload.source)
    return {"saved": True, "topic": result}


# ================================================================ scheduler
@app.get("/api/schedule")
async def schedule():
    return await db.get_schedule()


@app.post("/api/tasks")
async def create_task(payload: TaskIn):
    return await db.add_task(payload.title, payload.topic, payload.deadline)


@app.patch("/api/tasks/{task_id}")
async def toggle(task_id: int):
    return await db.toggle_task(task_id)


@app.delete("/api/tasks/{task_id}")
async def remove(task_id: int):
    await db.delete_task(task_id)
    return {"deleted": task_id}


# ================================================================ AI coach
@app.get("/api/coach")
async def coach():
    topics = await db.get_topics()
    sched = await db.get_schedule()
    recent = await db.recent_attempts(6)

    weak = sorted([t for t in topics if t["score"] < 65], key=lambda t: t["score"])[:4]
    if not weak:
        weak = sorted(topics, key=lambda t: t["score"])[:2]

    pcts = [a["pct"] for a in recent]
    trend = (sum(pcts[:3]) / max(1, len(pcts[:3])) -
             sum(pcts[3:]) / max(1, len(pcts[3:]))) if len(pcts) >= 4 else 0

    ctx = "\n".join([
        "Weak topics (mastery %):",
        *[f"- {t['name']} ({t['subject']}): {int(t['score'])}%" for t in weak],
        "Upcoming deadlines:",
        *([f"- {t['title']} [{t.get('tier')}] due {t['deadline']}"
           for t in sched["queue"][:3]] or ["- none"]),
        f"Recent accuracy trend: {'rising' if trend > 0 else 'falling'} ({round(trend,1)} pts).",
    ])

    try:
        raw = await groq_chat([
            {"role": "system",
             "content": "You are El-Peak, a sharp, encouraging study coach. Reply with "
                        'STRICT JSON only, no markdown: {"headline":str,"focus":'
                        '[{"topic":str,"why":str,"drill":str}],"tips":[4 strings],"plan":'
                        '[{"day":str,"task":str} with exactly 7 entries],"motivation":str}'},
            {"role": "user",
             "content": f"Build a 7-day revision plan from this student data:\n{ctx}"},
        ], temperature=0.6, max_tokens=1200)
        data = extract_json(raw)
        data.setdefault("focus", [])
        data.setdefault("tips", [])
        data.setdefault("plan", [])
        data["ai"] = True
        return data
    except Exception:
        return heuristic_coach(weak, sched, trend)


def heuristic_coach(weak, sched, trend) -> dict:
    """Rule-based fallback so the Coach always works, even offline."""
    focus = [{
        "topic": t["name"],
        "why": f"Mastery at {int(t['score'])}% — below your 65% safety line.",
        "drill": f"15-min mixed drill on {t['name']}, then a 5-question quiz to lock it in.",
    } for t in weak[:3]]

    plan, days = [], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, day in enumerate(days):
        if weak:
            t = weak[i % len(weak)]
            task = (f"Drill '{t['name']}' — 25 min focus block + mini-quiz"
                    if i % 2 == 0 else
                    f"Retest '{t['name']}' and review wrong answers")
        else:
            task = "Maintenance set across your strongest topics."
        plan.append({"day": day, "task": task})

    return {
        "headline": f"Summit plan: shore up {len(weak)} weak ridge(s) before deadlines hit.",
        "focus": focus,
        "tips": [
            "Attack your weakest topic first each session — willpower is freshest early.",
            "Use spaced repetition: retest every weak topic 48h after drilling it.",
            "Interleave subjects within a session instead of marathon-blocking one.",
            "Simulate pressure: time your quizzes to match real exam conditions.",
        ],
        "plan": plan,
        "motivation": ("Your trend line is data, not destiny. "
                       "Every rep on a weak topic raises the whole range.")
        if trend <= 0 else
        "Momentum is on your side — keep the streak alive and push one tier higher.",
        "ai": False,
    }


# ================================================================ Groq core
async def groq_chat(messages: list, temperature: float = 0.6,
                    max_tokens: int = 1200) -> str:
    key = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("No AI API key configured")
    payload = {"model": GROQ_MODEL, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(GROQ_URL, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def extract_json(text: str):
    """Robustly pull JSON out of an LLM reply (handles fences and chatter)."""
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            pass
        lo, hi = c.find("{"), c.rfind("}")
        if lo != -1 and hi > lo:
            try:
                return json.loads(c[lo:hi + 1])
            except Exception:
                pass
    raise ValueError("No parseable JSON in model response")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)