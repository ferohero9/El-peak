"""
El-Peak — FastAPI backend v3.
REST endpoints + Groq (llama-3.3-70b-versatile): AI quizzes, PDF-to-quiz,
AI coach, AI Tutor Chat, with OCR fallback and heuristic fallbacks.
"""
import io
import json
import os
import random
import re
from pathlib import Path

import httpx
import pymupdf as fitz  # PyMuPDF
import pytesseract
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import database as db

# Safe imports for PDF/OCR
try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    PDF_OCR_ENABLED = True
except ImportError:
    print("⚠️ Warning: PyMuPDF/pytesseract/Pillow not installed. PDF features disabled.")
    PDF_OCR_ENABLED = False

# Load .env file safely
env_path = Path(__file__).parent / "key.env"
try:
    from dotenv import load_dotenv
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print("✅ key.env file found and loaded successfully.")
    else:
        print("❌ key.env file NOT found! I am looking for it here:", env_path)
except ImportError:
    print("❌ python-dotenv not installed! Run: pip install python-dotenv")

app = FastAPI(title="El-Peak API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

INDEX = Path(__file__).parent / "templates" / "index.html"

@app.on_event("startup")
async def startup():
    await db.init_db()
    if os.getenv("GROQ_API_KEY"):
        print("✅ GROQ_API_KEY found in environment!")
    else:
        print("❌ GROQ_API_KEY NOT found! Make sure your .env file has a line like: GROQ_API_KEY=gsk_...")

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(INDEX)

# ================================================================ schemas
class SubmitIn(BaseModel):
    topic_name: str = Field(min_length=1)
    correct: int = Field(ge=0)
    total: int = Field(ge=1)
    source: str = "static"

class AIQuizIn(BaseModel):
    topic: str = Field(min_length=1)
    num_questions: int = Field(default=4, ge=1, le=30)  # Updated to allow 1-30
    difficulty: str = "mixed"

class TaskIn(BaseModel):
    title: str = Field(min_length=1)
    topic: str | None = None
    deadline: str = Field(min_length=10)

class AskIn(BaseModel):
    question: str = Field(min_length=1)
    history: list = []

# ================================================================ dashboard
@app.get("/api/stats")
async def stats(): return await db.get_stats()

@app.get("/api/topics")
async def topics(): return await db.get_topics()

@app.get("/api/recent")
async def recent(limit: int = 20): return await db.recent_attempts(limit)

@app.get("/api/heatmap")
async def heatmap(): return await db.heatmap()

@app.get("/manifest.json", include_in_schema=False)
async def manifest():
    return JSONResponse({
        "name": "El-Peak Study Compass",
        "short_name": "El-Peak",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#070710",
        "theme_color": "#070710",
        "icons": [
            {
                "src": "https://api.iconify.design/fluent/flash-24-filled.png?color=%236366f1",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "https://api.iconify.design/fluent/flash-24-filled.png?color=%236366f1",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    })
# ================================================================ quiz engine
STATIC_QUIZZES = {
    "Mathematics": [
        {"question": "What is the derivative of x³?", "options": ["3x²", "x²", "3x", "x³/3"], "answer_index": 0, "explanation": "Power rule: d/dx(xⁿ) = n·xⁿ⁻¹, so 3·x²."},
        {"question": "Evaluate ∫ 2x dx.", "options": ["x² + C", "2x² + C", "x + C", "2 + C"], "answer_index": 0, "explanation": "Reverse power rule: ∫2x dx = x² + C."},
        {"question": "Solve x² − 5x + 6 = 0.", "options": ["x = 2, 3", "x = −2, −3", "x = 1, 6", "x = 0, 5"], "answer_index": 0, "explanation": "Factors as (x−2)(x−3) = 0."},
        {"question": "lim(x→0) sin(x)/x = ?", "options": ["1", "0", "∞", "undefined"], "answer_index": 0, "explanation": "A foundational limit: sin(x)/x → 1 as x → 0."},
        {"question": "Slope of the tangent to y = x² at x = 2?", "options": ["4", "2", "8", "1"], "answer_index": 0, "explanation": "dy/dx = 2x; at x=2 the slope is 4."},
    ],
    "Physics": [
        {"question": "The SI unit of force is the…", "options": ["Newton", "Joule", "Watt", "Pascal"], "answer_index": 0, "explanation": "1 N = 1 kg·m/s². Joule is energy, Watt power, Pascal pressure."},
        {"question": "Earth's gravitational acceleration ≈ ?", "options": ["9.8 m/s²", "9.8 m/s", "3.0 m/s²", "12.4 m/s²"], "answer_index": 0, "explanation": "g ≈ 9.81 m/s² at sea level."},
        {"question": "Kinetic energy is given by…", "options": ["½mv²", "mv", "mgh", "ma"], "answer_index": 0, "explanation": "KE = ½mv²; mgh is gravitational potential energy."},
        {"question": "Newton's First Law describes…", "options": ["Inertia", "Action–reaction", "F = ma", "Universal gravitation"], "answer_index": 0, "explanation": "Bodies maintain their state of motion unless acted on by a net force."},
        {"question": "Speed of light in vacuum ≈ ?", "options": ["3×10⁸ m/s", "3×10⁶ m/s", "3×10¹⁰ m/s", "9.8 m/s"], "answer_index": 0, "explanation": "c ≈ 299,792,458 m/s ≈ 3×10⁸ m/s."},
    ],
    "Computer Science": [
        {"question": "Big-O of binary search on a sorted array?", "options": ["O(log n)", "O(n)", "O(n log n)", "O(1)"], "answer_index": 0, "explanation": "Each step halves the search space → logarithmic."},
        {"question": "Which structure is FIFO?", "options": ["Queue", "Stack", "Tree", "Hash map"], "answer_index": 0, "explanation": "First-In-First-Out: a queue. A stack is LIFO."},
        {"question": "HTTP 404 means…", "options": ["Not Found", "Server Error", "Unauthorized", "Redirect"], "answer_index": 0, "explanation": "404 = resource not found; 500 = server error; 401/403 = auth."},
        {"question": "Which keyword defines a function in Python?", "options": ["def", "func", "fn", "lambda only"], "answer_index": 0, "explanation": "`def name():` defines a named function."},
        {"question": "Which SQL statement retrieves data?", "options": ["SELECT", "INSERT", "UPDATE", "ALTER"], "answer_index": 0, "explanation": "SELECT queries rows; the others modify schema/data."},
    ],
    "History": [
        {"question": "World War II ended in…", "options": ["1945", "1944", "1946", "1939"], "answer_index": 0, "explanation": "Europe surrendered May 1945; Japan in September 1945."},
        {"question": "The first crewed Moon landing was in…", "options": ["1969", "1965", "1972", "1959"], "answer_index": 0, "explanation": "Apollo 11, July 20 1969."},
        {"question": "The Renaissance began in…", "options": ["Italy", "France", "England", "Spain"], "answer_index": 0, "explanation": "It started in Italian city-states like Florence in the 14th c."},
        {"question": "First President of the United States?", "options": ["George Washington", "Thomas Jefferson", "John Adams", "Lincoln"], "answer_index": 0, "explanation": "Washington served 1789–1797."},
        {"question": "The Berlin Wall fell in…", "options": ["1989", "1991", "1985", "1993"], "answer_index": 0, "explanation": "November 9, 1989."},
    ],
    "Biology": [
        {"question": "The 'powerhouse of the cell' is the…", "options": ["Mitochondrion", "Nucleus", "Ribosome", "Golgi body"], "answer_index": 0, "explanation": "Mitochondria produce ATP via cellular respiration."},
        {"question": "Plants convert light energy via…", "options": ["Photosynthesis", "Respiration", "Fermentation", "Osmosis"], "answer_index": 0, "explanation": "6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂."},
        {"question": "DNA's structure is a…", "options": ["Double helix", "Single strand", "Triple ring", "Sheet"], "answer_index": 0, "explanation": "Watson & Crick (1953): two antiparallel strands."},
        {"question": "Red blood cells primarily carry…", "options": ["Oxygen", "Hormones", "Antibodies", "Glucose"], "answer_index": 0, "explanation": "Hemoglobin in RBCs binds and transports O₂."},
        {"question": "The largest organ of the human body is the…", "options": ["Skin", "Liver", "Brain", "Lungs"], "answer_index": 0, "explanation": "Skin ≈ 1.5–2 m²; the liver is the largest internal organ."},
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
        subject = random.choice(list(STATIC_QUIZZES))
    bank = STATIC_QUIZZES[subject]
    qs = random.sample(bank, min(n, len(bank)))
    return {"topic": topic, "subject": subject, "questions": qs, "ai": False, "source": "static"}

@app.get("/api/quiz/subjects")
async def quiz_subjects(): return {"subjects": list(STATIC_QUIZZES.keys())}

@app.get("/api/quiz/static")
async def quiz_static(topic: str = "Mathematics", n: int = 4):
    return _match_static(topic, max(3, min(5, n)))

def _clean_questions(data, limit: int) -> list:
    qs = data.get("questions", data if isinstance(data, list) else [])
    clean = []
    for q in qs[:limit]:
        opts = [str(o) for o in q.get("options", [])][:6]
        try: idx = int(q.get("answer_index", 0))
        except Exception: continue
        if q.get("question") and len(opts) >= 3 and 0 <= idx < len(opts):
            clean.append({
                "question": str(q["question"]),
                "options": opts,
                "answer_index": idx,
                "explanation": str(q.get("explanation", "—")),
                "context": str(q.get("context", ""))
            })
    return clean

QUIZ_SCHEMA = ('{"questions":[{"question":"...","options":["A","B","C","D"],"answer_index":0,"explanation":"...","context":"exact quote from text if pdf"}]}')

@app.post("/api/quiz/ai")
async def quiz_ai(payload: AIQuizIn):
    messages = [
        {"role": "system", "content": "You are El-Peak's quiz engine. Reply with STRICT JSON only — no prose, no markdown — matching exactly this schema: " + QUIZ_SCHEMA + ". Vary the correct answer position."},
        {"role": "user", "content": f"Generate exactly {payload.num_questions} {payload.difficulty} multiple-choice questions about '{payload.topic}' for a student. Each must have 4 plausible options and a one-sentence explanation."},
    ]
    try:
        # Increase token limit dynamically for larger quizzes
        max_tokens = max(1200, payload.num_questions * 250)
        raw = await groq_chat(messages, temperature=0.7, max_tokens=max_tokens)
        clean = _clean_questions(extract_json(raw), payload.num_questions)
        if not clean: raise ValueError("empty question set")
        return {"topic": payload.topic, "subject": "AI", "questions": clean, "ai": True, "source": "ai"}
    except Exception as e:
        print(f"⚠️ AI Quiz Error: {e}")
        fallback = _match_static(payload.topic, 4)
        fallback["note"] = "AI unavailable — served from the static bank."
        return fallback

@app.post("/api/quiz/pdf")
async def quiz_pdf(file: UploadFile = File(...), difficulty: str = Form("mixed"), topic: str = Form(""), num_questions: int = Form(0)):
    if not PDF_OCR_ENABLED:
        raise HTTPException(500, "PDF libraries not installed on server.")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024: raise HTTPException(413, "PDF too large (max 15 MB).")
    
    text = ""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        for page in doc[:40]: text += page.get_text()
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 200:
            text = ""
            for page in doc[:10]:
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text += pytesseract.image_to_string(img)
            text = re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        raise HTTPException(422, f"Could not read that file. Is it a valid PDF? ({str(e)})")

    if len(text) < 200: raise HTTPException(422, "Not enough readable text in this PDF (even with OCR).")

    # Auto-determine number of questions based on text length
    if not num_questions or num_questions == 0:
        if len(text) > 6000: auto_n = 5
        elif len(text) > 2500: auto_n = 4
        else: auto_n = 3
        num_questions = auto_n

    label = topic.strip() or Path(file.filename or "upload.pdf").stem
    excerpt = text[:9000]
    messages = [
        {"role": "system", "content": "You are El-Peak's quiz engine. Reply with STRICT JSON only — no prose, no markdown — matching exactly this schema: " + QUIZ_SCHEMA + ". Vary the correct answer position."},
        {"role": "user", "content": f"Using ONLY the excerpt below from the student's document '{label}', write exactly {num_questions} {difficulty} multiple-choice questions. Make them smart: test conceptual understanding, application, or inference rather than just rote memorization. Each must have 4 plausible options, a one-sentence explanation, and a 'context' field containing an EXACT quote (max 300 chars) from the text.\n\nEXCERPT:\n" + excerpt},
    ]
    try:
        raw = await groq_chat(messages, temperature=0.6, max_tokens=2000)
        clean = _clean_questions(extract_json(raw), num_questions)
        if not clean: raise ValueError("empty question set")
        return {"topic": label, "subject": "PDF", "questions": clean, "ai": True, "source": "pdf"}
    except Exception as e:
        print(f"⚠️ AI PDF Quiz Error: {e}")
        raise HTTPException(503, "AI is unavailable right now — PDF quizzes need the AI engine. Use the Static Bank meanwhile.")

@app.post("/api/quiz/submit")
async def quiz_submit(payload: SubmitIn):
    if payload.correct > payload.total: raise HTTPException(422, "correct cannot exceed total")
    result = await db.record_attempt(payload.topic_name, payload.correct, payload.total, payload.source)
    return {"saved": True, "topic": result}

# ================================================================ scheduler
@app.get("/api/schedule")
async def schedule(): return await db.get_schedule()

@app.post("/api/tasks")
async def create_task(payload: TaskIn): return await db.add_task(payload.title, payload.topic, payload.deadline)

@app.patch("/api/tasks/{task_id}")
async def toggle(task_id: int): return await db.toggle_task(task_id)

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
    if not weak: weak = sorted(topics, key=lambda t: t["score"])[:2]
    pcts = [a["pct"] for a in recent]
    trend = (sum(pcts[:3]) / max(1, len(pcts[:3])) - sum(pcts[3:]) / max(1, len(pcts[3:]))) if len(pcts) >= 4 else 0
    ctx = "\n".join([
        "Weak topics (mastery %):",
        *[f"- {t['name']} ({t['subject']}): {int(t['score'])}%" for t in weak],
        "Upcoming deadlines:",
        *([f"- {t['title']} [{t.get('tier')}] due {t['deadline']}" for t in sched["queue"][:3]] or ["- none"]),
        f"Recent accuracy trend: {'rising' if trend > 0 else 'falling'} ({round(trend, 1)} pts).",
    ])
    try:
        raw = await groq_chat([
            {"role": "system", "content": "You are El-Peak, a sharp, encouraging study coach. Reply with STRICT JSON only, no markdown: {\"headline\":str,\"focus\":[{\"topic\":str,\"why\":str,\"drill\":str}],\"tips\":[4 strings],\"plan\":[{\"day\":str,\"task\":str} with exactly 7 entries],\"motivation\":str}"},
            {"role": "user", "content": f"Build a 7-day revision plan from this student data:\n{ctx}"},
        ], temperature=0.6, max_tokens=1200)
        data = extract_json(raw)
        data.setdefault("focus", []); data.setdefault("tips", []); data.setdefault("plan", [])
        data["ai"] = True
        return data
    except Exception as e:
        print(f"⚠️ Coach Fallback Triggered: {e}")
        return heuristic_coach(weak, trend)

def heuristic_coach(weak, trend) -> dict:
    focus = [{"topic": t["name"], "why": f"Mastery at {int(t['score'])}% — below your 65% safety line.", "drill": f"15-min mixed drill on {t['name']}, then a 5-question quiz to lock it in."} for t in weak[:3]]
    plan, days = [], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, day in enumerate(days):
        if weak:
            t = weak[i % len(weak)]
            task = (f"Drill '{t['name']}' — 25 min focus block + mini-quiz" if i % 2 == 0 else f"Retest '{t['name']}' and review wrong answers")
        else: task = "Maintenance set across your strongest topics."
        plan.append({"day": day, "task": task})
    return {
        "headline": f"Summit plan: shore up {len(weak)} weak ridge(s) before deadlines hit.",
        "focus": focus,
        "tips": ["Attack your weakest topic first each session — willpower is freshest early.", "Use spaced repetition: retest every weak topic 48h after drilling it.", "Interleave subjects within a session instead of marathon-blocking one.", "Simulate pressure: time your quizzes to match real exam conditions."],
        "plan": plan,
        "motivation": "Your trend line is data, not destiny. Every rep on a weak topic raises the whole range." if trend <= 0 else "Momentum is on your side — keep the streak alive and push one tier higher.",
        "ai": False,
    }

# ================================================================ AI Tutor Chat
@app.post("/api/ask")
async def ask_tutor(payload: AskIn):
    messages = [
        {"role": "system", "content": "You are El-Peak, a friendly, sharp AI study tutor. Explain concepts clearly. Use markdown for formatting (bold, italics, bullet points, code blocks). Keep answers concise but thorough. Be encouraging!"},
        *payload.history,
        {"role": "user", "content": payload.question}
    ]
    try:
        reply = await groq_chat(messages, temperature=0.5, max_tokens=1000)
        return {"reply": reply}
    except Exception as e:
        print(f"⚠️ Tutor Chat Error: {e}")
        raise HTTPException(503, "El-Peak is thinking too hard right now. Try again in a moment.")

# ================================================================ Groq core
async def groq_chat(messages: list, temperature: float = 0.6, max_tokens: int = 1200) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY. Did you create a .env file?")
    
    payload = {"model": GROQ_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(GROQ_URL, headers=headers, json=payload)
        if r.status_code != 200:
            try:
                err_data = r.json()
                err_msg = err_data.get("error", {}).get("message", str(err_data))
            except Exception:
                err_msg = r.text
            print(f"\n❌ GROQ API ERROR ({r.status_code}): {err_msg}\n")
            raise RuntimeError(f"Groq API Error: {err_msg}")
            
        return r.json()["choices"][0]["message"]["content"]

def extract_json(text: str):
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence: candidates.insert(0, fence.group(1).strip())
    for c in candidates:
        try: return json.loads(c)
        except Exception: pass
        lo, hi = c.find("{"), c.rfind("}")
        if lo != -1 and hi > lo:
            try: return json.loads(c[lo:hi + 1])
            except Exception: pass
    raise ValueError("No parseable JSON in model response")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
