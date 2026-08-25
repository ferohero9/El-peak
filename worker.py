"""
El-Peak — ARQ Background Worker (P2.5)
Processes CPU-heavy PDF extraction + OCR + Groq quiz generation off the
web dyno. The web app enqueues `process_pdf_quiz` and polls for results;
this process executes the job.

Run with:
    arq worker.WorkerSettings

Required env (same as web app):
    DATABASE_URL  — Neon pooled connection string
    REDIS_URL     — shared Redis (ARQ queue + results)
    GROQ_API_KEY

Optional env:
    WORKER_MAX_JOBS     — concurrent jobs per worker process (default 2;
                          this replaces the old in-web-process semaphore)
    WORKER_JOB_TIMEOUT  — seconds before a job is killed (default 300)

Design notes:
- This file is intentionally self-contained (own Groq helpers, own PDF
  helpers) so the worker process does not import the FastAPI app.
- CPU-bound fitz/tesseract work runs in a thread executor so the worker's
  event loop stays responsive.
- Quota accounting (pdf_uploads + usage events) happens HERE, after a
  successful generation, matching the old synchronous flow.
"""
import io
import os
import re
import json
import shutil
import logging
import asyncio
from typing import Any, List

import httpx
from arq.connections import RedisSettings

import database as db

logger = logging.getLogger("el-peak.worker")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Round-13 (re-verified live against console.groq.com/docs/models + /deprecations):
# current Production text models are openai/gpt-oss-120b, openai/gpt-oss-20b,
# qwen/qwen3.6-27b. Groq announced llama-3.1-8b-instant + llama-3.3-70b-versatile
# deprecation on 2026-06-17 (shutdown 2026-08-16) — those ids are REMOVED from
# the fallback chain. qwen3.6-27b added as cross-family tertiary fallback.
# Round-19 (Option 3, mirrors main.py — KEEP IN SYNC): language-routed tiers:
#   free: en -> gpt-oss-20b      ar -> qwen3.6-27b non-thinking
#   plus: en -> gpt-oss-120b     ar -> qwen3.6-27b thinking
#   pro:  all -> qwen3.6-27b thinking
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MODEL_FREE = os.getenv("GROQ_MODEL_FREE", "openai/gpt-oss-20b")
GROQ_MODEL_PLUS = os.getenv("GROQ_MODEL_PLUS", "openai/gpt-oss-120b")
GROQ_MODEL_QWEN = os.getenv("GROQ_MODEL_QWEN", "qwen/qwen3.6-27b")

# Round-19 (Option 3): language-routed model selection (mirrors main.py).
ROUTE_MODELS = {
    "free": {
        "en": {"model": GROQ_MODEL_FREE, "thinking": False},
        "ar": {"model": GROQ_MODEL_QWEN, "thinking": False},
    },
    "plus": {
        "en": {"model": GROQ_MODEL_PLUS, "thinking": False},
        "ar": {"model": GROQ_MODEL_QWEN, "thinking": True},
    },
    "pro": {
        "en": {"model": GROQ_MODEL_QWEN, "thinking": True},
        "ar": {"model": GROQ_MODEL_QWEN, "thinking": True},
    },
}
ROUTE_FALLBACKS = {
    ("free", "en"): [GROQ_MODEL_FREE, GROQ_MODEL_PLUS, GROQ_MODEL_QWEN],
    ("free", "ar"): [GROQ_MODEL_QWEN, GROQ_MODEL_FREE, GROQ_MODEL_PLUS],
    ("plus", "en"): [GROQ_MODEL_PLUS, GROQ_MODEL_FREE, GROQ_MODEL_QWEN],
    ("plus", "ar"): [GROQ_MODEL_QWEN, GROQ_MODEL_PLUS, GROQ_MODEL_FREE],
    ("pro", "en"): [GROQ_MODEL_QWEN, GROQ_MODEL_PLUS, GROQ_MODEL_FREE],
    ("pro", "ar"): [GROQ_MODEL_QWEN, GROQ_MODEL_PLUS, GROQ_MODEL_FREE],
}

# Round-19: Arabic-script detection (mirrors main.py is_arabic).
_ARABIC_RANGES = (
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
)

def is_arabic(text: str, threshold: float = 0.25) -> bool:
    if not text:
        return False
    letters = 0
    arabic = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            o = ord(ch)
            if any(lo <= o <= hi for lo, hi in _ARABIC_RANGES):
                arabic += 1
    return letters > 0 and (arabic / letters) >= threshold

QUIZ_SCHEMA = (
    '{"questions":[{"question": "...", "options":["A","B","C","D"],'
    '"answer_index":0, "explanation": "...", "context": "exact quote from text if pdf"}]}'
)

# Optional PDF dependencies
try:
    import pymupdf as fitz
    import pytesseract
    from PIL import Image
    PDF_IMPORT_OK = True
except Exception:
    fitz = None
    pytesseract = None
    Image = None
    PDF_IMPORT_OK = False

PDF_TESSERACT_OK = shutil.which("tesseract") is not None
PDF_OCR_ENABLED = PDF_IMPORT_OK and PDF_TESSERACT_OK

# -----------------------------
# GROQ HELPERS
# -----------------------------
def _unique_models(models: List[str]) -> List[str]:
    seen = set()
    out = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    return out

async def groq_chat(
    client: httpx.AsyncClient,
    messages: List[dict],
    temperature: float = 0.6,
    max_tokens: int = 1200,
    model_override: str = None,
    timeout: int = 60,
    thinking: bool = False,
    reasoning_effort: str = None,
    identity: str = None  # KEEP-IN-SYNC with main.py (unused in the PDF pipeline)
) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY.")
    # Round-19b (mirrors main.py): append an honest-identity system message
    # when asked to. The PDF quiz pipeline never passes this, but the mirror
    # stays signature-identical to main.py's groq_chat.
    model_to_use_pre = model_override or GROQ_MODEL
    if identity:
        messages = list(messages) + [{
            "role": "system",
            "content": (
                f"IDENTITY: You are {identity} inside the El-Peak study app, running as "
                f"the model '{model_to_use_pre}' on Groq. If the user asks who you are, what "
                "AI/model/chatbot you are, or whether you are ChatGPT/GPT/Qwen or similar, "
                f"answer honestly and briefly: you are {identity} in El-Peak, powered by "
                f"'{model_to_use_pre}' running on Groq. NEVER claim to be ChatGPT, GPT, or "
                "any other company's product."
            )
        }]
    payload = {
        "model": model_override or GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    # Round-19 (Option 3, mirrors main.py): per-family reasoning controls.
    # qwen -> reasoning_effort none|default; thinking uses reasoning_format
# "hidden" so thinking content never pollutes message.content (PDF quiz JSON
    # stays parseable). gpt-oss -> low|medium|high (platform default if unset).
    model_to_use = payload["model"]
    if "qwen" in model_to_use:
        effort = reasoning_effort or ("default" if thinking else "none")
        payload["reasoning_effort"] = effort
        if effort != "none":
            payload["reasoning_format"] = "hidden"
            payload["top_p"] = 0.95
            payload["max_tokens"] = min(16384, max_tokens + 2500)
        else:
            payload["top_p"] = 0.80
            payload["presence_penalty"] = 1.5
    elif "gpt-oss" in model_to_use and reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    r = await client.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        try:
            err = r.json()
            err_msg = err.get("error", {}).get("message", str(err))
        except Exception:
            err_msg = r.text
        logger.error("GROQ API ERROR %s: %s", r.status_code, err_msg)
        raise RuntimeError(f"Groq API Error: {err_msg}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

async def groq_chat_with_fallback(
    client: httpx.AsyncClient,
    messages: List[dict],
    tier: str,
    temperature: float = 0.6,
    max_tokens: int = 1200,
    lang: str = "en"
) -> str:
    tier = (tier or "free").strip().lower()
    # Round-19 (Option 3): resolve the language-routed primary + thinking flag.
    lang = (lang or "en").strip().lower()
    if lang not in ("en", "ar"):
        lang = "en"
    tier_routes = ROUTE_MODELS.get(tier, ROUTE_MODELS["free"])
    route = tier_routes.get(lang, tier_routes["en"])
    thinking = route["thinking"]
    models = [route["model"]]
    models.extend(ROUTE_FALLBACKS.get((tier, lang), [GROQ_MODEL]))
    models = _unique_models(models)
    last_error = None
    for model in models:
        try:
            return await groq_chat(
                client,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model_override=model,
                thinking=thinking
            )
        except Exception as e:
            last_error = e
            logger.warning("Groq model %s failed: %s", model, str(e))
            continue
    raise last_error or RuntimeError("Groq unavailable.")

# -----------------------------
# JSON / QUESTION CLEANING
# -----------------------------
def extract_json(text: str) -> Any:
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        lo = candidate.find("{")
        hi = candidate.rfind("}")
        if lo != -1 and hi > lo:
            try:
                return json.loads(candidate[lo:hi + 1])
            except Exception:
                pass
    raise ValueError("No parseable JSON in model response")

def _clean_questions(data: Any, limit: int) -> List[dict]:
    if isinstance(data, list):
        qs = data
    elif isinstance(data, dict):
        qs = data.get("questions", [])
    else:
        qs = []
    clean = []
    for q in qs[:limit]:
        if not isinstance(q, dict):
            continue
        question = str(q.get("question", "")).strip()
        raw_options = q.get("options", [])
        if not isinstance(raw_options, list):
            continue
        options = []
        for opt in raw_options[:6]:
            opt_str = str(opt).strip()
            if opt_str:
                options.append(opt_str)
        if not question or len(options) < 3:
            continue
        answer_index = None
        try:
            answer_index = int(q.get("answer_index", 0))
        except Exception:
            answer_index = None
        if answer_index is None or answer_index < 0 or answer_index >= len(options):
            answer_text = str(q.get("answer", "")).strip().lower()
            found = False
            if answer_text:
                for i, opt in enumerate(options):
                    if opt.strip().lower() == answer_text:
                        answer_index = i
                        found = True
                        break
            if not found:
                answer_index = 0
        explanation = str(q.get("explanation", "—")).strip() or "—"
        context = str(q.get("context", "")).strip()
        clean.append({
            "question": question,
            "options": options,
            "answer_index": answer_index,
            "explanation": explanation,
            "context": context
        })
    return clean

# -----------------------------
# PDF / OCR (CPU-bound, run in executor)
# -----------------------------
def _ocr_page(page, dpi: int = 130) -> str:
    if not PDF_OCR_ENABLED:
        return ""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img, lang="ara+eng")
    except Exception:
        try:
            return pytesseract.image_to_string(img)
        except Exception:
            return ""

def _extract_pdf_text_sync(data: bytes, max_pages: int, max_ocr_pages: int) -> str:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not read that file. Is it a valid PDF? {e}")
    try:
        text = ""
        for page in doc[:min(len(doc), max_pages)]:
            try:
                text += page.get_text()
            except Exception:
                continue
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 200 and PDF_OCR_ENABLED:
            ocr_text = ""
            for page in doc[:min(len(doc), max_ocr_pages)]:
                try:
                    ocr_text += _ocr_page(page)
                except Exception:
                    continue
            text = re.sub(r"\s+", " ", ocr_text).strip()
        return text
    finally:
        try:
            doc.close()
        except Exception:
            pass

# -----------------------------
# ARQ JOB
# -----------------------------
async def process_pdf_quiz(
    ctx,
    user_id: int,
    data_hex: str,
    difficulty: str,
    label: str,
    num_questions: int,
    language: str,
    grade: str,
    tier: str,
    max_pages: int,
    max_ocr_pages: int
):
    """
    Argument order MUST match main.py's enqueue_job call:
    (user_id, data_hex, difficulty, label, num_questions, language, grade,
     tier, max_pages, max_ocr_pages)
    Returns a dict; on failure returns {"error": "..."} which the status
    endpoint surfaces to the client.
    """
    if not PDF_IMPORT_OK:
        return {"error": "PDF libraries are not installed on the worker."}

    try:
        data = bytes.fromhex(data_hex)
    except Exception:
        return {"error": "Corrupted upload payload."}

    # CPU-bound work off the event loop
    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(
            None,
            _extract_pdf_text_sync,
            data,
            max_pages,
            max_ocr_pages
        )
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error("PDF extraction crashed for user %s: %s", user_id, e)
        return {"error": "PDF extraction failed."}

    if len(text) < 200:
        return {
            "error": "Not enough readable text in this PDF. If it is scanned, OCR may need better quality."
        }

    if not num_questions or num_questions <= 0:
        if len(text) > 6000:
            num_questions = 5
        elif len(text) > 2500:
            num_questions = 4
        else:
            num_questions = 3
    num_questions = min(num_questions, 5)

    excerpt = text[:12000]
    if (language or "en").strip().lower() == "ar":
        lang_instruction = (
            "P1: The student's account language is Arabic. Generate the questions, options, and explanations "
            "STRICTLY in Arabic, regardless of the language the PDF/document text is written in. "
            "CRITICAL: If the document involves Arabic poetry, literature, or history, you MUST be strictly factual."
        )
    else:
        lang_instruction = "Generate in English."

    messages = [
        {
            "role": "system",
            "content": (
                "You are El-Peak's quiz engine. Reply with STRICT JSON only — no prose, no markdown — "
                f"matching exactly this schema: {QUIZ_SCHEMA}. "
                "Vary the correct answer position. "
                f"{lang_instruction}"
            )
        },
        {
            "role": "user",
            "content": (
                f"Using ONLY the excerpt below from the student's document '{label}', write exactly "
                f"{num_questions} {difficulty} multiple-choice questions. "
                "Make them smart: test conceptual understanding, application, or inference rather than just rote memorization. "
                "Each must have 4 plausible options, a one-sentence explanation, and a 'context' field containing an EXACT quote "
                "(max 300 chars) from the text.\n\nEXCERPT:\n" + excerpt
            )
        }
    ]

    client = ctx["http"]
    try:
        # Round-19 (Option 3): route by OUTPUT language. Arabic account OR
        # Arabic-script document/label -> Qwen (best Arabic + math); English
        # docs stay on the fast GPT-OSS engine. Pro = Qwen deep-thinking.
        route_lang = "ar" if (
            (language or "en").strip().lower() == "ar"
            or is_arabic(excerpt[:2000])
            or is_arabic(label)
        ) else "en"
        raw = await groq_chat_with_fallback(
            client,
            messages=messages,
            tier=tier,
            temperature=0.6,
            max_tokens=1600,
            lang=route_lang
        )
        clean = _clean_questions(extract_json(raw), num_questions)
    except Exception as e:
        logger.error("PDF quiz Groq error for user %s: %s", user_id, e)
        return {"error": "AI is unavailable right now. Please try again."}

    if not clean:
        return {"error": "Empty question set from PDF AI."}

    session_id = await db.create_quiz_session(
        user_id=user_id,
        topic=label,
        source="pdf",
        difficulty=difficulty,
        questions=clean
    )
    # Quota accounting happens here, after success (same as old sync flow).
    await db.increment_usage(user_id, pdf_uploads=1)
    await db.log_usage_event(user_id, "pdf_upload", 1, tier)

    return {
        "topic": label,
        "subject": "PDF",
        "difficulty": difficulty,
        "questions": clean,
        "ai": True,
        "source": "pdf",
        "session_id": session_id
    }

# -----------------------------
# ARQ LIFECYCLE + SETTINGS
# -----------------------------
async def startup(ctx):
    await db.init_db()
    ctx["http"] = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
    )
    logger.info("ARQ worker started.")

async def shutdown(ctx):
    await ctx["http"].aclose()
    await db.close_pool()
    logger.info("ARQ worker stopped.")

class WorkerSettings:
    functions = [process_pdf_quiz]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_URL", "redis://localhost:6379"))
    # Concurrency cap per worker process (replaces the old web-process semaphore).
    max_jobs = int(os.getenv("WORKER_MAX_JOBS", "2"))
    job_timeout = int(os.getenv("WORKER_JOB_TIMEOUT", "300"))
    # Keep results for 1 hour so the polling endpoint can retrieve them.
    keep_result = 3600
