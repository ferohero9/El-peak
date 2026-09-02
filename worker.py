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
# Round-22 (mirrors main.py — KEEP IN SYNC): GPT-OSS dropped entirely; routes
# now run on the compound systems + qwen3 family:
#   free: en -> groq/compound-mini   ar -> qwen3.6-27b effort=none (+brevity)
#   plus: en -> qwen3.8-27b low      ar -> qwen3.8-27b low
#   pro:  en -> groq/compound        ar -> qwen3.8-27b high (== native xhigh)
# Verified against console.groq.com/docs/compound/systems + /docs/reasoning:
# qwen3.6 accepts only none|default; qwen3.8 adds low|medium|high; compound
# systems take no reasoning params at all.
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_MODEL_FREE = os.getenv("GROQ_MODEL_FREE", "groq/compound-mini")
GROQ_MODEL_PLUS = os.getenv("GROQ_MODEL_PLUS", "qwen/qwen3.8-27b")
GROQ_MODEL_PRO = os.getenv("GROQ_MODEL_PRO", "groq/compound")
GROQ_MODEL_QWEN = os.getenv("GROQ_MODEL_QWEN", "qwen/qwen3.6-27b")
GROQ_EFFORT_PLUS = os.getenv("GROQ_EFFORT_PLUS", "low")
GROQ_EFFORT_PRO = os.getenv("GROQ_EFFORT_PRO", "high")

# Round-19 + Round-22 (Option 3, mirrors main.py): language-routed model
# selection with per-route reasoning_effort.
ROUTE_MODELS = {
    "free": {
        "en": {"model": GROQ_MODEL_FREE, "thinking": False, "effort": None},
        "ar": {"model": GROQ_MODEL_QWEN, "thinking": False, "effort": "none"},
    },
    "plus": {
        "en": {"model": GROQ_MODEL_PLUS, "thinking": False, "effort": GROQ_EFFORT_PLUS},
        "ar": {"model": GROQ_MODEL_PLUS, "thinking": False, "effort": GROQ_EFFORT_PLUS},
    },
    "pro": {
        "en": {"model": GROQ_MODEL_PRO, "thinking": True, "effort": None},
        "ar": {"model": GROQ_MODEL_PLUS, "thinking": True, "effort": GROQ_EFFORT_PRO},
    },
}
ROUTE_FALLBACKS = {
    ("free", "en"): [GROQ_MODEL_FREE, GROQ_MODEL_PLUS, GROQ_MODEL_QWEN],
    ("free", "ar"): [GROQ_MODEL_QWEN, GROQ_MODEL_PLUS, GROQ_MODEL_FREE],
    ("plus", "en"): [GROQ_MODEL_PLUS, GROQ_MODEL_FREE, GROQ_MODEL_QWEN],
    ("plus", "ar"): [GROQ_MODEL_PLUS, GROQ_MODEL_QWEN, GROQ_MODEL_FREE],
    ("pro", "en"): [GROQ_MODEL_PRO, GROQ_MODEL_PLUS, GROQ_MODEL_FREE],
    ("pro", "ar"): [GROQ_MODEL_PLUS, GROQ_MODEL_QWEN, GROQ_MODEL_FREE],
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
    identity: str = None,  # KEEP-IN-SYNC with main.py (unused in the PDF pipeline)
    json_mode: bool = False
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
    # Round-22 (mirrors main.py): strict JSON mode on structured outputs.
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    # Round-19 + Round-22 (Option 3, mirrors main.py): per-family reasoning
    # controls. qwen -> none|default (3.6) / none|default|low|medium|high (3.8);
    # thinking uses reasoning_format "hidden" so thinking content never
    # pollutes message.content (PDF quiz JSON stays parseable — and Groq
    # REQUIRES parsed|hidden when JSON mode is on). compound systems take no
    # reasoning params. gpt-oss -> low|medium|high (env-override legacy).
    model_to_use = payload["model"]
    if "qwen" in model_to_use:
        effort = reasoning_effort or ("default" if thinking else "none")
        if effort == "xhigh":
            effort = "high"
        # qwen3.6 ONLY accepts none|default — sanitize cross-family fallbacks.
        if "3.8" not in model_to_use and effort not in ("none", "default"):
            effort = "default" if effort in ("low", "medium", "high") else "none"
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
    # Round-22 (mirrors main.py): a 400 from a model that rejects an optional
    # param (response_format / reasoning_*) is retried once with a minimal
    # payload so a new Groq model generation can't hard-break PDF quizzes.
    _optional_keys = ("response_format", "reasoning_effort", "reasoning_format",
                      "top_p", "presence_penalty")
    if r.status_code == 400 and any(k in payload for k in _optional_keys):
        minimal = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        logger.warning(
            "Groq 400 on %s with optional params; retrying once with a minimal payload.",
            model_to_use
        )
        r = await client.post(GROQ_URL, headers=headers, json=minimal, timeout=timeout)
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
    lang: str = "en",
    json_mode: bool = False
) -> str:
    tier = (tier or "free").strip().lower()
    # Round-19 + Round-22 (Option 3): resolve language-routed primary,
    # thinking flag, and per-route reasoning_effort.
    lang = (lang or "en").strip().lower()
    if lang not in ("en", "ar"):
        lang = "en"
    tier_routes = ROUTE_MODELS.get(tier, ROUTE_MODELS["free"])
    route = tier_routes.get(lang, tier_routes["en"])
    thinking = route["thinking"]
    route_effort = route.get("effort")
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
                thinking=thinking,
                reasoning_effort=route_effort,
                json_mode=json_mode
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

PDF_TIER_QUESTION_CAPS = {
    "free": 10,
    "plus": 20,
    "pro": 30
}

def _extract_pdf_text_sync(data: bytes, max_pages: int, max_ocr_pages: int) -> dict:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not read that file. Is it a valid PDF? {e}")
    try:
        total_pages = len(doc)
        processed_pages = min(total_pages, max_pages)
        text = ""
        for page in doc[:processed_pages]:
            try:
                text += page.get_text()
            except Exception:
                continue
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 200 and PDF_OCR_ENABLED:
            ocr_text = ""
            for page in doc[:min(total_pages, max_ocr_pages)]:
                try:
                    ocr_text += _ocr_page(page)
                except Exception:
                    continue
            text = re.sub(r"\s+", " ", ocr_text).strip()
        return {
            "text": text,
            "total_pages": total_pages,
            "processed_pages": processed_pages
        }
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
        extraction = await loop.run_in_executor(
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

    text = extraction.get("text", "")
    processed_pages = extraction.get("processed_pages", 1)

    if len(text) < 200:
        return {
            "error": "Not enough readable text in this PDF. If it is scanned, OCR may need better quality."
        }

    tier = (tier or "free").strip().lower()
    tier_cap = PDF_TIER_QUESTION_CAPS.get(tier, 10)

    if not num_questions or num_questions <= 0:
        # Dynamic Content Analysis: evaluate text volume, page count, and content density
        text_len = len(text)
        if text_len > 18000 or processed_pages >= 12:
            computed = 25 + min(5, (text_len - 18000) // 4000)
        elif text_len > 12000 or processed_pages >= 8:
            computed = 18 + min(6, (text_len - 12000) // 1500)
        elif text_len > 6000 or processed_pages >= 4:
            computed = 10 + min(7, (text_len - 6000) // 1000)
        elif text_len > 2500 or processed_pages >= 2:
            computed = 6 + min(3, (text_len - 2500) // 1200)
        else:
            computed = 4
        num_questions = max(3, min(computed, tier_cap))
    else:
        num_questions = max(1, min(num_questions, tier_cap))

    excerpt = text[:18000]
    # Round-20 (mirrors main.py): output language follows the DOCUMENT's script
    # (Arabic PDF -> Arabic quiz) plus a hard anti-Arabizi rule — the model
    # must write Arabic in ARABIC SCRIPT, never Latin transliteration.
    doc_is_ar = is_arabic(excerpt[:2000]) or is_arabic(label)
    # Round-22 (mirrors main.py): free-tier Arabic runs on qwen3.6-27b with its
    # tight 8K TPM / 200K TPD pool — force brevity on free-AR PDF quizzes too.
    route_lang = "ar" if (
        (language or "en").strip().lower() == "ar"
        or doc_is_ar
    ) else "en"
    free_ar_brevity = (tier == "free" and route_lang == "ar")
    if (language or "en").strip().lower() == "ar" or doc_is_ar:
        lang_instruction = (
            "P1: Generate the questions, options, and explanations STRICTLY in Arabic "
            "(Modern Standard Arabic, الفصحى) written in ARABIC SCRIPT, regardless of the "
            "language mix in the document text. "
            "CRITICAL — ARABIC SCRIPT ONLY: never write Arabic words in Latin letters "
            "(no Arabizi / Franco-Arab — write 'كتاب' never 'kitab' or 'ktb'). Latin "
            "characters are allowed ONLY for proper nouns, math symbols, chemical "
            "formulas, and code. "
            "CRITICAL: If the document involves Arabic poetry, literature, or history, you MUST be strictly factual. "
            + (
                f"BREVITY (mandatory): at most {num_questions} questions; every explanation is ONE short "
                "sentence of at most 15 words; keep each context quote under 120 characters."
                if free_ar_brevity else ""
            )
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
                "(max 300 chars) from the text. "
                "QUALITY RULES (mandatory): (1) Each explanation must name the SPECIFIC fact, words, or line "
                "from the excerpt that makes its own answer correct — never a generic template explanation "
                "that could apply to any question (e.g. never 'the repetition of the X sound' unless that "
                "exact sound is in the quoted line). (2) The context quote must be copied VERBATIM from the "
                "excerpt — never paraphrased or invented. (3) Before you finish, re-check every question: "
                "the correct option must actually be derivable from the excerpt, and the explanation must "
                "talk about the same content as the question.\n\nEXCERPT:\n" + excerpt
            )
        }
    ]

    calc_tokens = max(1200, min(8192, num_questions * 220 + 500))
    max_tokens_val = min(calc_tokens, 2400) if free_ar_brevity else calc_tokens

    client = ctx["http"]
    try:
        # Round-19 + Round-22 (Option 3): route by OUTPUT language. Arabic
        # account OR Arabic-script document/label -> Qwen; English docs stay
        # on the compound-mini engine. Pro-EN = groq/compound, Pro-AR =
        # qwen3.8 at effort high (native xhigh).
        raw = await groq_chat_with_fallback(
            client,
            messages=messages,
            tier=tier,
            temperature=0.6,
            max_tokens=max_tokens_val,
            lang=route_lang,
            json_mode=True  # Round-22 rule 2: strict JSON mode on PDF quizzes
        )
        # Round-20: ARABIZI GUARD (mirrors main.py) — if an Arabic quiz comes
        # back in Latin letters, retry ONCE with an escalated instruction.
        if route_lang == "ar" and not is_arabic(raw, 0.08):
            logger.warning(
                "Arabizi guard triggered for PDF '%s' — Latin-script reply, retrying once.",
                label[:60]
            )
            retry_raw = await groq_chat_with_fallback(
                client,
                messages=messages + [{
                    "role": "user",
                    "content": (
                        "IMPORTANT: Your previous reply was rejected because it wrote Arabic using Latin "
                        "letters (Arabizi). Regenerate the FULL quiz with every question, option, and "
                        "explanation written in ARABIC SCRIPT (الحروف العربية) — the same valid JSON schema, "
                        "all content in proper Arabic."
                    )
                }],
                tier=tier,
                temperature=0.6,
                max_tokens=max_tokens_val,
                lang=route_lang,
                json_mode=True
            )
            if is_arabic(retry_raw, 0.08):
                raw = retry_raw
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
