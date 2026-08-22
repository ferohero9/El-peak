"""
El-Peak — FastAPI backend v12.3 (P2 final)
Production-oriented, safer, more reliable, improved quiz engine,
better payments, better AI handling, better PDF pipeline.

P2 changes:
- Rate limits, AI cache, and single-flight locks moved to Redis (cache.py)
  so the app is fully stateless and safe across multiple instances.
- PDF upload converted to an async ARQ background job: the endpoint
  returns a job_id immediately; clients poll /api/quiz/pdf/status/{job_id}.
- ARQ redis pool created/closed in lifespan.
- Neon Postgres: no code changes needed; use Neon's pooled connection
  string as DATABASE_URL (see P2.3 notes).
"""
import os
import re
import json
import time
import hmac
import hashlib
import logging
import secrets
import shutil
import random
import asyncio
import dataclasses
import datetime as dt
from pathlib import Path
from typing import Annotated, Optional, List, Any
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, status, Request, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
import jwt
import bcrypt
from itsdangerous import URLSafeSerializer, BadSignature

# P1.4: Observability
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from prometheus_fastapi_instrumentator import Instrumentator

# P2.5: background job queue
from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job, JobStatus

import database as db
import cache

# Optional PDF dependencies (kept in web process only for /health reporting;
# actual extraction runs in worker.py)
try:
    import pymupdf as fitz  # noqa: F401
    import pytesseract  # noqa: F401
    from PIL import Image  # noqa: F401
    PDF_IMPORT_OK = True
except Exception:
    PDF_IMPORT_OK = False

PDF_TESSERACT_OK = shutil.which("tesseract") is not None
PDF_OCR_ENABLED = PDF_IMPORT_OK and PDF_TESSERACT_OK

BASE_DIR = Path(__file__).resolve().parent

# Load local env files if present
env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = BASE_DIR / "key.env"
try:
    from dotenv import load_dotenv
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("el-peak")

# -----------------------------
# CONFIG
# -----------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
DEBUG = os.getenv("DEBUG", "true" if ENVIRONMENT != "production" else "false").strip().lower() in ("1", "true", "yes", "on")

# P0.5: SECRET_KEY must exist and be strong in production; no insecure fallback.
SECRET_KEY = os.getenv("SECRET_KEY")
if ENVIRONMENT == "production":
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY must be set in production.")
    if SECRET_KEY == "el_peak_local_dev_secret_change_me" or len(SECRET_KEY) < 32:
        raise RuntimeError("SECRET_KEY is weak in production. Use a random value of at least 32 characters.")
else:
    if not SECRET_KEY:
        SECRET_KEY = "el_peak_local_dev_secret_change_me"
        logger.warning("SECRET_KEY not set. Using insecure development key.")
ALGORITHM = "HS256"

# P0.5: explicit allowlist; wildcard only permitted outside production.
_origins_raw = os.getenv("ALLOWED_ORIGINS", "").strip()
if _origins_raw:
    ALLOWED_ORIGINS = [o.strip() for o in _origins_raw.split(",") if o.strip()]
else:
    if ENVIRONMENT == "production":
        raise RuntimeError("ALLOWED_ORIGINS must be set to an explicit comma-separated allowlist in production.")
    ALLOWED_ORIGINS = ["*"]
    logger.warning("ALLOWED_ORIGINS unset; using wildcard (development only).")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PAYMOB_API_KEY = os.getenv("PAYMOB_API_KEY")
PAYMOB_INTEGRATION_ID = os.getenv("PAYMOB_INTEGRATION_ID", "12345")
PAYMOB_IFRAME_ID = os.getenv("PAYMOB_IFRAME_ID", "12345")
PAYMOB_HMAC_SECRET = os.getenv("PAYMOB_HMAC_SECRET", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv(
    "REDIRECT_URI",
    "https://el-peak-production.up.railway.app/api/auth/google/callback"
)
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "15"))

# P0.5: simulated payments can never be enabled in production.
ALLOW_SIMULATED_PAYMENTS_RAW = os.getenv("ALLOW_SIMULATED_PAYMENTS")
if ENVIRONMENT == "production":
    ALLOW_SIMULATED_PAYMENTS = False
    if ALLOW_SIMULATED_PAYMENTS_RAW is not None and ALLOW_SIMULATED_PAYMENTS_RAW.strip().lower() in ("1", "true", "yes", "on"):
        logger.error("ALLOW_SIMULATED_PAYMENTS is set but ignored in production.")
else:
    if ALLOW_SIMULATED_PAYMENTS_RAW is None:
        ALLOW_SIMULATED_PAYMENTS = True
    else:
        ALLOW_SIMULATED_PAYMENTS = ALLOW_SIMULATED_PAYMENTS_RAW.strip().lower() in ("1", "true", "yes", "on")

# P0.5: anonymous auth throttle. 12s minimum interval ~= 5 attempts/min per IP.
AUTH_RATE_LIMIT_SECONDS = int(os.getenv("AUTH_RATE_LIMIT_SECONDS", "12"))

# P2.2 / P2.5: Redis for cross-instance state and the ARQ job queue.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# P1.1: cache TTLs
AI_QUIZ_CACHE_TTL = 7 * 24 * 3600
COACH_CACHE_TTL = 3600

INDEX = BASE_DIR / "index.html"
if not INDEX.exists():
    INDEX = BASE_DIR / "templates" / "index.html"

session_serializer = URLSafeSerializer(SECRET_KEY, salt="oauth")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# -----------------------------
# TIER CONFIG
# -----------------------------
TIER_LIMITS = {
    "free": {
        "ai_questions": 30,
        "pdf_uploads": 3,
        "pdf_max_mb": 1,
        "pdf_max_pages": 20,
        "pdf_max_ocr_pages": 5,
        "max_ai_questions_per_request": 5
    },
    "plus": {
        "ai_questions": 120,
        "pdf_uploads": 15,
        "pdf_max_mb": 15,
        "pdf_max_pages": 40,
        "pdf_max_ocr_pages": 10,
        "max_ai_questions_per_request": 5
    },
    "pro": {
        "ai_questions": 350,
        "pdf_uploads": 45,
        "pdf_max_mb": 15,
        "pdf_max_pages": 60,
        "pdf_max_ocr_pages": 15,
        "max_ai_questions_per_request": 5
    }
}
TIER_MODELS = {
    "free": "llama-3.1-8b-instant",
    "plus": "openai/gpt-oss-20b",
    "pro": "llama-3.3-70b-versatile"
}
# P1.2: preserved fallback chain acts as our circuit breaker.
TIER_FALLBACK_MODELS = {
    "free": [
        "llama-3.1-8b-instant"
    ],
    "plus": [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant"
    ],
    "pro": [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant"
    ]
}
TIER_PRICES = {
    "plus": 9900,
    "pro": 19900
}

# -----------------------------
# SHARED HTTP CLIENT (P0.3)
# -----------------------------
HTTP_CLIENT: httpx.AsyncClient | None = None

def _build_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0
        ),
    )

async def init_http_client():
    global HTTP_CLIENT
    if HTTP_CLIENT is None:
        HTTP_CLIENT = _build_http_client()

async def close_http_client():
    global HTTP_CLIENT
    if HTTP_CLIENT is not None:
        await HTTP_CLIENT.aclose()
        HTTP_CLIENT = None

def get_http_client() -> httpx.AsyncClient:
    global HTTP_CLIENT
    if HTTP_CLIENT is None:
        HTTP_CLIENT = _build_http_client()
    return HTTP_CLIENT

# -----------------------------
# ARQ POOL (P2.5)
# -----------------------------
ARQ_POOL = None

async def init_arq_pool():
    global ARQ_POOL
    # Fail fast when Redis is absent (e.g. local dev without Redis): default
    # arq settings retry ~5 times with long timeouts and stall app startup.
    # With ARQ_POOL None the PDF endpoint still answers with its explicit 503.
    settings = RedisSettings.from_dsn(REDIS_URL)
    settings = dataclasses.replace(settings, conn_timeout=3, conn_retries=1, conn_retry_delay=1)
    try:
        ARQ_POOL = await create_pool(settings)
        logger.info("ARQ redis pool connected (%s).", REDIS_URL)
    except Exception as e:
        logger.error("Could not connect ARQ pool to Redis (%s). PDF jobs will be rejected.", e)
        ARQ_POOL = None

async def close_arq_pool():
    global ARQ_POOL
    if ARQ_POOL is not None:
        await ARQ_POOL.close()
        ARQ_POOL = None

# -----------------------------
# HELPERS
# -----------------------------
def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False

def create_access_token(data: dict, days: int = 30) -> str:
    to_encode = data.copy()
    expire = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=days)
    to_encode.update({
        "exp": expire,
        "iat": dt.datetime.now(dt.timezone.utc),
        "type": "access"
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def effective_tier_from_user(user: dict) -> str:
    tier = (user.get("tier") or "free").strip().lower()
    if tier == "free":
        return "free"
    expires_raw = user.get("subscription_expires_at")
    subscription_status = (user.get("subscription_status") or "").strip().lower()
    if not expires_raw:
        # Backwards compatibility for older paid users without expiry.
        return tier
    try:
        expires = db.parse_dt(expires_raw)
        now = dt.datetime.utcnow()
        if expires >= now:
            return tier
        return "free"
    except Exception:
        if subscription_status == "active":
            return tier
        return "free"

def normalize_grade(value: Optional[str]) -> str:
    if not value:
        return "general"
    v = str(value).strip().lower()
    if v in ("g1-6", "primary", "primary (1-6)", "primary (1 - 6)", "p1-6"):
        return "g1-6"
    if v in ("g7-9", "preparatory", "preparatory (7-9)", "preparatory (7 - 9)", "prep"):
        return "g7-9"
    if v in ("g10-12", "secondary", "secondary (10-12)", "secondary (10 - 12)", "sec"):
        return "g10-12"
    if "primary" in v or "1-6" in v:
        return "g1-6"
    if "preparatory" in v or "7-9" in v:
        return "g7-9"
    if "secondary" in v or "10-12" in v:
        return "g10-12"
    return "general"

def grade_difficulty_key(grade: str, difficulty: str) -> str:
    grade = normalize_grade(grade)
    difficulty = (difficulty or "mixed").strip().lower()
    mapping = {
        "g1-6": {
            "easy": "1-2",
            "mixed": "3-4",
            "hard": "5-6"
        },
        "g7-9": {
            "easy": "7",
            "mixed": "8",
            "hard": "9"
        },
        "g10-12": {
            "easy": "10",
            "mixed": "11",
            "hard": "12"
        },
        "general": {
            "easy": "easy",
            "mixed": "mixed",
            "hard": "hard"
        }
    }
    return mapping.get(grade, mapping["general"]).get(difficulty, "mixed")

def _unique_models(models: List[str]) -> List[str]:
    seen = set()
    out = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    return out

def _paymob_configured() -> bool:
    try:
        integration_id = int(PAYMOB_INTEGRATION_ID)
        iframe_id = int(PAYMOB_IFRAME_ID)
    except Exception:
        return False
    return bool(
        PAYMOB_API_KEY and
        integration_id != 12345 and
        iframe_id != 12345
    )

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")

def _hmac_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value == True else "false"
    return str(value)

def client_ip(request: Request) -> str:
    """P0.5: respect Railway/LB proxy header, fall back to socket peer."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        first = fwd.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"

# P1.1 / P2.2: cache key builders (moved out of database.py so main.py is
# self-contained; hashing is pure, no DB needed).
def make_ai_quiz_cache_key(topic: str, grade: str, difficulty: str, n: int, lang: str) -> str:
    normalized_topic = (topic or "").strip().lower()
    raw = f"quiz|{normalized_topic}|{grade}|{(difficulty or '').strip().lower()}|{n}|{lang}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def make_coach_cache_key(user_id: int) -> str:
    current_hour = int(time.time() // 3600)
    return f"coach|{user_id}|{current_hour}"

async def single_flight_cached(cache_key: str, ttl: int, generate):
    """
    P1.1 + P2.2: Redis-backed cache with cross-instance single-flight.
    Returns (payload, was_cached). `generate` is an async callable producing
    the payload to cache on miss. Concurrent identical requests wait for the
    first generator's result (up to ~10s) instead of firing duplicate Groq calls.
    """
    cached = await cache.get_ai_cache_redis(cache_key, ttl)
    if cached is not None:
        return cached, True

    lock = cache.get_single_flight_lock_redis(cache_key)
    acquired = False
    try:
        acquired = bool(await lock.acquire())
    except Exception as e:
        logger.warning("single-flight lock acquire failed for %s: %s", cache_key, e)

    if acquired:
        try:
            # Double-check after acquiring: another instance may have finished.
            cached = await cache.get_ai_cache_redis(cache_key, ttl)
            if cached is not None:
                return cached, True
            payload = await generate()
            try:
                await cache.set_ai_cache_redis(cache_key, payload, ttl)
            except Exception as e:
                logger.warning("AI cache write failed: %s", e)
            return payload, False
        finally:
            try:
                await lock.release()
            except Exception:
                pass

    # Did not acquire: another request is generating. Poll the cache briefly.
    for _ in range(10):
        await asyncio.sleep(1)
        cached = await cache.get_ai_cache_redis(cache_key, ttl)
        if cached is not None:
            return cached, True

    # Give up waiting; generate without coordination (rare).
    payload = await generate()
    return payload, False

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

# -----------------------------
# PYDANTIC SCHEMAS
# -----------------------------
class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    username: Optional[str] = None
    language: str = "en"
    user_type: str = "general"
    grade: Optional[str] = None

class SubmitIn(BaseModel):
    topic_name: Optional[str] = None
    topic: Optional[str] = None
    correct: int = Field(ge=0)
    total: int = Field(ge=1)
    source: str = "static"
    session_id: Optional[int] = None

class AIQuizIn(BaseModel):
    topic: str = Field(min_length=1)
    num_questions: int = Field(default=4, ge=1, le=30)
    difficulty: str = "mixed"

class TaskIn(BaseModel):
    title: str = Field(min_length=1)
    topic: Optional[str] = None
    deadline: str = Field(min_length=10)

class AskIn(BaseModel):
    question: str = Field(min_length=1)
    history: List[dict] = []

class CheckoutIn(BaseModel):
    tier: str

class SettingsIn(BaseModel):
    username: Optional[str] = None
    language: str = "en"
    user_type: str = "general"
    grade: Optional[str] = None

class PasswordIn(BaseModel):
    old_password: str
    new_password: str = Field(min_length=6)

class GoogleExchangeIn(BaseModel):
    code: str

# -----------------------------
# APP + LIFECYCLE
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_http_client()
    await db.init_db()
    await cache.init_redis()
    await init_arq_pool()

    # P1.3: Start background cleanup task
    cleanup_task = asyncio.create_task(db.run_periodic_cleanup())

    # P1.4: Observability
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=0.1,
            integrations=[StarletteIntegration(), FastApiIntegration()],
        )
        logger.info("Sentry initialized.")

    logger.info("El-Peak startup complete.")
    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    await close_arq_pool()
    await cache.close_redis()
    await close_http_client()
    await db.close_pool()
    logger.info("El-Peak shutdown complete.")

app = FastAPI(
    title="El-Peak API",
    version="12.3",
    lifespan=lifespan
)

# P1.4: Metrics
if os.getenv("ENABLE_METRICS", "true").lower() in ("1", "true", "yes", "on"):
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# P0.4: gzip anything above 500 bytes (HTML shell, JSON payloads).
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response

# -----------------------------
# AUTH DEPENDENCY
# -----------------------------
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_raw = payload.get("sub")
        if user_id_raw is None:
            raise credentials_exception
        user_id = int(user_id_raw)
    except jwt.PyJWTError:
        raise credentials_exception
    except Exception:
        raise credentials_exception
    user = await db.get_user_full(user_id)
    if not user:
        raise credentials_exception
    user["effective_tier"] = effective_tier_from_user(user)
    return user

async def check_rate_limit(user_id: int):
    # P2.2: Redis-backed, correct across instances.
    allowed, wait_seconds = await cache.check_rate_limit_redis(f"ai:{user_id}", RATE_LIMIT_SECONDS)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached. Please wait {wait_seconds} seconds before generating more AI content."
        )

async def check_auth_ip_limit(request: Request):
    """P0.5 + P2.2: throttle anonymous auth endpoints per IP (~5/min), via Redis."""
    allowed, wait_seconds = await cache.check_rate_limit_redis(f"ip:{client_ip(request)}", AUTH_RATE_LIMIT_SECONDS)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many attempts. Please wait {wait_seconds} seconds."
        )

# -----------------------------
# GROQ
# -----------------------------
async def groq_chat(
    messages: List[dict],
    temperature: float = 0.6,
    max_tokens: int = 1200,
    model_override: Optional[str] = None,
    timeout: int = 120
) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY.")
    model_to_use = model_override or GROQ_MODEL
    payload = {
        "model": model_to_use,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    client = get_http_client()
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
    messages: List[dict],
    tier: str,
    temperature: float = 0.6,
    max_tokens: int = 1200,
    model_override: Optional[str] = None
) -> str:
    tier = (tier or "free").strip().lower()
    models = []
    if model_override:
        models.append(model_override)
    models.append(TIER_MODELS.get(tier, GROQ_MODEL))
    models.extend(TIER_FALLBACK_MODELS.get(tier, [GROQ_MODEL]))
    models = _unique_models(models)
    last_error = None
    for model in models:
        try:
            return await groq_chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model_override=model
            )
        except Exception as e:
            last_error = e
            logger.warning("Groq model %s failed: %s", model, str(e))
            continue
    raise last_error or RuntimeError("Groq unavailable.")

async def validate_quiz_topic(topic: str) -> None:
    """
    P1.3: cheap pre-check on the smallest model — reject gibberish,
    commands/injections, or topics with no factual content BEFORE paying
    for a full generation call. Not counted against the monthly AI-question
    quota (only successful generations are); still sits inside the
    request's rate limit. Fails OPEN: if the gate itself errors, the real
    generation call proceeds.
    """
    system = (
        "You are a quiz-topic validator. Reply with exactly one word and nothing else: "
        "VALID if the text names a real subject a factual multiple-choice quiz could be "
        "written about; INVALID if it is gibberish, a request or command, or has no "
        "factual content to quiz on."
    )
    user = f"Topic to validate: {topic[:200]}"
    try:
        raw = await groq_chat(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=4,
            model_override="llama-3.1-8b-instant",
            timeout=15
        )
    except Exception as e:
        logger.info("Topic gate unavailable (fail-open): %s", str(e))
        return
    verdict = (raw or "").strip().upper()
    if not verdict.startswith("VALID"):
        raise HTTPException(
            400,
            "That topic doesn't look quizzable. Try a real subject like 'Photosynthesis' or 'Cold War'."
        )

# -----------------------------
# STATIC QUIZ BANK
# -----------------------------
STATIC_QUIZZES = {
    "g1-6": {
        "Arabic (اللغة العربية)": {
            "1-2": [
                {"question": "كم عدد حروف اللغة العربية؟", "options": ["28 حرفاً", "26 حرفاً", "30 حرفاً", "32 حرفاً"], "answer_index": 0, "explanation": "اللغة العربية تحتوي على 28 حرفاً أساسياً.", "context": ""},
                {"question": "ما جمع كلمة 'قلم'؟", "options": ["أقلام", "كتب", "قلمة", "قوالم"], "answer_index": 0, "explanation": "جمع تكسير لكلمة قلم هو 'أقلام'.", "context": ""},
                {"question": "ما مضاد كلمة 'كبير'؟", "options": ["صغير", "طويل", "ضخم", "قصير"], "answer_index": 0, "explanation": "عكس كلمة كبير هو صغير.", "context": ""}
            ],
            "3-4": [
                {"question": "ما هو الفعل في جملة 'ذاكر الطالب الدرس'؟", "options": ["ذاكر", "الطالب", "الدرس", "لا يوجد"], "answer_index": 0, "explanation": "الفعل هو كلمة تدل على حدث، وهنا 'ذاكر'.", "context": ""},
                {"question": "كم نوعاً للفعل الماضي؟", "options": ["3 أنواع", "نوعان", "4 أنواع", "نوع واحد"], "answer_index": 0, "explanation": "الفعل الماضي يكون: صحيح، معتل، مهموز.", "context": ""},
                {"question": "ما مرادف كلمة 'سعيد'؟", "options": ["فرح", "حزين", "غاضب", "متعب"], "answer_index": 0, "explanation": "مرادف كلمة سعيد هو فرح.", "context": ""}
            ],
            "5-6": [
                {"question": "ما نوع 'إن' في جملة 'إن العلم نور'؟", "options": ["حرف ناسخ", "فعل ناقص", "اسم موصول", "حرف جر"], "answer_index": 0, "explanation": "'إن' حرف ناسخ ينصب المبتدأ.", "context": ""},
                {"question": "ما هو المفعول المطلق؟", "options": ["مصدر يؤكد الفعل", "اسم يدل على زمن", "اسم يدل على مكان", "اسم يدل على آلة"], "answer_index": 0, "explanation": "المفعول المطلق هو مصدر يؤكد حدث الفعل.", "context": ""},
                {"question": "ما جمع كلمة 'بحر'؟", "options": ["بحار", "أبحر", "بُحُور", "كل ما سبق"], "answer_index": 3, "explanation": "كلمة بحر تجمع على 'بحار' و'أبحر' و'بُحُور'.", "context": ""}
            ]
        },
        "Math (الرياضيات)": {
            "1-2": [
                {"question": "What is 5 + 7?", "options": ["12", "11", "13", "10"], "answer_index": 0, "explanation": "5 + 7 = 12.", "context": ""},
                {"question": "What is 9 - 4?", "options": ["5", "6", "4", "3"], "answer_index": 0, "explanation": "9 - 4 = 5.", "context": ""},
                {"question": "What shape is a ball?", "options": ["Sphere", "Cube", "Triangle", "Square"], "answer_index": 0, "explanation": "A ball is shaped like a sphere.", "context": ""}
            ],
            "3-4": [
                {"question": "What is 6 x 8?", "options": ["48", "42", "56", "36"], "answer_index": 0, "explanation": "6 multiplied by 8 is 48.", "context": ""},
                {"question": "What is 1/2 as a decimal?", "options": ["0.5", "0.2", "1.2", "0.05"], "answer_index": 0, "explanation": "1 divided by 2 is 0.5.", "context": ""},
                {"question": "How many sides does a pentagon have?", "options": ["5", "6", "4", "7"], "answer_index": 0, "explanation": "A pentagon has 5 sides.", "context": ""}
            ],
            "5-6": [
                {"question": "What is the LCM of 4 and 6?", "options": ["12", "24", "2", "8"], "answer_index": 0, "explanation": "The Least Common Multiple of 4 and 6 is 12.", "context": ""},
                {"question": "Solve for x: 3x = 15", "options": ["5", "3", "12", "45"], "answer_index": 0, "explanation": "x = 15 / 3 = 5.", "context": ""},
                {"question": "What is the area of a triangle with base 4 and height 6?", "options": ["12", "24", "10", "48"], "answer_index": 0, "explanation": "Area = 1/2 * base * height = 1/2 * 4 * 6 = 12.", "context": ""}
            ]
        },
        "Science (العلوم)": {
            "1-2": [
                {"question": "كم عدد حواس الإنسان؟", "options": ["5 حواس", "3 حواس", "7 حواس", "10 حواس"], "answer_index": 0, "explanation": "حواس الإنسان هي السمع، البصر، الشم، التذوق، اللمس.", "context": ""},
                {"question": "What do plants need to grow?", "options": ["Sunlight", "Darkness", "Plastic", "Rocks"], "answer_index": 0, "explanation": "Plants need sunlight to make food.", "context": ""}
            ],
            "3-4": [
                {"question": "What are the three states of matter?", "options": ["Solid, Liquid, Gas", "Hot, Cold, Warm", "Big, Small, Medium", "Red, Blue, Green"], "answer_index": 0, "explanation": "Matter can be solid, liquid, or gas.", "context": ""},
                {"question": "ما هو الكوكب الذي نعيش عليه؟", "options": ["الأرض", "المريخ", "المشتري", "القمر"], "answer_index": 0, "explanation": "نعيش على كوكب الأرض.", "context": ""}
            ],
            "5-6": [
                {"question": "What organ pumps blood?", "options": ["Heart", "Lungs", "Brain", "Liver"], "answer_index": 0, "explanation": "The heart pumps blood throughout the body.", "context": ""},
                {"question": "ما هي وظيفة الجذور في النبات؟", "options": ["تثبت النبات وتمتص الماء", "تصنع الغذاء", "تنشر البذور", "تخزن الغذاء"], "answer_index": 0, "explanation": "الجذور تثبت النبات في التربة وتمتص منها الماء والأملاح.", "context": ""}
            ]
        }
    },
    "g7-9": {
        "Arabic (اللغة العربية)": {
            "7": [
                {"question": "ما نوع كلمة 'طالب' في جملة 'الطالبُ مجتهدٌ'؟", "options": ["مبتدأ", "خبر", "فعل", "حرف"], "answer_index": 0, "explanation": "'الطالب' مبتدأ مرفوع وعلامة رفعه الضمة.", "context": ""},
                {"question": "ما هو التشبيه في قولنا 'الجندي كالأسد'؟", "options": ["حيث شبه الجندي بالأسد", "استعارة", "كناية", "طباق"], "answer_index": 0, "explanation": "في هذا التشبيه، شبه الكاتب الجندي بالأسد في الشجاعة.", "context": ""}
            ],
            "8": [
                {"question": "الجار والمجرور في جملة 'ذهبت إلى المدرسة'؟", "options": ["إلى المدرسة", "ذهبت", "المدرسة", "لا يوجد"], "answer_index": 0, "explanation": "حرف الجر 'إلى' واسمه 'المدرسة' يسميان الجار والمجرور.", "context": ""},
                {"question": "اسم الإشارة للقريب هو؟", "options": ["هذا", "ذلك", "أولئك", "هؤلاء"], "answer_index": 0, "explanation": "'هذا' يدل على المفرد القريب.", "context": ""}
            ],
            "9": [
                {"question": "من هو شاعر النيل؟", "options": ["حافظ إبراهيم", "أحمد شوقي", "إيليا أبو ماضي", "نزار قباني"], "answer_index": 0, "explanation": "حافظ إبراهيم هو شاعر النيل.", "context": ""},
                {"question": "ما الفرق بين التشبيه والاستعارة؟", "options": ["التشبيه يذكر المشبه والمشبه به", "لا يوجد فرق", "الاستعارة أقصر", "التشبيه يحذف المشبه"], "answer_index": 0, "explanation": "في التشبيه يذكر المشبه والمشبه به، بينما في الاستعارة يذكر أحدهما ويدل على الآخر.", "context": ""}
            ]
        },
        "Math (الرياضيات)": {
            "7": [
                {"question": "If x + 5 = 12, what is x?", "options": ["7", "5", "17", "12"], "answer_index": 0, "explanation": "x = 12 - 5 = 7.", "context": ""},
                {"question": "What is the area of a rectangle with length 4 and width 3?", "options": ["12", "7", "14", "1"], "answer_index": 0, "explanation": "Area = length * width = 4 * 3 = 12.", "context": ""}
            ],
            "8": [
                {"question": "What is the sum of angles in a triangle?", "options": ["180 degrees", "90 degrees", "360 degrees", "270 degrees"], "answer_index": 0, "explanation": "The sum of interior angles of a triangle is always 180 degrees.", "context": ""},
                {"question": "Solve: 2(x - 3) = 8", "options": ["7", "5", "11", "4"], "answer_index": 0, "explanation": "2x - 6 = 8 => 2x = 14 => x = 7.", "context": ""}
            ],
            "9": [
                {"question": "What is the value of x in x^2 = 81?", "options": ["9", "8", "10", "7"], "answer_index": 0, "explanation": "9 * 9 = 81.", "context": ""},
                {"question": "What is the slope of a horizontal line?", "options": ["0", "1", "Undefined", "-1"], "answer_index": 0, "explanation": "A horizontal line has no steepness, so its slope is 0.", "context": ""}
            ]
        },
        "Science (العلوم)": {
            "7": [
                {"question": "What is the powerhouse of the cell?", "options": ["Mitochondria", "Nucleus", "Ribosome", "Cytoplasm"], "answer_index": 0, "explanation": "Mitochondria produce ATP, the cell's energy currency.", "context": ""},
                {"question": "What is the chemical symbol for water?", "options": ["H2O", "CO2", "O2", "NaCl"], "answer_index": 0, "explanation": "Water is made of 2 Hydrogen atoms and 1 Oxygen atom.", "context": ""}
            ],
            "8": [
                {"question": "What force pulls objects towards Earth?", "options": ["Gravity", "Friction", "Magnetism", "Tension"], "answer_index": 0, "explanation": "Gravity is the force that attracts objects toward the center of the Earth.", "context": ""},
                {"question": "ما هي وحدة قياس القوة؟", "options": ["نيوتن", "جول", "واط", "متر"], "answer_index": 0, "explanation": "تقاس القوة بوحدة نيوتن.", "context": ""}
            ],
            "9": [
                {"question": "What is the atomic number of Oxygen?", "options": ["8", "6", "16", "12"], "answer_index": 0, "explanation": "Oxygen has 8 protons in its nucleus.", "context": ""},
                {"question": "What type of energy is stored in food?", "options": ["Chemical", "Kinetic", "Potential", "Nuclear"], "answer_index": 0, "explanation": "Food stores chemical energy in its bonds.", "context": ""}
            ]
        }
    },
    "g10-12": {
        "Arabic (اللغة العربية)": {
            "10": [
                {"question": "ما هو المفعول المطلق في جملة 'انتصر الجيش انتصاراً عظيماً'؟", "options": ["انتصاراً", "الجيش", "عظيماً", "انتصر"], "answer_index": 0, "explanation": "المفعول المطلق هو مصدر يؤكد الفعل، وهنا 'انتصاراً'.", "context": ""},
                {"question": "من قائل مسرحية 'مجنون ليلى'؟", "options": ["أحمد شوقي", "توفيق الحكيم", "نجيب محفوظ", "أحمد أمين"], "answer_index": 0, "explanation": "مسرحية مجنون ليلى من أشهر أعمال أمير الشعراء أحمد شوقي.", "context": ""}
            ],
            "11": [
                {"question": "البلاغة علم يبحث فيه عن؟", "options": ["المحسنات والصور البيانية", "النحو والإعراب", "تاريخ الأدب", "الكتابة الإنشائية"], "answer_index": 0, "explanation": "علم البلاغة يبحث في المحسنات البديعية والمعنوية والصور البيانية.", "context": ""},
                {"question": "التشبيه هو؟", "options": ["تشبيه شيء بشيء لبيان وجه الشبه", "إسناد معنى لشيء ليس منه", "ذكر الشيء بلفظه", "ذكر الشيء بضده"], "answer_index": 0, "explanation": "التشبيه هو عقد مماثلة بين طرفين لبيان وجه الشبه.", "context": ""}
            ],
            "12": [
                {"question": "ما هي مدرسة الإحياء والبعث؟", "options": ["مدرسة أدبية تهتم بتقليد الشعر القديم", "مدرسة حديثة ترفض الشعر", "مدرسة تكتب الرواية فقط", "مدرسة تنتمي للأدب المهجري"], "answer_index": 0, "explanation": "مدرسة الإحياء والبعث هي مدرسة شعرية حافظت على القديم وأضافت إليه روحاً جديدة.", "context": ""},
                {"question": "ما الفرق بين الوراقة والطباعة؟", "options": ["الوراقة كتابة يدوية والطباعة آلية", "لا فرق", "الوراقة أحدث", "الطباعة تستخدم الحبر"], "answer_index": 0, "explanation": "الوراقة هي صناعة الكتابة على الورق يدوياً، بينما الطباعة تتم بواسطة الآلات.", "context": ""}
            ]
        },
        "Math (الرياضيات)": {
            "10": [
                {"question": "What is log10(100)?", "options": ["2", "10", "0.1", "100"], "answer_index": 0, "explanation": "10^2 = 100, so log10(100) = 2.", "context": ""},
                {"question": "What is the value of sin(30)?", "options": ["1/2", "1", "√3/2", "0"], "answer_index": 0, "explanation": "sin(30) = 1/2.", "context": ""}
            ],
            "11": [
                {"question": "What is the derivative of x^3?", "options": ["3x^2", "x^2", "3x", "3"], "answer_index": 0, "explanation": "d/dx(x^n) = n x^(n-1).", "context": ""},
                {"question": "What is the limit of (x^2 - 1)/(x - 1) as x approaches 1?", "options": ["2", "1", "0", "Undefined"], "answer_index": 0, "explanation": "Factoring gives (x-1)(x+1)/(x-1) = x+1. At x=1, value is 2.", "context": ""}
            ],
            "12": [
                {"question": "What is the integral of 2x dx?", "options": ["x^2 + C", "2 + C", "x^2 / 2 + C", "2x^2 + C"], "answer_index": 0, "explanation": "The anti-derivative of 2x is x^2 + C.", "context": ""},
                {"question": "What is the determinant of [[2,0],[0,2]]?", "options": ["4", "0", "2", "1"], "answer_index": 0, "explanation": "2*2 - 0*0 = 4.", "context": ""}
            ]
        },
        "Physics (الفيزياء)": {
            "10": [
                {"question": "What is the unit of force?", "options": ["Newton", "Joule", "Watt", "Pascal"], "answer_index": 0, "explanation": "Force is measured in Newtons (N).", "context": ""},
                {"question": "What is the acceleration due to gravity on Earth?", "options": ["9.8 m/s^2", "8.9 m/s^2", "10.2 m/s^2", "9.5 m/s^2"], "answer_index": 0, "explanation": "g ≈ 9.8 m/s^2 on Earth.", "context": ""}
            ],
            "11": [
                {"question": "What is Ohm's Law?", "options": ["V = I * R", "P = I * V", "V = I / R", "R = V * I"], "answer_index": 0, "explanation": "Voltage equals current times resistance.", "context": ""},
                {"question": "What type of wave is light?", "options": ["Transverse", "Longitudinal", "Mechanical", "Sound"], "answer_index": 0, "explanation": "Light is a transverse electromagnetic wave.", "context": ""}
            ],
            "12": [
                {"question": "What is the charge of an electron?", "options": ["Negative", "Positive", "Neutral", "Variable"], "answer_index": 0, "explanation": "Electrons carry a negative elementary charge.", "context": ""},
                {"question": "What is the speed of light in vacuum?", "options": ["3 x 10^8 m/s", "3 x 10^6 m/s", "3 x 10^10 m/s", "9.8 m/s"], "answer_index": 0, "explanation": "c ≈ 300,000,000 m/s.", "context": ""}
            ]
        }
    },
    "general": {
        "General Knowledge": {
            "easy": [
                {"question": "What is the capital of Egypt?", "options": ["Cairo", "Alexandria", "Giza", "Luxor"], "answer_index": 0, "explanation": "Cairo is the capital and largest city of Egypt.", "context": ""},
                {"question": "كم عدد القارات في العالم؟", "options": ["7 قارات", "5 قارات", "6 قارات", "8 قارات"], "answer_index": 0, "explanation": "يوجد 7 قارات في العالم.", "context": ""}
            ],
            "mixed": [
                {"question": "What is the largest ocean on Earth?", "options": ["Pacific", "Atlantic", "Indian", "Arctic"], "answer_index": 0, "explanation": "The Pacific Ocean is the largest and deepest.", "context": ""},
                {"question": "ما هو أطول نهر في العالم؟", "options": ["النيل", "الأمازون", "المسيسيبي", "اليانغتسي"], "answer_index": 0, "explanation": "نهر النيل هو أطول نهر في العالم.", "context": ""}
            ],
            "hard": [
                {"question": "What is the chemical symbol for Gold?", "options": ["Au", "Ag", "Gd", "Go"], "answer_index": 0, "explanation": "Au comes from the Latin word 'aurum'.", "context": ""},
                {"question": "ما هي عاصمة اليابان؟", "options": ["طوكيو", "كيوتو", "أوساكا", "سيول"], "answer_index": 0, "explanation": "طوكيو هي عاصمة اليابان.", "context": ""}
            ]
        }
    }
}

# Replace the broken QUIZ_SCHEMA with this:
QUIZ_SCHEMA = (
    '{"questions":[{"question": "...", "options":["A","B","C","D"],'
    '"answer_index":0, "explanation": "...", "context": "exact quote from text if pdf"}]}'
)

def _match_static(topic: str, n: int, grade: str = "general", difficulty: str = "mixed") -> dict:
    grade = normalize_grade(grade)
    bank_dict = STATIC_QUIZZES.get(grade, STATIC_QUIZZES["general"])
    key = (topic or "").strip().lower()
    subject = None
    for name in bank_dict:
        name_lower = name.lower()
        if name_lower == key or name_lower in key or key in name_lower:
            subject = name
            break
    if subject is None:
        if grade == "general" and "General Knowledge" in bank_dict:
            subject = "General Knowledge"
        else:
            subject = list(bank_dict.keys())[0]
    subject_bank = bank_dict[subject]
    desired_key = grade_difficulty_key(grade, difficulty)
    questions = subject_bank.get(desired_key, [])
    if not questions:
        questions = [q for qs in subject_bank.values() for q in qs]
    if not questions:
        return {
            "topic": topic,
            "subject": subject,
            "grade": grade,
            "difficulty": difficulty,
            "questions": [],
            "ai": False,
            "source": "static"
        }
    qs = random.sample(questions, min(n, len(questions)))
    return {
        "topic": topic,
        "subject": subject,
        "grade": grade,
        "difficulty": difficulty,
        "questions": qs,
        "ai": False,
        "source": "static"
    }

# -----------------------------
# STATIC ROUTES
# -----------------------------
@app.get("/", include_in_schema=False)
async def home():
    if INDEX.exists():
        resp = FileResponse(INDEX)
        # P0.4: HTML shell must revalidate so sw.js / deploys propagate;
        # ETag/Last-Modified from FileResponse still allow cheap 304s.
        resp.headers["Cache-Control"] = "no-cache"
        return resp
    return HTMLResponse("<h1>El-Peak</h1><p>index.html not found.</p>")

@app.get("/manifest.json", include_in_schema=False)
async def manifest():
    resp = JSONResponse({
        "name": "El-Peak Study Compass",
        "short_name": "El-Peak",
        "description": "AI-powered study companion, quiz engine, coach, and scheduler.",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "orientation": "portrait-primary",
        "background_color": "#070710",
        "theme_color": "#070710",
        "categories": ["education", "productivity"],
        "icons": [
            {
                "src": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Crect width='512' height='512' rx='96' fill='%23070710'/%3E%3Cpath d='M96 400 L208 112 L264 272 L320 176 L416 400' fill='none' stroke='%236366f1' stroke-width='28' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E",
                "sizes": "any",
                "type": "image/svg+xml",
                "purpose": "any"
            },
            {
                "src": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Crect width='512' height='512' rx='96' fill='%23070710'/%3E%3Cpath d='M96 400 L208 112 L264 272 L320 176 L416 400' fill='none' stroke='%236366f1' stroke-width='28' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E",
                "sizes": "512x512",
                "type": "image/svg+xml",
                "purpose": "maskable"
            }
        ]
    })
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp

@app.get("/sw.js", include_in_schema=False)
async def service_worker():
    sw_path = BASE_DIR / "sw.js"
    if sw_path.exists():
        resp = FileResponse(sw_path, media_type="application/javascript")
        # P0.4: service worker must always revalidate.
        resp.headers["Cache-Control"] = "no-cache"
        return resp
    return JSONResponse({}, media_type="application/javascript")

@app.get("/health", include_in_schema=False)
async def health():
    try:
        await db.ping()
        return {"status": "ok", "db": "ok", "pdf_ocr_enabled": PDF_OCR_ENABLED, "arq": ARQ_POOL is not None}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "db": "unavailable",
                "detail": str(e)
            }
        )

@app.get("/api/health")
async def api_health():
    return await health()

# -----------------------------
# AUTH ROUTES
# -----------------------------
@app.post("/api/auth/signup")
async def signup(
    payload: SignupIn,
    _: Annotated[None, Depends(check_auth_ip_limit)]
):
    existing = await db.get_user_by_email(payload.email)
    if existing:
        raise HTTPException(400, "Email already registered.")
    # P0.1: bcrypt is CPU-bound; keep it off the event loop.
    hashed_pw = await run_in_threadpool(hash_password, payload.password)
    normalized_grade = normalize_grade(payload.grade)
    user = await db.create_user(
        email=payload.email,
        hashed_pw=hashed_pw,
        user_type=payload.user_type,
        grade=normalized_grade,
        username=payload.username,
        language=payload.language,
        provider="local"
    )
    if not user:
        raise HTTPException(500, "Failed to create user.")
    token = create_access_token({
        "sub": str(user["id"]),
        "email": user["email"]
    })
    return {
        "token": token,
        "email": user["email"]
    }

@app.post("/api/auth/login")
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    _: Annotated[None, Depends(check_auth_ip_limit)]
):
    user = await db.get_user_by_email(form_data.username)
    ok = False
    if user:
        ok = await run_in_threadpool(verify_password, form_data.password, user["password"])
    if not user or not ok:
        raise HTTPException(401, "Incorrect email or password.")
    await db.update_last_login(user["id"])
    token = create_access_token({
        "sub": str(user["id"]),
        "email": user["email"]
    })
    return {
        "token": token,
        "email": user["email"]
    }

@app.get("/api/auth/google")
async def google_login():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(500, "Google OAuth not configured.")
    state = session_serializer.dumps({
        "nonce": secrets.token_urlsafe(16),
        "ts": int(time.time())
    })
    auth_url = (
        "https://accounts.google.com/o/oauth2/auth"
        f"?response_type=code"
        f"&client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope=openid email profile"
        f"&state={state}"
        f"&prompt=select_account"
    )
    return {
        "auth_url": auth_url,
        "state": state
    }

@app.get("/api/auth/google/callback")
async def google_callback(request: Request, code: str, state: str):
    try:
        session_serializer.loads(state)
    except BadSignature:
        raise HTTPException(400, "Invalid OAuth state.")
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(500, "Google OAuth not configured.")
    client = get_http_client()
    token_resp = await client.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        },
        timeout=30
    )
    token_resp.raise_for_status()
    access_token = token_resp.json().get("access_token")
    if not access_token:
        raise HTTPException(500, "Failed to get Google access token.")
    user_resp = await client.get(
        "https://www.googleapis.com/oauth2/v1/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30
    )
    user_resp.raise_for_status()
    google_data = user_resp.json()
    email = google_data.get("email")
    if not email:
        raise HTTPException(400, "Could not get email from Google.")
    user = await db.get_user_by_email(email)
    if not user:
        random_password = secrets.token_urlsafe(32)
        hashed_pw = await run_in_threadpool(hash_password, random_password)
        user = await db.create_user(
            email=email,
            hashed_pw=hashed_pw,
            user_type="general",
            grade=None,
            username=email.split("@")[0],
            language="en",
            provider="google"
        )
        if not user:
            raise HTTPException(500, "Failed to create user.")
    await db.update_last_login(user["id"])
    oauth_code = await db.create_oauth_code(user["id"])
    return RedirectResponse(url=f"/?oauth_code={oauth_code}")

@app.post("/api/auth/google/exchange")
async def google_exchange(payload: GoogleExchangeIn):
    user_id = await db.consume_oauth_code(payload.code)
    if not user_id:
        raise HTTPException(400, "Invalid or expired OAuth code.")
    user = await db.get_user_full(user_id)
    if not user:
        raise HTTPException(404, "User not found.")
    token = create_access_token({
        "sub": str(user["id"]),
        "email": user["email"]
    })
    return {
        "token": token,
        "email": user["email"]
    }

# -----------------------------
# SETTINGS & PROFILE
# -----------------------------
@app.get("/api/user/settings")
async def get_settings(user: Annotated[dict, Depends(get_current_user)]):
    settings = await db.get_user_by_id(user["id"])
    if not settings:
        raise HTTPException(404, "User not found.")
    settings["effective_tier"] = effective_tier_from_user(user)
    return settings

@app.put("/api/user/settings")
async def update_settings(payload: SettingsIn, user: Annotated[dict, Depends(get_current_user)]):
    normalized_grade = normalize_grade(payload.grade)
    await db.update_user_settings(
        user_id=user["id"],
        username=payload.username or "",
        language=payload.language,
        user_type=payload.user_type,
        grade=normalized_grade
    )
    return {"status": "success"}

@app.put("/api/user/password")
async def update_password(payload: PasswordIn, user: Annotated[dict, Depends(get_current_user)]):
    db_user = await db.get_user_by_email(user["email"])
    ok = False
    if db_user:
        ok = await run_in_threadpool(verify_password, payload.old_password, db_user["password"])
    if not db_user or not ok:
        raise HTTPException(401, "Incorrect current password.")
    hashed_pw = await run_in_threadpool(hash_password, payload.new_password)
    await db.update_user_password(user["id"], hashed_pw)
    return {"status": "success"}

# -----------------------------
# USAGE & BILLING
# -----------------------------
@app.get("/api/usage")
async def get_usage(user: Annotated[dict, Depends(get_current_user)]):
    await db.reset_usage_if_needed(user["id"])
    usage = await db.get_user_usage(user["id"])
    if not usage:
        raise HTTPException(404, "User not found.")
    tier = effective_tier_from_user(usage)
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    return {
        "tier": tier,
        "stored_tier": usage.get("tier", "free"),
        "subscription_status": usage.get("subscription_status", "inactive"),
        "subscription_expires_at": usage.get("subscription_expires_at"),
        "ai_used": usage.get("ai_questions_used", 0),
        "ai_limit": limits["ai_questions"],
        "pdf_used": usage.get("pdf_uploads_used", 0),
        "pdf_limit": limits["pdf_uploads"],
        "pdf_max_mb": limits["pdf_max_mb"],
        "prices": TIER_PRICES,
        "simulated_payments": ALLOW_SIMULATED_PAYMENTS and not _paymob_configured()
    }

@app.post("/api/payment/checkout")
async def create_checkout(payload: CheckoutIn, user: Annotated[dict, Depends(get_current_user)]):
    tier = payload.tier.strip().lower()
    if tier not in TIER_PRICES:
        raise HTTPException(400, "Invalid tier selected.")
    if not _paymob_configured():
        if ALLOW_SIMULATED_PAYMENTS:
            logger.warning("Paymob not configured. Simulating successful checkout for user %s.", user["id"])
            await db.upgrade_user_tier(user["id"], tier)
            return {"checkout_url": "/?upgraded=true"}
        raise HTTPException(503, "Payments are not configured.")
    amount = TIER_PRICES[tier]
    merchant_order_id = f"user_{user['id']}_{tier}_{int(time.time())}"
    await db.create_payment(
        user_id=user["id"],
        tier=tier,
        amount_cents=amount,
        merchant_order_id=merchant_order_id,
        status="pending"
    )
    try:
        client = get_http_client()
        auth_resp = await client.post(
            "https://accept.paymob.com/api/auth/tokens",
            json={"api_key": PAYMOB_API_KEY},
            timeout=45
        )
        auth_resp.raise_for_status()
        auth_token = auth_resp.json().get("token")
        if not auth_token:
            raise HTTPException(500, "Failed to authenticate with Paymob.")
        order_resp = await client.post(
            "https://accept.paymob.com/api/ecommerce/orders",
            json={
                "auth_token": auth_token,
                "delivery_needed": "false",
                "amount_cents": amount,
                "currency": "EGP",
                "merchant_order_id": merchant_order_id,
                "items": []
            },
            timeout=45
        )
        order_resp.raise_for_status()
        paymob_order_id = order_resp.json().get("id")
        payment_key_resp = await client.post(
            "https://accept.paymob.com/api/acceptance/payment_keys",
            json={
                "auth_token": auth_token,
                "amount_cents": amount,
                "expiration": 3600,
                "order_id": paymob_order_id,
                "billing_data": {
                    "apartment": "NA",
                    "email": user["email"],
                    "floor": "NA",
                    "first_name": "El-Peak",
                    "street": "NA",
                    "building": "NA",
                    "phone_number": "01000000000",
                    "shipping_method": "NA",
                    "postal_code": "NA",
                    "city": "Cairo",
                    "country": "EG",
                    "last_name": "User",
                    "state": "NA"
                },
                "currency": "EGP",
                "integration_id": int(PAYMOB_INTEGRATION_ID)
            },
            timeout=45
        )
        payment_key_resp.raise_for_status()
        payment_token = payment_key_resp.json().get("token")
        if not payment_token:
            raise HTTPException(500, "Failed to generate payment key.")
        await db.update_payment_paymob_order(
            merchant_order_id=merchant_order_id,
            paymob_order_id=str(paymob_order_id)
        )
        iframe_url = f"https://accept.paymob.com/api/acceptance/iframes/{PAYMOB_IFRAME_ID}?payment_token={payment_token}"
        return {"checkout_url": iframe_url}
    except httpx.HTTPStatusError as exc:
        logger.error("Paymob API Error: %s", exc.response.text)
        await db.update_payment_status(merchant_order_id, "gateway_error")
        raise HTTPException(500, "Payment gateway error.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Checkout Error: %s", str(e))
        await db.update_payment_status(merchant_order_id, "checkout_error")
        raise HTTPException(500, "Failed to initiate payment.")

@app.post("/api/payment/webhook")
async def payment_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid payload.")
    obj = payload.get("obj", {}) or {}
    order_obj = obj.get("order", {}) or {}
    merchant_order_id = str(order_obj.get("merchant_order_id", "")).strip()
    if PAYMOB_HMAC_SECRET:
        req_hmac = request.headers.get("X-PAYMOB-HMAC", "")
        source_data = obj.get("source_data", {}) or {}
        concat = "".join([
            _hmac_value(obj.get("amount_cents", "")),
            _hmac_value(obj.get("created_at", "")),
            _hmac_value(obj.get("currency", "")),
            _hmac_value(obj.get("error_occured", "")),
            _hmac_value(obj.get("has_parent_transaction", "")),
            _hmac_value(obj.get("id", "")),
            _hmac_value(obj.get("integration_id", "")),
            _hmac_value(obj.get("is_hmac_authenticated", "")),
            _hmac_value(obj.get("is_auth", "")),
            _hmac_value(obj.get("is_capture", "")),
            _hmac_value(obj.get("is_refunded", "")),
            _hmac_value(obj.get("is_standalone_payment", "")),
            _hmac_value(obj.get("is_voided", "")),
            _hmac_value(order_obj.get("id", "")),
            _hmac_value(obj.get("owner", "")),
            _hmac_value(obj.get("pending", "")),
            _hmac_value(source_data.get("pan", "")),
            _hmac_value(source_data.get("sub_type", "")),
            _hmac_value(source_data.get("type", "")),
            _hmac_value(obj.get("success", ""))
        ])
        calculated_hmac = hmac.new(
            PAYMOB_HMAC_SECRET.encode("utf-8"),
            concat.encode("utf-8"),
            hashlib.sha512
        ).hexdigest()
        if calculated_hmac != req_hmac:
            logger.error("Paymob HMAC verification failed for %s", merchant_order_id)
            raise HTTPException(403, "Invalid HMAC signature.")
    success = _as_bool(payload.get("success")) or _as_bool(obj.get("success"))
    if not merchant_order_id:
        return {"status": "ignored", "reason": "missing merchant_order_id"}
    if not success:
        await db.update_payment_status(
            merchant_order_id=merchant_order_id,
            status="failed",
            raw_payload=json.dumps(payload, ensure_ascii=False)
        )
        return {"status": "ignored", "reason": "payment not successful"}
    parts = merchant_order_id.split("_")
    if len(parts) < 3:
        return {"status": "ignored", "reason": "invalid merchant_order_id"}
    try:
        user_id = int(parts[1])
    except Exception:
        return {"status": "ignored", "reason": "invalid user id"}
    tier = parts[2].strip().lower()
    if tier not in TIER_PRICES:
        return {"status": "ignored", "reason": "invalid tier"}
    upgraded = await db.finalize_payment(
        user_id=user_id,
        tier=tier,
        merchant_order_id=merchant_order_id,
        amount_cents=obj.get("amount_cents"),
        raw_payload=json.dumps(payload, ensure_ascii=False)
    )
    if upgraded:
        logger.info("Upgraded user %s to %s", user_id, tier)
    return {
        "status": "success",
        "upgraded": upgraded
    }

@app.get("/api/payments")
async def payments(user: Annotated[dict, Depends(get_current_user)]):
    return await db.get_payments(user["id"], limit=20)

# -----------------------------
# DASHBOARD ROUTES
# -----------------------------
@app.get("/api/stats")
async def stats(user: Annotated[dict, Depends(get_current_user)]):
    return await db.get_stats(user["id"])

@app.get("/api/topics")
async def topics(user: Annotated[dict, Depends(get_current_user)]):
    return await db.get_topics(user["id"])

@app.get("/api/recent")
async def recent(user: Annotated[dict, Depends(get_current_user)], limit: int = 20):
    return await db.recent_attempts(user["id"], limit)

@app.get("/api/heatmap")
async def heatmap(user: Annotated[dict, Depends(get_current_user)]):
    return await db.heatmap(user["id"])

# -----------------------------
# QUIZ ENGINE
# -----------------------------
@app.get("/api/quiz/subjects")
async def quiz_subjects(user: Annotated[dict, Depends(get_current_user)]):
    grade = normalize_grade(user.get("grade"))
    subjects = list(STATIC_QUIZZES.get(grade, STATIC_QUIZZES["general"]).keys())
    return {
        "grade": grade,
        "subjects": subjects
    }

@app.get("/api/quiz/static")
async def quiz_static(
    user: Annotated[dict, Depends(get_current_user)],
    topic: str = Query(default="General Knowledge"),
    n: int = Query(default=4, ge=1, le=10),
    difficulty: str = Query(default="mixed")
):
    grade = normalize_grade(user.get("grade"))
    n = max(3, min(10, n))
    quiz = _match_static(topic, n, grade, difficulty)
    if quiz["questions"]:
        session_id = await db.create_quiz_session(
            user_id=user["id"],
            topic=quiz.get("topic") or quiz.get("subject") or "General",
            source="static",
            difficulty=difficulty,
            questions=quiz["questions"]
        )
        quiz["session_id"] = session_id
    return quiz

@app.post("/api/quiz/ai")
async def quiz_ai(payload: AIQuizIn, user: Annotated[dict, Depends(get_current_user)]):
    await check_rate_limit(user["id"])
    await db.reset_usage_if_needed(user["id"])
    usage = await db.get_user_usage(user["id"])
    tier = effective_tier_from_user(usage or user)
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    num_questions = min(payload.num_questions, limits["max_ai_questions_per_request"])
    if usage["ai_questions_used"] + num_questions > limits["ai_questions"]:
        raise HTTPException(
            403,
            f"You have reached your monthly AI question limit ({limits['ai_questions']}). Please upgrade."
        )
    language = (user.get("language") or "en").strip().lower()
    grade = normalize_grade(user.get("grade"))

    # P1.1 + P2.2: Redis cache with cross-instance single-flight
    cache_key = make_ai_quiz_cache_key(payload.topic, grade, payload.difficulty, num_questions, language)

    # P1.3: topic gate — only when we'd actually pay for a generation (cache miss).
    if not await cache.get_ai_cache_redis(cache_key, AI_QUIZ_CACHE_TTL):
        await validate_quiz_topic(payload.topic)

    async def generate():
        if language == "ar":
            lang_instruction = (
                "Please generate the questions, options, and explanations strictly in Arabic. "
                "CRITICAL: If the topic involves Arabic poetry, literature, or history, you MUST be strictly factual. "
                "DO NOT invent or hallucinate poetry verses. If you do not know the exact verse, ask a general question "
                "about the poem's theme or author instead."
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
                    f"Generate exactly {num_questions} {payload.difficulty} multiple-choice questions about "
                    f"'{payload.topic}' for a student in grade group {grade}. "
                    "Each must have 4 plausible options and a one-sentence explanation."
                )
            }
        ]
        max_tokens = max(400, num_questions * 250)
        raw = await groq_chat_with_fallback(
            messages=messages,
            tier=tier,
            temperature=0.7,
            max_tokens=max_tokens
        )
        clean = _clean_questions(extract_json(raw), num_questions)
        if not clean:
            raise ValueError("Empty question set from AI.")
        return {"questions": clean}

    try:
        result, was_cached = await single_flight_cached(cache_key, AI_QUIZ_CACHE_TTL, generate)
        clean = result.get("questions", [])
    except Exception as e:
        logger.warning("AI Quiz Error: %s", str(e))
        fallback = _match_static(
            topic=payload.topic,
            n=min(4, num_questions),
            grade=grade,
            difficulty=payload.difficulty
        )
        fallback["note"] = "AI unavailable — served from the static bank."
        return fallback

    session_id = await db.create_quiz_session(
        user_id=user["id"],
        topic=payload.topic,
        source="ai",
        difficulty=payload.difficulty,
        questions=clean
    )
    # Quota is recorded on cache hits AND misses (no paywall bypass).
    await db.increment_usage(user["id"], ai_questions=num_questions)
    await db.log_usage_event(user["id"], "ai_question", num_questions, tier)
    return {
        "topic": payload.topic,
        "subject": "AI",
        "grade": grade,
        "difficulty": payload.difficulty,
        "questions": clean,
        "ai": True,
        "source": "ai",
        "session_id": session_id,
        "cached": was_cached
    }

@app.post("/api/quiz/pdf")
async def quiz_pdf_enqueue(
    file: UploadFile = File(...),
    difficulty: str = Form("mixed"),
    topic: str = Form(""),
    num_questions: int = Form(0),
    user: dict = Depends(get_current_user)
):
    """
    P2.5: PDF processing is offloaded to the ARQ background worker.
    Returns a job_id immediately; clients poll /api/quiz/pdf/status/{job_id}.
    NOTE (frontend contract change): the old synchronous response shape is gone;
    see the provided `uploadPdfAndPoll()` JS function for the new client flow.
    """
    if ARQ_POOL is None:
        raise HTTPException(503, "Background worker is not configured. Set REDIS_URL and run the ARQ worker.")
    await check_rate_limit(user["id"])
    await db.reset_usage_if_needed(user["id"])
    usage = await db.get_user_usage(user["id"])
    tier = effective_tier_from_user(usage or user)
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    if usage["pdf_uploads_used"] >= limits["pdf_uploads"]:
        raise HTTPException(
            403,
            f"You have reached your monthly PDF upload limit ({limits['pdf_uploads']}). Please upgrade."
        )
    max_bytes = int(limits["pdf_max_mb"] * 1024 * 1024)
    size = getattr(file, "size", None)
    if size is not None and size > max_bytes:
        raise HTTPException(
            413,
            f"PDF too large. Your plan allows up to {limits['pdf_max_mb']}MB."
        )
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(422, "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise HTTPException(
            413,
            f"PDF too large. Your plan allows up to {limits['pdf_max_mb']}MB."
        )

    label = topic.strip() or Path(file.filename or "upload.pdf").stem
    language = (user.get("language") or "en").strip().lower()
    grade = normalize_grade(user.get("grade"))

    # Scaffold note: bytes are hex-encoded through Redis. For files > ~5MB in
    # production, upload to S3/object storage first and pass the URL instead.
    job = await ARQ_POOL.enqueue_job(
        "process_pdf_quiz",
        user["id"],
        data.hex(),
        difficulty,
        label,
        num_questions,
        language,
        grade,
        tier,
        limits["pdf_max_pages"],
        limits["pdf_max_ocr_pages"]
    )
    if not job:
        raise HTTPException(503, "Could not enqueue PDF job. Try again.")
    return {
        "job_id": job.job_id,
        "status": "queued"
    }

@app.get("/api/quiz/pdf/status/{job_id}")
async def quiz_pdf_status(job_id: str, user: dict = Depends(get_current_user)):
    """P2.5: poll endpoint for the background PDF job."""
    if ARQ_POOL is None:
        raise HTTPException(503, "Background worker is not configured.")
    job = Job(job_id, ARQ_POOL)
    info = await job.info()
    if not info:
        raise HTTPException(404, "Job not found.")
    # NOTE: job.info() returns a JobDef (queue metadata, no status/result).
    # In arq 0.28 those live on job.status() and job.result_info().
    status = await job.status()
    if status == JobStatus.complete:
        result_info = await job.result_info()
        raw = result_info.result if result_info else None
        if not isinstance(raw, dict):
            # Crashed job: arq stores the exception string as the result.
            return {"status": "error", "detail": "PDF processing failed. The file may be unreadable."}
        if raw.get("error"):
            return {"status": "error", "detail": str(raw["error"])}
        return {"status": "complete", "result": raw}
    # 'queued', 'in_progress', 'deferred' — client keeps polling.
    return {"status": status.value}

@app.post("/api/quiz/submit")
async def quiz_submit(payload: SubmitIn, user: Annotated[dict, Depends(get_current_user)]):
    if payload.correct > payload.total:
        raise HTTPException(422, "correct cannot exceed total.")
    topic_name = (payload.topic or payload.topic_name or "General").strip() or "General"
    result = await db.record_attempt(
        user_id=user["id"],
        topic_name=topic_name,
        correct=payload.correct,
        total=payload.total,
        source=payload.source
    )
    if payload.session_id:
        await db.complete_quiz_session(user["id"], payload.session_id)
    return {
        "saved": True,
        "topic": result
    }

@app.get("/api/quiz/sessions")
async def quiz_sessions(user: Annotated[dict, Depends(get_current_user)], limit: int = 20):
    return await db.get_quiz_sessions(user["id"], limit)

@app.get("/api/quiz/sessions/{session_id}")
async def quiz_session_detail(session_id: int, user: Annotated[dict, Depends(get_current_user)]):
    session = await db.get_quiz_session(user["id"], session_id)
    if not session:
        raise HTTPException(404, "Quiz session not found.")
    return session

# -----------------------------
# SCHEDULER
# -----------------------------
@app.get("/api/schedule")
async def schedule(user: Annotated[dict, Depends(get_current_user)]):
    return await db.get_schedule(user["id"])

@app.post("/api/tasks")
async def create_task(payload: TaskIn, user: Annotated[dict, Depends(get_current_user)]):
    return await db.add_task(user["id"], payload.title, payload.topic, payload.deadline)

@app.patch("/api/tasks/{task_id}")
async def toggle(task_id: int, user: Annotated[dict, Depends(get_current_user)]):
    return await db.toggle_task(user["id"], task_id)

@app.delete("/api/tasks/{task_id}")
async def remove(task_id: int, user: Annotated[dict, Depends(get_current_user)]):
    await db.delete_task(user["id"], task_id)
    return {"deleted": task_id}

# -----------------------------
# AI COACH
# -----------------------------
@app.get("/api/coach")
async def coach(
    user: Annotated[dict, Depends(get_current_user)],
    subjects: Optional[List[str]] = Query(None)
):
    await check_rate_limit(user["id"])
    topics = await db.get_topics(user["id"])
    sched = await db.get_schedule(user["id"])
    recent = await db.recent_attempts(user["id"], 6)
    
    # P-Fix: Sanitize topics to prevent crashes from NULL DB values
    safe_topics = []
    for t in topics:
        safe_topics.append({
            "name": t.get("name") or "Unknown Topic",
            "subject": t.get("subject") or "General",
            "score": float(t.get("score") if t.get("score") is not None else 50.0)
        })
        
    # P1.4: optional subject focus — when given, the weak-topic analysis and
    # fallback selection are restricted to those topics; default is unchanged.
    if subjects:
        wanted = {s.strip().lower() for s in subjects if s and s.strip()}
        safe_topics = [
            t for t in safe_topics
            if t["name"].strip().lower() in wanted or t["subject"].strip().lower() in wanted
        ]

    weak = sorted([t for t in safe_topics if t["score"] < 65], key=lambda t: t["score"])[:4]
    if not weak:
        weak = sorted(safe_topics, key=lambda t: t["score"])[:2]
        
    pcts = [a["pct"] for a in recent if a.get("pct") is not None]
    if len(pcts) >= 4:
        trend = (sum(pcts[:3]) / max(1, len(pcts[:3]))) - (sum(pcts[3:]) / max(1, len(pcts[3:])))
    else:
        trend = 0
        
    language = (user.get("language") or "en").strip().lower()
    if language == "ar":
        language_instruction = "Write the plan in Arabic."
    else:
        language_instruction = "Write the plan in English."
        
    # P-Fix: Safely extract schedule deadlines
    deadlines = []
    for t in sched.get("queue", [])[:3]:
        title = t.get("title") or "Task"
        tier = t.get("tier") or "PLANNED"
        deadline = t.get("deadline") or "soon"
        deadlines.append(f"- {title} [{tier}] due {deadline}")
    if not deadlines:
        deadlines = ["- none"]

    ctx = "\n".join([
        "Weak topics (mastery %):",
        *[f"- {t['name']} ({t['subject']}): {int(t['score'])}%" for t in weak],
        "Upcoming deadlines:",
        *deadlines,
        f"Recent accuracy trend: {'rising' if trend > 0 else 'falling'} ({round(trend, 1)} pts)."
    ])
    
    try:
        raw = await groq_chat_with_fallback(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are El-Peak, a sharp, encouraging study coach. "
                        "Reply with STRICT JSON only, no markdown: "
                        '{"headline":str,"focus":[{"topic":str,"why":str,"drill":str}],'
                        '"tips":[4 strings],"plan":[{"day":str,"task":str} with exactly 7 entries],'
                        '"motivation":str}. '
                        f"{language_instruction}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Build a 7-day revision plan from this student data:\n{ctx}"
                }
            ],
            tier=effective_tier_from_user(user),
            temperature=0.6,
            max_tokens=1200
        )
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("Coach response was not a JSON object.")
        data.setdefault("headline", "Your 7-day summit plan.")
        data.setdefault("focus", [])
        data.setdefault("tips", [])
        data.setdefault("plan", [])
        data.setdefault("motivation", "Small consistent reps beat rare heroic sessions.")
        data["ai"] = True
        return data
    except Exception as e:
        logger.warning("Coach fallback triggered: %s", str(e))
        return heuristic_coach(weak, trend)

def heuristic_coach(weak: List[dict], trend: float) -> dict:
    focus = [
        {
            "topic": t["name"],
            "why": f"Mastery at {int(t['score'])}% — below your 65% safety line.",
            "drill": f"15-min mixed drill on {t['name']}, then a 5-question quiz to lock it in."
        }
        for t in weak[:3]
    ]
    plan = []
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, day in enumerate(days):
        if weak:
            t = weak[i % len(weak)]
            if i % 2 == 0:
                task = f"Drill '{t['name']}' — 25 min focus block + mini-quiz."
            else:
                task = f"Retest '{t['name']}' and review wrong answers."
        else:
            task = "Maintenance set across your strongest topics."
        plan.append({
            "day": day,
            "task": task
        })
    return {
        "headline": f"Summit plan: shore up {len(weak)} weak ridge(s) before deadlines hit.",
        "focus": focus,
        "tips": [
            "Attack your weakest topic first each session.",
            "Use spaced repetition.",
            "Interleave subjects.",
            "Simulate pressure: time your quizzes."
        ],
        "plan": plan,
        "motivation": (
            "Your trend line is data, not destiny. Every rep on a weak topic raises the whole range."
            if trend <= 0 else
            "Momentum is on your side — keep the streak alive."
        ),
        "ai": False
    }
# -----------------------------
# AI TUTOR
# -----------------------------
@app.post("/api/ask")
async def ask_tutor(payload: AskIn, user: Annotated[dict, Depends(get_current_user)]):
    await check_rate_limit(user["id"])
    language = (user.get("language") or "en").strip().lower()
    if language == "ar":
        language_instruction = "If the user asks in Arabic, answer in Arabic. Otherwise answer clearly."
    else:
        language_instruction = "Answer primarily in English unless the user asks in Arabic."
    sanitized_history = []
    for item in payload.history[-12:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = str(item.get("content", ""))[:3000]
        if role not in ("user", "assistant"):
            continue
        if not content.strip():
            continue
        sanitized_history.append({
            "role": role,
            "content": content
        })
    messages = [
        {
            "role": "system",
            "content": (
                "You are El-Peak, a friendly, sharp AI study tutor. "
                "Explain concepts clearly. Use markdown for formatting. "
                "Keep answers concise but thorough. "
                f"{language_instruction}"
            )
        },
        *sanitized_history,
        {
            "role": "user",
            "content": payload.question[:5000]
        }
    ]
    try:
        reply = await groq_chat_with_fallback(
            messages=messages,
            tier=effective_tier_from_user(user),
            temperature=0.5,
            max_tokens=1000
        )
        return {"reply": reply}
    except Exception as e:
        logger.error("Tutor Chat Error: %s", str(e))
        raise HTTPException(503, "El-Peak is thinking too hard right now. Please try again.")

# -----------------------------
# DEV RUNNER
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=DEBUG
    )
