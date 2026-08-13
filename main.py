"""
El-Peak — FastAPI backend v4 (Multi-User Auth).
"""
import io
import os
import random
import re
import json
import datetime as dt
from pathlib import Path
from typing import Annotated

import httpx
import pymupdf as fitz
import pytesseract
from PIL import Image
import fastapi
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, status, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
import jwt
import bcrypt
import secrets
from itsdangerous import URLSafeSerializer

import database as db

try:
    import fitz
    import pytesseract
    from PIL import Image
    PDF_OCR_ENABLED = True
except ImportError:
    PDF_OCR_ENABLED = False

env_path = Path(__file__).parent / "key.env"
try:
    from dotenv import load_dotenv
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass
# Paymob Config
PAYMOB_API_KEY = os.getenv("PAYMOB_API_KEY")
PAYMOB_INTEGRATION_ID = os.getenv("PAYMOB_INTEGRATION_ID", "5853170") 
app = FastAPI(title="El-Peak API", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
INDEX = Path(__file__).parent / "templates" / "index.html"

# --- AUTH CONFIG ---
SECRET_KEY = os.getenv("SECRET_KEY", "el_peak_super_secret_dev_key_change_in_production")
ALGORITHM = "HS256"
def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    pwd_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(pwd_bytes, hashed_bytes)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = dt.datetime.utcnow() + dt.timedelta(days=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        # Cast the string to an integer for PostgreSQL
        user_id = int(user_id_str)
    except jwt.PyJWTError:
        raise credentials_exception
    
    user_email = payload.get("email")
    if user_id is None or user_email is None:
        raise credentials_exception
    return {"id": user_id, "email": user_email}

@app.on_event("startup")
async def startup():
    await db.init_db()
    if os.getenv("GROQ_API_KEY"):
        print("✅ GROQ_API_KEY found!")

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(INDEX)

@app.get("/manifest.json", include_in_schema=False)
async def manifest():
    return {
        "name": "El-Peak Study Compass",
        "short_name": "El-Peak",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#070710",
        "theme_color": "#070710",
        "icons": [
            {
                "src": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='none' stroke='%236366f1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M2 18 L7 4 L10 12 L13 6 L18 18'/%3E%3C/svg%3E",
                "sizes": "any",
                "type": "image/svg+xml"
            }
        ]
    }

# Google OAuth Config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "https://el-peak-production.up.railway.app/api/auth/google/callback")
session_serializer = URLSafeSerializer(SECRET_KEY, salt="oauth")
# Tier Limits Config
TIER_LIMITS = {
    "free": {"ai_questions": 30, "pdf_uploads": 3},
    "plus": {"ai_questions": 500, "pdf_uploads": 50},
    "pro": {"ai_questions": 2000, "pdf_uploads": 200}
}

# Models per tier (Free uses 8B to save resources, Plus/Pro use 70B)
TIER_MODELS = {
    "free": "llama-3.1-8b-instant",
    "plus": "llama-3.3-70b-versatile",
    "pro": "llama-3.3-70b-versatile"
}
# ================================================================ AUTH ROUTES
class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)

@app.post("/api/auth/signup")
async def signup(payload: SignupIn):
    existing = await db.get_user_by_email(payload.email)
    if existing:
        raise HTTPException(400, "Email already registered")
    hashed_pw = hash_password(payload.password)  # <-- Changed this
    user = await db.create_user(payload.email, hashed_pw)
    if not user:
        raise HTTPException(500, "Failed to create user")
    token = create_access_token({"sub": str(user["id"]), "email": user["email"]})
    return {"token": token, "email": user["email"]}

@app.post("/api/auth/login")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await db.get_user_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):  # <-- Changed this
        raise HTTPException(401, "Incorrect email or password")
    await db.cleanup_old_pdf_attempts()
    token = create_access_token({"sub": str(user["id"]), "email": user["email"]})
    return {"token": token, "email": user["email"]}

@app.get("/api/auth/google")
async def google_login(request: Request):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(500, "Google OAuth not configured")
    
    state = session_serializer.dumps({"nonce": secrets.token_urlsafe(16)})
    auth_url = (
        f"https://accounts.google.com/o/oauth2/auth"
        f"?response_type=code&client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope=openid email profile&state={state}"
    )
    return {"auth_url": auth_url, "state": state}

@app.get("/api/auth/google/callback")
async def google_callback(request: Request, code: str, state: str):
    # Exchange code for token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        token_resp.raise_for_status()
        access_token = token_resp.json().get("access_token")
        
        # Get user info
        user_resp = await client.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_resp.raise_for_status()
        google_data = user_resp.json()

    email = google_data.get("email")
    if not email:
        raise HTTPException(400, "Could not get email from Google")
        
    # Find or create user
    user = await db.get_user_by_email(email)
    if not user:
        # Create a dummy password for Google users (they won't use it)
        import hashlib
        dummy_pw = hashlib.sha256(f"google_{email}".encode()).hexdigest()
        user = await db.create_user(email, dummy_pw)
        if not user:
            raise HTTPException(500, "Failed to create user")

    # Issue JWT
    jwt_token = create_access_token({"sub": str(user["id"]), "email": user["email"]})
    
    # Redirect to frontend with token in URL
    return RedirectResponse(url=f"/?token={jwt_token}")
@app.get("/api/usage")
async def get_usage(user: Annotated[dict, Depends(get_current_user)]):
    usage = await db.get_user_usage(user["id"])
    if not usage: raise HTTPException(404, "User not found")
    
    tier = usage["tier"]
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    
    return {
        "tier": tier,
        "ai_used": usage["ai_questions_used"],
        "ai_limit": limits["ai_questions"],
        "pdf_used": usage["pdf_uploads_used"],
        "pdf_limit": limits["pdf_uploads"]
    }

class CheckoutIn(BaseModel):
    tier: str # Must be "plus" or "pro"

# Pricing in Piasters (1 EGP = 100 Piasters)
TIER_PRICES = {
    "plus": 9900,  # 99 EGP
    "pro": 19900   # 199 EGP
}

@app.post("/api/payment/checkout")
async def create_checkout(payload: CheckoutIn, user: Annotated[dict, Depends(get_current_user)]):
    if payload.tier not in TIER_PRICES:
        raise HTTPException(400, "Invalid tier selected")
    
    if not PAYMOB_API_KEY:
        raise HTTPException(500, "Payment gateway not configured")

    amount = TIER_PRICES[payload.tier]
    # We hide the user_id and tier in the merchant_order_id so the webhook knows who to upgrade!
    merchant_order_id = f"user_{user['id']}_{payload.tier}"

    async with httpx.AsyncClient() as client:
        # 1. Authenticate with Paymob
        auth_resp = await client.post(
            "https://accept.paymob.com/api/auth/tokens",
            json={"api_key": PAYMOB_API_KEY}
        )
        auth_resp.raise_for_status()
        auth_token = auth_resp.json().get("token")
        if not auth_token:
            raise HTTPException(500, "Failed to authenticate with Paymob")

        # 2. Register Order
        order_resp = await client.post(
            "https://accept.paymob.com/api/ecommerce/orders",
            json={
                "auth_token": auth_token,
                "delivery_needed": "false",
                "amount_cents": amount,
                "currency": "EGP",
                "merchant_order_id": merchant_order_id,
                "items": []
            }
        )
        order_resp.raise_for_status()
        paymob_order_id = order_resp.json().get("id")

        # 3. Generate Payment Key
        payment_key_resp = await client.post(
            "https://accept.paymob.com/api/acceptance/payment_keys",
            json={
                "auth_token": auth_token,
                "amount_cents": amount,
                "expiration": 3600,
                "order_id": paymob_order_id,
                "billing_data": {
                    "apartment": "NA", "email": user["email"], "floor": "NA",
                    "first_name": "El-Peak", "street": "NA", "building": "NA",
                    "phone_number": "01000000000", "shipping_method": "NA",
                    "postal_code": "NA", "city": "Cairo", "country": "EG",
                    "last_name": "User", "state": "NA"
                },
                "currency": "EGP",
                "integration_id": int(PAYMOB_INTEGRATION_ID)
            }
        )
        payment_key_resp.raise_for_status()
        payment_token = payment_key_resp.json().get("token")

    if not payment_token:
        raise HTTPException(500, "Failed to generate payment key")

    # Return the URL the frontend should redirect the user to
    iframe_url = f"https://accept.paymob.com/api/acceptance/iframes/{PAYMOB_INTEGRATION_ID}?payment_token={payment_token}"
    
    # Note: Paymob's standard iframe URL format is usually:
    # https://accept.paymob.com/api/acceptance/iframes/{IFRAME_ID}?payment_token={TOKEN}
    # If you have a specific iframe ID from your Paymob dashboard, replace PAYMOB_INTEGRATION_ID in the URL above with it.
    
    return {"checkout_url": iframe_url}

import hmac
import hashlib

@app.post("/api/payment/webhook")
async def payment_webhook(request: fastapi.Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid payload")

    # --- CRITICAL: HMAC Security Verification ---
    hmac_secret = os.getenv("PAYMOB_HMAC_SECRET", "")
    if hmac_secret:
        req_hmac = request.headers.get("X-PAYMOB-HMAC", "")
        # Paymob sends the HMAC in a specific concatenated format
        obj = payload.get("obj", {})
        concat = "".join([
            str(obj.get("amount_cents", "")),
            str(obj.get("created_at", "")),
            str(obj.get("currency", "")),
            str(obj.get("error_occured", "")),
            str(obj.get("has_parent_transaction", "")),
            str(obj.get("id", "")),
            str(obj.get("integration_id", "")),
            str(obj.get("is_hmac_authenticated", "")),
            str(obj.get("is_auth", "")),
            str(obj.get("is_capture", "")),
            str(obj.get("is_refunded", "")),
            str(obj.get("is_standalone_payment", "")),
            str(obj.get("is_voided", "")),
            str(obj.get("order", {}).get("id", "")),
            str(obj.get("owner", "")),
            str(obj.get("pending", "")),
            str(obj.get("source_data", {}).get("pan", "")),
            str(obj.get("source_data", {}).get("sub_type", "")),
            str(obj.get("source_data", {}).get("type", "")),
            str(obj.get("success", ""))
        ])
        calculated_hmac = hmac.new(hmac_secret.encode('utf-8'), concat.encode('utf-8'), hashlib.sha512).hexdigest()
        
        if calculated_hmac != req_hmac:
            print("❌ Paymob HMAC Verification Failed!")
            raise HTTPException(403, "Invalid HMAC signature")

    # --- Process Successful Payment ---
    if payload.get("success") == "true" or payload.get("obj", {}).get("success") == "true":
        obj = payload.get("obj", {})
        order_obj = obj.get("order", {})
        merchant_order_id = order_obj.get("merchant_order_id", "")
        
        parts = merchant_order_id.split("_")
        if len(parts) == 3:
            user_id = int(parts[1])
            tier = parts[2]
            print(f"✅ Upgrading User {user_id} to {tier}!")
            await db.upgrade_user_tier(user_id, tier)
            return {"status": "success"}
            
    return {"status": "ignored"}
# ================================================================ Schemas
class SubmitIn(BaseModel):
    topic_name: str = Field(min_length=1)
    correct: int = Field(ge=0)
    total: int = Field(ge=1)
    source: str = "static"

class AIQuizIn(BaseModel):
    topic: str = Field(min_length=1)
    num_questions: int = Field(default=4, ge=1, le=30)
    difficulty: str = "mixed"

class TaskIn(BaseModel):
    title: str = Field(min_length=1)
    topic: str | None = None
    deadline: str = Field(min_length=10)

class AskIn(BaseModel):
    question: str = Field(min_length=1)
    history: list = []

# ================================================================ Dashboard
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

# ================================================================ Quiz Engine
STATIC_QUIZZES = {
    "Mathematics": [
        {"question": "What is the derivative of x³?", "options": ["3x²", "x²", "3x", "x³/3"], "answer_index": 0, "explanation": "Power rule: d/dx(xⁿ) = n·xⁿ⁻¹, so 3·x²."},
        {"question": "Evaluate ∫ 2x dx.", "options": ["x² + C", "2x² + C", "x + C", "2 + C"], "answer_index": 0, "explanation": "Reverse power rule: ∫2x dx = x² + C."},
        {"question": "Solve x² − 5x + 6 = 0.", "options": ["x = 2, 3", "x = −2, −3", "x = 1, 6", "x = 0, 5"], "answer_index": 0, "explanation": "Factors as (x−2)(x−3) = 0."},
        {"question": "lim(x→0) sin(x)/x = ?", "options": ["1", "0", "∞", "undefined"], "answer_index": 0, "explanation": "A foundational limit: sin(x)/x → 1 as x → 0."},
        {"question": "Slope of the tangent to y = x² at x = 2?", "options": ["4", "2", "8", "1"], "answer_index": 0, "explanation": "dy/dx = 2x; at x=2 the slope is 4."},
    ],
    "Physics": [
        {"question": "The SI unit of force is the…", "options": ["Newton", "Joule", "Watt", "Pascal"], "answer_index": 0, "explanation": "1 N = 1 kg·m/s²."},
        {"question": "Earth's gravitational acceleration ≈ ?", "options": ["9.8 m/s²", "9.8 m/s", "3.0 m/s²", "12.4 m/s²"], "answer_index": 0, "explanation": "g ≈ 9.81 m/s² at sea level."},
        {"question": "Kinetic energy is given by…", "options": ["½mv²", "mv", "mgh", "ma"], "answer_index": 0, "explanation": "KE = ½mv²."},
        {"question": "Newton's First Law describes…", "options": ["Inertia", "Action–reaction", "F = ma", "Universal gravitation"], "answer_index": 0, "explanation": "Bodies maintain their state of motion unless acted on by a net force."},
        {"question": "Speed of light in vacuum ≈ ?", "options": ["3×10⁸ m/s", "3×10⁶ m/s", "3×10¹⁰ m/s", "9.8 m/s"], "answer_index": 0, "explanation": "c ≈ 299,792,458 m/s."},
    ],
    "Computer Science": [
        {"question": "Big-O of binary search on a sorted array?", "options": ["O(log n)", "O(n)", "O(n log n)", "O(1)"], "answer_index": 0, "explanation": "Each step halves the search space → logarithmic."},
        {"question": "Which structure is FIFO?", "options": ["Queue", "Stack", "Tree", "Hash map"], "answer_index": 0, "explanation": "First-In-First-Out: a queue."},
        {"question": "HTTP 404 means…", "options": ["Not Found", "Server Error", "Unauthorized", "Redirect"], "answer_index": 0, "explanation": "404 = resource not found."},
        {"question": "Which keyword defines a function in Python?", "options": ["def", "func", "fn", "lambda only"], "answer_index": 0, "explanation": "`def name():`."},
        {"question": "Which SQL statement retrieves data?", "options": ["SELECT", "INSERT", "UPDATE", "ALTER"], "answer_index": 0, "explanation": "SELECT queries rows."},
    ],
    "History": [
        {"question": "World War II ended in…", "options": ["1945", "1944", "1946", "1939"], "answer_index": 0, "explanation": "Europe surrendered May 1945."},
        {"question": "The first crewed Moon landing was in…", "options": ["1969", "1965", "1972", "1959"], "answer_index": 0, "explanation": "Apollo 11, July 20 1969."},
        {"question": "The Renaissance began in…", "options": ["Italy", "France", "England", "Spain"], "answer_index": 0, "explanation": "Started in Italian city-states like Florence in the 14th c."},
        {"question": "First President of the United States?", "options": ["George Washington", "Thomas Jefferson", "John Adams", "Lincoln"], "answer_index": 0, "explanation": "Washington served 1789–1797."},
        {"question": "The Berlin Wall fell in…", "options": ["1989", "1991", "1985", "1993"], "answer_index": 0, "explanation": "November 9, 1989."},
    ],
    "Biology": [
        {"question": "The 'powerhouse of the cell' is the…", "options": ["Mitochondrion", "Nucleus", "Ribosome", "Golgi body"], "answer_index": 0, "explanation": "Mitochondria produce ATP."},
        {"question": "Plants convert light energy via…", "options": ["Photosynthesis", "Respiration", "Fermentation", "Osmosis"], "answer_index": 0, "explanation": "6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂."},
        {"question": "DNA's structure is a…", "options": ["Double helix", "Single strand", "Triple ring", "Sheet"], "answer_index": 0, "explanation": "Watson & Crick (1953)."},
        {"question": "Red blood cells primarily carry…", "options": ["Oxygen", "Hormones", "Antibodies", "Glucose"], "answer_index": 0, "explanation": "Hemoglobin transports O₂."},
        {"question": "The largest organ of the human body is the…", "options": ["Skin", "Liver", "Brain", "Lungs"], "answer_index": 0, "explanation": "Skin is the largest organ."},
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
async def quiz_subjects():
    return {"subjects": list(STATIC_QUIZZES.keys())}

@app.get("/api/quiz/static")
async def quiz_static(user: Annotated[dict, Depends(get_current_user)], topic: str = "Mathematics", n: int = 4):
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
                "question": str(q["question"]), "options": opts,
                "answer_index": idx, "explanation": str(q.get("explanation", "—")),
                "context": str(q.get("context", ""))
            })
    return clean

QUIZ_SCHEMA = ('{"questions":[{"question":"...","options":["A","B","C","D"],"answer_index":0,"explanation":"...","context":"exact quote from text if pdf"}]}')

@app.post("/api/quiz/ai")
async def quiz_ai(payload: AIQuizIn, user: Annotated[dict, Depends(get_current_user)]):
    # --- PAYWALL LOGIC ---
    usage = await db.get_user_usage(user["id"])
    tier = usage["tier"]
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    
    if usage["ai_questions_used"] + payload.num_questions > limits["ai_questions"]:
        raise HTTPException(403, f"You have reached your monthly AI question limit ({limits['ai_questions']}). Please upgrade to Plus.")
    
    groq_model = TIER_MODELS.get(tier, TIER_MODELS["free"])
    
    messages = [
        {"role": "system", "content": "You are El-Peak's quiz engine. Reply with STRICT JSON only — no prose, no markdown — matching exactly this schema: " + QUIZ_SCHEMA + ". Vary the correct answer position."},
        {"role": "user", "content": f"Generate exactly {payload.num_questions} {payload.difficulty} multiple-choice questions about '{payload.topic}' for a student. Each must have 4 plausible options and a one-sentence explanation."},
    ]
    try:
        max_tokens = max(1200, payload.num_questions * 250)
        raw = await groq_chat(messages, temperature=0.7, max_tokens=max_tokens, model_override=groq_model)
        clean = _clean_questions(extract_json(raw), payload.num_questions)
        if not clean: 
            raise ValueError("empty question set")
        
        # Increment usage ONLY on success
        await db.increment_usage(user["id"], ai_questions=payload.num_questions)
        
        return {"topic": payload.topic, "subject": "AI", "questions": clean, "ai": True, "source": "ai"}
    except Exception as e:
        print(f"⚠️ AI Quiz Error: {e}")
        fallback = _match_static(payload.topic, 4)
        fallback["note"] = "AI unavailable — served from the static bank."
        return fallback

@app.post("/api/quiz/pdf")
async def quiz_pdf(file: UploadFile = File(...), difficulty: str = Form("mixed"), topic: str = Form(""), num_questions: int = Form(0), user: dict = Depends(get_current_user)):
    if not PDF_OCR_ENABLED:
        raise HTTPException(500, "PDF libraries not installed on server.")
        
    # --- PAYWALL LOGIC ---
    usage = await db.get_user_usage(user["id"])
    tier = usage["tier"]
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    
    if usage["pdf_uploads_used"] >= limits["pdf_uploads"]:
        raise HTTPException(403, f"You have reached your monthly PDF upload limit ({limits['pdf_uploads']}). Please upgrade to Plus.")

    data = await file.read()
    if len(data) > 15 * 1024 * 1024: 
        raise HTTPException(413, "PDF too large (max 15 MB).")
    
    text = ""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        for page in doc[:40]: 
            text += page.get_text()
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

    if len(text) < 200: 
        raise HTTPException(422, "Not enough readable text in this PDF (even with OCR).")

    if not num_questions or num_questions == 0:
        if len(text) > 6000: auto_n = 5
        elif len(text) > 2500: auto_n = 4
        else: auto_n = 3
        num_questions = auto_n

    label = topic.strip() or Path(file.filename or "upload.pdf").stem
    excerpt = text[:9000]
    
    groq_model = TIER_MODELS.get(tier, TIER_MODELS["free"])
    
    messages = [
        {"role": "system", "content": "You are El-Peak's quiz engine. Reply with STRICT JSON only — no prose, no markdown — matching exactly this schema: " + QUIZ_SCHEMA + ". Vary the correct answer position."},
        {"role": "user", "content": f"Using ONLY the excerpt below from the student's document '{label}', write exactly {num_questions} {difficulty} multiple-choice questions. Make them smart: test conceptual understanding, application, or inference rather than just rote memorization. Each must have 4 plausible options, a one-sentence explanation, and a 'context' field containing an EXACT quote (max 300 chars) from the text.\n\nEXCERPT:\n" + excerpt},
    ]
    
    try:
        raw = await groq_chat(messages, temperature=0.6, max_tokens=2000, model_override=groq_model)
        clean = _clean_questions(extract_json(raw), num_questions)
        if not clean: 
            raise ValueError("empty question set")
            
        await db.increment_usage(user["id"], pdf_uploads=1)
        
        return {"topic": label, "subject": "PDF", "questions": clean, "ai": True, "source": "pdf"}
    except Exception as e:
        print(f"⚠️ AI PDF Quiz Error: {e}")
        raise HTTPException(503, "AI is unavailable right now.")

@app.post("/api/quiz/submit")
async def quiz_submit(payload: SubmitIn, user: Annotated[dict, Depends(get_current_user)]):
    if payload.correct > payload.total: raise HTTPException(422, "correct cannot exceed total")
    result = await db.record_attempt(user["id"], payload.topic_name, payload.correct, payload.total, payload.source)
    return {"saved": True, "topic": result}

# ================================================================ Scheduler
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

# ================================================================ AI Coach
@app.get("/api/coach")
async def coach(user: Annotated[dict, Depends(get_current_user)]):
    topics = await db.get_topics(user["id"])
    sched = await db.get_schedule(user["id"])
    recent = await db.recent_attempts(user["id"], 6)
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
        "tips": ["Attack your weakest topic first each session.", "Use spaced repetition.", "Interleave subjects.", "Simulate pressure: time your quizzes."],
        "plan": plan,
        "motivation": "Your trend line is data, not destiny. Every rep on a weak topic raises the whole range." if trend <= 0 else "Momentum is on your side — keep the streak alive.",
        "ai": False,
    }

# ================================================================ AI Tutor Chat
@app.post("/api/ask")
async def ask_tutor(payload: AskIn, user: Annotated[dict, Depends(get_current_user)]):
    messages = [
        {"role": "system", "content": "You are El-Peak, a friendly, sharp AI study tutor. Explain concepts clearly. Use markdown for formatting. Keep answers concise but thorough."},
        *payload.history,
        {"role": "user", "content": payload.question}
    ]
    try:
        reply = await groq_chat(messages, temperature=0.5, max_tokens=1000)
        return {"reply": reply}
    except Exception as e:
        print(f"⚠️ Tutor Chat Error: {e}")
        raise HTTPException(503, "El-Peak is thinking too hard right now.")

@app.post("/api/payment/webhook")
async def payment_webhook(request: fastapi.Request):
    # Paymob will send a POST request here when payment is successful
    data = await request.json()
    # In production, you MUST verify the HMAC signature from Paymob here
    # to ensure the request is actually from Paymob.
    
    if data.get("success") == "true":
        # Extract user ID and tier from the merchant_order_id (e.g., "user_5_plus")
        order_id = data.get("order", {}).get("merchant_order_id", "")
        parts = order_id.split("_")
        if len(parts) == 3:
            user_id = int(parts[1])
            tier = parts[2]
            await db.upgrade_user_tier(user_id, tier)
            return {"status": "success"}
            
    return {"status": "ignored"}

# ================================================================ Groq Core
async def groq_chat(messages: list, temperature: float = 0.6, max_tokens: int = 1200, model_override: str = None) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key: raise RuntimeError("Missing GROQ_API_KEY.")
    
    # Use override if provided (for Pro/Plus tiers), else use default from .env
    model_to_use = model_override if model_override else GROQ_MODEL
    
    payload = {"model": model_to_use, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    # ... keep the rest of the function the same ...

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
