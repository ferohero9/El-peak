"""
El-Peak — ARQ Background Worker (P2.5)
Processes CPU-heavy PDF/OCR tasks off the main web event loop.
Run with: arq worker.WorkerSettings
"""
import os
import io
import re
import json
import logging
import asyncio
from arq.worker import Worker
from arq.connections import RedisSettings

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

import httpx
import database as db

logger = logging.getLogger("el-peak.worker")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
QUIZ_SCHEMA = (
    '{"questions":[{"question": "...", "options":["A","B","C","D"],'
    '"answer_index":0, "explanation": "...", "context": "exact quote from text if pdf"}]}'
)

def _ocr_page(page, dpi: int = 130) -> str:
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
        if len(text) < 200:
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

async def process_pdf_quiz(ctx, user_id: int, data_hex: str, difficulty: str, topic: str, num_questions: int, language: str, grade: str):
    """
    ARQ Job: Extracts text from PDF, calls Groq, saves to DB.
    """
    await db.init_db()
    data = bytes.fromhex(data_hex)
    
    try:
        text = _extract_pdf_text_sync(data, 60, 15) # Using Pro limits as max safe bound
    except ValueError as e:
        return {"error": str(e)}
        
    if len(text) < 200:
        return {"error": "Not enough readable text in this PDF."}
        
    if not num_questions or num_questions <= 0:
        if len(text) > 6000: num_questions = 5
        elif len(text) > 2500: num_questions = 4
        else: num_questions = 3
        
    label = topic.strip() or "PDF Upload"
    excerpt = text[:12000]
    
    messages = [
        {
            "role": "system",
            "content": f"You are El-Peak's quiz engine. Reply with STRICT JSON only: {QUIZ_SCHEMA}"
        },
        {
            "role": "user",
            "content": f"Using ONLY the excerpt below from '{label}', write exactly {num_questions} {difficulty} multiple-choice questions.\n\nEXCERPT:\n" + excerpt
        }
    ]
    
    key = os.getenv("GROQ_API_KEY")
    payload = {"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.6, "max_tokens": 2000}
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"}, json=payload)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        
    # Simplified JSON extraction for worker
    try:
        data_dict = json.loads(raw)
        clean = data_dict.get("questions", [])
    except Exception:
        clean = []
        
    if not clean:
        return {"error": "Empty question set from PDF AI."}
        
    session_id = await db.create_quiz_session(user_id, label, "pdf", difficulty, clean)
    await db.increment_usage(user_id, pdf_uploads=1)
    await db.log_usage_event(user_id, "pdf_upload", 1, "pro") # Simplified tier logic for worker
    
    await db.close_pool()
    return {"questions": clean, "session_id": session_id, "topic": label}

class WorkerSettings:
    functions = [process_pdf_quiz]
    redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_URL", "redis://localhost:6379"))
    max_jobs = 2
    job_timeout = 300 # 5 minutes
