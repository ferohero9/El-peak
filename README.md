# El-Peak — AI-Powered Study Compass & Practice Engine

El-Peak is a full-stack AI study companion and Progressive Web App built with FastAPI, PostgreSQL, and Groq LLMs.

## Features

- Static Egyptian curriculum quiz banks
- AI-generated quizzes using Groq
- PDF-to-quiz conversion with OCR fallback
- Mastery tracking and analytics
- Urgency-weighted scheduler
- AI coach with 7-day revision plans
- AI tutor chat
- Free / Plus / Pro tiers
- Paymob checkout and webhook upgrades
- Google OAuth
- PWA support

## Stack

- FastAPI
- PostgreSQL / Neon
- asyncpg
- Groq API
- PyMuPDF
- Tesseract OCR
- Vanilla JS SPA
- PWA manifest + service worker

## Local Development

1. Create `.env` from `.env.example`.
2. Install dependencies:

```bash
pip install -r requirements.txt
