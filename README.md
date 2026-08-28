# El-Peak — AI-Powered Study Compass & Practice Engine

El-Peak is a full-stack AI study companion and Progressive Web App built with FastAPI, PostgreSQL, and Groq LLMs.

## Features

- Static Egyptian curriculum quiz banks
- AI-generated quizzes using Groq
- **Language-aware model routing (Round 19)**: Free EN→GPT-OSS-20B / AR→Qwen3.6-27B fast; Plus EN→GPT-OSS-120B / AR→Qwen deep-thinking; Pro→Qwen deep-thinking for all languages
- PDF-to-quiz conversion with OCR fallback
- Mastery tracking and analytics
- Urgency-weighted scheduler
- AI coach with 7-day revision plans
- AI tutor chat
- Free / Plus / Pro tiers
- **Paymob Pixel embedded checkout (Round 21)**: card form renders in-app (no redirect) with full branding + Arabic RTL; automatic hosted-iframe fallback; server-verified activation
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

## Deploying an update to Railway

The app is deployed from these files (Dockerfile + railway.json are included).
Two ways to ship an update:

### Option A — GitHub (recommended, auto-deploys)

1. Unzip `el-peak-deploy.zip` into your GitHub repo root (replace old files).
2. Commit + push:

```bash
git add -A
git commit -m "Round 19: language routing + boot screen + honest tier perks"
git push
```

3. Railway rebuilds automatically (healthcheck `/health`, restart ON_FAILURE).
4. Watch the deploy logs until "Deploy live" appears.

### Option B — Railway CLI (no GitHub)

```bash
npm i -g @railway/cli
railway login
cd el-peak-deploy        # the unzipped folder
railway link             # pick your El-Peak project + service
railway up
```

### Environment variables

No new variables are **required** — `GROQ_MODEL_QWEN` defaults to
`qwen/qwen3.6-27b` in code. Keep `DATABASE_URL`, `SECRET_KEY`, `GROQ_API_KEY`,
`REDIS_URL`, Paymob keys as they already are on Railway. Optional model
overrides are documented in `.env.example`.

**Round-21 optional:** set `PAYMOB_PUBLIC_KEY` (Dashboard → Settings →
Account Info, looks like `egy_pk_live_...`) to switch checkout from the
hosted-iframe redirect to **Paymob Pixel (Embedded Checkout)** — the card
form renders inside the app, branded, EN+AR, and payment completion is
confirmed server-side via `/api/payment/verify` (Paymob transaction inquiry
by `merchant_order_id`). Without it, or if the Intention API errors, the
checkout modal automatically embeds the classic hosted iframe
(`iframes/{iframe_id}?payment_token=...`) instead — same flow as before.

### Post-deploy smoke test — Pixel checkout (after setting PAYMOB_PUBLIC_KEY)

1. Profile → Upgrade Plan → pick a plan → **Proceed to Payment**: a Secure
   Checkout modal must open IN-PAGE (no redirect) with the plan/price pill
   and the card form styled to match the app (indigo accent, 12px radius).
2. Pay with a Paymob **test card** → the modal must flip to “Payment received
   — activating your plan…” then “Payment successful!” and close; the
   Profile tier badge should show the new tier within a few seconds.
3. Railway logs: one `POST /v1/intention/` per checkout (Pixel path), then
   `Pixel verify upgraded user <id> to <tier> (week)` after payment — or
   `Paymob Intention failed ... falling back to the hosted-iframe checkout`
   if your account isn't enabled for Intentions (then the hosted iframe
   still appears in the same modal — that is the designed fallback).
4. Arabic UI: the card form must render RTL with Arabic labels/placeholders
   (بيانات البطاقة / رقم البطاقة / ادفع).

### Post-deploy smoke test (2 minutes, first live run of routing)

1. Hard-refresh the site — you should see the new boot/loading screen with the
   "AI can make mistakes" note (Arabic if your app language is Arabic).
2. Generate an **English** quiz on a free account → Railway logs should show
   `model='openai/gpt-oss-20b'`.
3. Generate an **Arabic** quiz on a Plus/Pro account → logs should show
   `model='qwen/qwen3.6-27b'` with `reasoning_effort='default'` +
   `reasoning_format='hidden'`, and the quiz JSON must still parse.
4. Open Profile → the plan boxes must list exact perks (questions/PDF quotas,
   per-language models) and the "Current Plan" ribbon on your tier.
5. If Arabic quizzes ever fall back or fail JSON parsing, check Railway logs
   for a 429/model-not-found from Groq and verify `qwen/qwen3.6-27b` is
   available on your API key.

## Local Development

1. Create `.env` from `.env.example`.
2. Install dependencies:

```bash
pip install -r requirements.txt
