# El-Peak — AI-Powered Study Compass & Practice Engine

El-Peak is a full-stack AI study companion and Progressive Web App built with FastAPI, PostgreSQL, and Groq LLMs.

## Features

- Static Egyptian curriculum quiz banks
- AI-generated quizzes using Groq
- **Compound + Qwen3.8 model architecture (Round 22)**: GPT-OSS dropped entirely. Free EN→`groq/compound-mini` (70K TPM) / AR→`qwen3.6-27b` (fast, brevity-capped); Plus→`qwen3.8-27b` at LOW reasoning effort (2M TPD); Pro EN→`groq/compound` (deep multi-step) / AR→`qwen3.8-27b` at HIGH effort (native xhigh). Strict JSON mode (`response_format: json_object`) on every quiz/plan call + a client-side AI request queue (≥2.5s spacing) with visible queue states
- PDF-to-quiz conversion with OCR fallback
- Mastery tracking and analytics
- Urgency-weighted scheduler
- AI coach with 7-day revision plans
- AI tutor chat
- Free / Plus / Pro tiers
- **Paymob Pixel embedded checkout (Round 21)**: card form renders in-app (no redirect) with full branding + Arabic RTL; automatic hosted-iframe fallback; server-verified activation
- **Automatic payment activation (Round 23)**: the payment modal silently polls `/api/payment/verify` while it is open (8s first check, then every 5s for ~2min, then every 15s, 20min cap; immediate re-check on any iframe navigation), so the plan flips to active — success UI, auto-close, toast — with ZERO button presses. The manual button remains as an optional fallback ("Check payment status")
- **Liquid-glass UI (Round 23)**: all glass surfaces run `backdrop-filter: blur() saturate(180%)` (new `--glass-bf` variable); background gradient stops roughly doubled for a colorful sky + a 4th pink aurora blob; low-end devices keep a 14px glass blur and a static (GPU-free) colorful aurora instead of losing the whole experience; low-end detection no longer misclassifies ordinary 4-core laptops
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

### Post-deploy smoke test (2 minutes, first live run of the Round-22 routing)

1. Hard-refresh the site — you should see the new boot/loading screen with the
   "AI can make mistakes" note (Arabic if your app language is Arabic).
2. Generate an **English** quiz on a free account → Railway logs should show
   `model='groq/compound-mini'`, and the outbound payload must include
   `response_format={'type': 'json_object'}`.
3. Generate an **Arabic** quiz on a Plus account → logs should show
   `model='qwen/qwen3.8-27b'` with `reasoning_effort='low'` +
   `reasoning_format='hidden'`; on a Pro account `reasoning_effort='high'`
   (the model's native xhigh). The quiz JSON must still parse and be in Arabic
   script. A **free** Arabic quiz runs on `qwen/qwen3.6-27b` with
   `reasoning_effort='none'` and short explanations (brevity-capped).
4. Open Profile → plan boxes must list the new per-language models
   (Compound-Mini/Qwen3.6-27B · Qwen3.8-27B · Compound/Qwen3.8-27B with the
   Deep-Thinking pills on Pro only) and the "Current Plan" ribbon on your tier.
5. Fire two AI actions within a second (e.g. ask the tutor, then generate a
   quiz): the second one shows an "in the AI queue" state and starts ≥2.5s
   after the first — no 429s.
6. If a model 400s on an optional param, logs show `retrying once with a
   minimal payload` and the call still succeeds (self-healing). If quizzes
   ever fail JSON parsing, check Railway logs for a 429/model-not-found from
   Groq and verify `groq/compound-mini`, `groq/compound`, `qwen/qwen3.6-27b`
   and `qwen/qwen3.8-27b` are all available on your API key.

## Local Development

1. Create `.env` from `.env.example`.
2. Install dependencies:

```bash
pip install -r requirements.txt
