```markdown
# 🏔️ El-Peak — AI-Powered Study Compass & Practice Engine

**El-Peak** is a production-ready, full-stack AI Study Companion and Progressive Web App (PWA) designed to transform how students learn, practice, and revise. Built with **FastAPI**, **PostgreSQL**, and **Groq LLMs**, El-Peak offers dynamic quiz generation, OCR-based document analysis, intelligent priority scheduling, and automated 7-day revision planning.

---

## ✨ Key Features

### 🎯 1. Dynamic Quiz Engine
* **Curated Static Question Banks**: Pre-loaded subject banks organized by Egyptian educational stages (*Primary G1–6, Preparatory G7–9, Secondary G10–12, and General Knowledge*).
* **AI-Generated Practice Quizzes**: Generates tailored multiple-choice questions on any custom topic using Groq's high-speed inference.
* **PDF-to-Quiz Conversion**: Upload study materials, lecture notes, or textbook PDFs up to 15MB. Utilizes **PyMuPDF** and **PyTesseract OCR** for scanned documents, generating conceptual questions paired with exact contextual quotes.

### 🤖 2. Smart AI Coach & AI Tutor
* **Personalized 7-Day Revision Plans**: Analyzes topic mastery scores, upcoming deadlines, and accuracy trends to construct a custom weekly study schedule with a rule-based offline fallback.
* **Interactive AI Tutor Chat**: Context-aware AI tutor providing step-by-step explanations, code formatting, and concept breakdowns.

### 📅 3. Urgency-Weighted Priority Scheduler
* **Automated Task Prioritization**: Calculates real-time task priority scores using a weighted algorithm:
  $$\text{Priority} = 0.62 \times \text{Deadline Urgency} + 0.38 \times \text{Topic Weakness}$$
* **Radar Blips**: Automatically flags unscheduled topics where student mastery falls below 60%.

### 📊 4. Deep Analytics & Mastery Radar
* **Dynamic Peak Score**: Tracks overall study progression using an exponential smoothing mastery algorithm.
* **Visual Progress Tracking**: SVG-based Radar Chart for topic mastery, line graphs for accuracy trends, and an **84-day GitHub-style Activity Heatmap**.
* **Streak Counter**: Daily engagement tracker keeping students accountable.

### 🌐 5. Multilingual & Edu-Stage Adaptability
* **Full RTL & Arabic Support**: Seamless switching between English and Arabic with full Right-To-Left UI mirroring.
* **Curriculum Tailoring**: Adjusts question bank difficulty and content relative to school stages (*Primary, Preparatory, Secondary, General*).

### 💳 6. Tiered Subscriptions & Paymob Payments
* **Tiered Access Control**: Integrates flexible usage limits per billing cycle (*Free, Plus, Pro*).
* **Payment Integration**: Native Paymob iframe checkout and secure HMAC signature validation for automated tier upgrades.

---

## 🛠️ Tech Stack

### **Backend**
* **Framework**: Python 3.10+ / FastAPI v11.0
* **ASGI Server**: Uvicorn
* **Database**: PostgreSQL (with `asyncpg` connection pooling)
* **LLM Engine**: Groq API (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-20b`)
* **Document Processing**: PyMuPDF (`fitz`), PyTesseract OCR, Pillow
* **Security & Auth**: PyJWT, Bcrypt, `itsdangerous` URL safe serializer, Google OAuth2

### **Frontend**
* **Architecture**: Single Page Application (SPA) / Vanilla JS + HTML5
* **Styling**: Modern CSS variables, Glassmorphism, Dark/Light theme switching
* **Smooth Scrolling**: Lenis Smooth Scroll
* **Markdown Rendering**: Marked.js
* **PWA**: Web App Manifest integration (`manifest.json`)

---

## 💳 Tier Architecture & Limits

| Tier | Price (EGP) | Monthly AI Questions | Monthly PDF Uploads | AI Model Provider |
| :--- | :--- | :--- | :--- | :--- |
| **Free** | 0 EGP | 30 Questions | 3 Uploads (Max 1MB) | `llama-3.1-8b-instant` |
| **Plus** | 99 EGP / mo | 500 Questions | 50 Uploads (Max 15MB) | `openai/gpt-oss-20b` |
| **Pro** | 199 EGP / mo | 2,000 Questions | 200 Uploads (Max 15MB) | `llama-3.3-70b-versatile` |
