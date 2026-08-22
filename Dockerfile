FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "if [ -n \"$REDIS_URL\" ]; then (while true; do arq worker.WorkerSettings; sleep 2; done) & fi; exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-2}"]
