# DocuMind - RAG chatbot over uploaded documents.
#
# The sentence-embedding model is baked into the image so the container never
# needs to download it at runtime. The app listens on $PORT, which Hugging Face
# Spaces sets to 7860 and Render sets to 10000.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app \
    RAG_DATA_DIR=/data \
    RAG_HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY web ./web
COPY run.py serve.py ./

# Bake the embedding model in and prove it loads inside the image.
RUN adduser --disabled-password --gecos "" app \
 && python -c "from app.embeddings import _download_model; _download_model()" \
 && python -c "from app.embeddings import embed; print('embedding model OK ->', embed(['hello world']).shape)" \
 && mkdir -p /data \
 && chown -R app:app /app /home/app /data

USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-7860}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
