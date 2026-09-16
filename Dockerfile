# Qwazon Dockerfile — ziemniak w kontenerze (CPU only, ~1GB)
# Build: docker build -t qwazon:0.2 .
# Run:   docker run -p 8000:8000 -p 7860:7860 qwazon:0.2
# Test:  curl http://localhost:8000/health

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (dla tokenizers, torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps — najpierw requirements dla cache
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn gradio tokenizers

# Torch CPU (lżejszy niż CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Kod
COPY qwazon/ ./qwazon/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY demo/ ./demo/
COPY checkpoints/qwazon-nano/config.json ./checkpoints/qwazon-nano/config.json
COPY checkpoints/qwazon-micro/config.json ./checkpoints/qwazon-micro/config.json
COPY tokenizer-qwazon/ ./tokenizer-qwazon/ 2>/dev/null || true

# Domyślne checkpointy to config only (wagi pobierz z HF Hub lub trenuj)
# Dla demo: micro jest mały, można skopiować wagi (11MB) jeśli chcesz
# COPY checkpoints/qwazon-micro/pytorch_model.bin ./checkpoints/qwazon-micro/pytorch_model.bin

EXPOSE 8000 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Domyślnie: API + Gradio
CMD ["sh", "-c", "uvicorn qwazon.api:app --host 0.0.0.0 --port 8000 & python demo/app.py --host 0.0.0.0 --port 7860 --checkpoint checkpoints/qwazon-nano"]
