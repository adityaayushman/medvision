# Build context is the repo root (see render.yaml: dockerContext: .), because
# the backend depends on the local `medchron` ML package in ../ml.
FROM python:3.12-slim

# opencv-python needs these even when no GUI is used (classic Docker gotcha).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ML core first (own layer — rebuilds only when imaging/model code changes)
COPY ml/pyproject.toml ml/pyproject.toml
COPY ml/src ml/src
RUN pip install --no-cache-dir ./ml

# CPU-only torch/torchvision — the deploy target has no GPU, and the CUDA
# wheels are ~10x larger for no benefit here. Verified working on CPU locally.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Backend
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/app backend/app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
WORKDIR /app/backend

# $PORT is injected by the host (e.g. Render); default to 8000 for `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
