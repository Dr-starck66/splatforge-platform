# ══════════════════════════════════════════════════════════
# SPLAT·FORGE — GPU Dockerfile
# Base: CUDA 12.1 + PyTorch 2.x + gsplat + pycolmap
# ══════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ── System deps ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git curl ffmpeg libgl1 libglib2.0-0 \
    colmap libcolmap-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python python3.11 1

WORKDIR /app

# ── Python deps (layered for cache efficiency) ─────────────
COPY requirements.txt .

# PyTorch first (large, cache separately)
RUN pip install --upgrade pip && \
    pip install torch==2.2.0 torchvision==0.17.0 \
        --index-url https://download.pytorch.org/whl/cu121

# gsplat compiles CUDA kernels at install time
RUN pip install gsplat==1.3.0

# Rest of deps
RUN pip install -r requirements.txt

# ── App ────────────────────────────────────────────────────
COPY app/ ./app/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
