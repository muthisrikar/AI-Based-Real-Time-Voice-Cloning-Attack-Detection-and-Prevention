# ==============================================================================
# AegisVoice: Self-Contained AI Voice Detection Container with FFmpeg
# ==============================================================================
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Install system dependencies: FFmpeg (with WebM/Opus codecs) and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with CUDA 12.4 GPU support for NVIDIA RTX 4050
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and static assets
COPY . /app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=15s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8000/api/health || exit 0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
