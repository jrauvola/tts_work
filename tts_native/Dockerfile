FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    rustc \
    cargo \
    pkg-config \
    libffi-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m pip install --upgrade pip setuptools wheel
COPY tts_native/requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Bypass interactive TOS prompt from coqui-tts
RUN mkdir -p /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 && \
    echo "Agreed to Coqui terms via Dockerfile." > /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/tos_agreed.txt

COPY protos/gen/python /app/protos/gen/python
COPY tts_common /app/tts_common
COPY tts_native /app/tts_native

ENV PYTHONPATH=/app/protos/gen/python:/app
EXPOSE 50052 8004
CMD ["python3", "-m", "tts_native.server"]

