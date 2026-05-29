FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_FILE=/app/unsensored_image_and_video_generator_ai_advanced.py \
    SCREEN_WIDTH=1280 \
    SCREEN_HEIGHT=800 \
    HF_HOME=/home/appuser/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/home/appuser/.cache/huggingface/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    fluxbox \
    fonts-dejavu-core \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxext6 \
    libxkbcommon-x11-0 \
    novnc \
    websockify \
    x11vnc \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/Pictures/ImageGenerator /home/appuser/.cache/huggingface \
    && chmod +x /app/docker/start.sh \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 6080 5900

VOLUME ["/home/appuser/Pictures/ImageGenerator", "/home/appuser/.cache/huggingface"]

ENTRYPOINT ["/app/docker/start.sh"]
