# Hugging Face Docker Space: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

# libGL/glib for OpenCV, which ultralytics pulls in.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=7860
WORKDIR /home/user/app

# CPU-only torch first, so requirements.txt doesn't pull the multi-GB CUDA build.
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app ./app

EXPOSE 7860
# One worker keeps the rate limiter, retention thread and loaded models in one
# process; threads handle concurrent requests. /data is the Space's persistent
# storage when attached; otherwise projects live on the container's disk.
CMD ["sh", "-c", "if [ -w /data ]; then export GEL_STORAGE_DIR=/data; fi; exec gunicorn --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:${PORT} app.wsgi:app"]
