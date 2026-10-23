# Generic image; also used as-is by the Hugging Face Space (port 7860, runs as uid 1000).
FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user server ./server
COPY --chown=user app ./app
COPY --chown=user scripts ./scripts
COPY --chown=user data ./data
COPY --chown=user annotations ./annotations
RUN mkdir -p var images && chown -R user:user /home/user/app
USER user
ENV PORT=7860 WV_DB=/home/user/app/var/wheelview.db WV_IMAGE_DIR=/home/user/app/images PYTHONUNBUFFERED=1
EXPOSE 7860
# 1) import events committed to git (idempotent); 2) serve. With WV_HF_DATASET set, the server additionally restores
#    events from / persists events to the HF dataset repo and downloads images from it (server/hf_persist.py).
CMD ["sh", "-c", "python scripts/sync_events.py import && cd server && exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --timeout-graceful-shutdown 60"]
