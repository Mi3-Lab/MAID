FROM python:3.12-slim
WORKDIR /srv/wheelview
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server ./server
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY annotations ./annotations
# images/ and var/ are volumes: mount a populated images dir (scripts/fetch_images.py) and a persistent var/ for SQLite.
ENV WV_DB=/srv/wheelview/var/wheelview.db WV_IMAGE_DIR=/srv/wheelview/images PORT=8000
VOLUME ["/srv/wheelview/var", "/srv/wheelview/images"]
EXPOSE 8000
# Import any committed annotations on start (idempotent), then serve.
CMD python scripts/sync_events.py import && cd server && exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers
