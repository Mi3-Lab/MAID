#!/usr/bin/env bash
# One-command local run:  ./run.sh [port]      (default 8770)
# Installs deps if missing, links/copies images if a local WheelView build is found, imports committed events, starts the server.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-${PORT:-8770}}"
python3 -c "import fastapi, uvicorn, httpx, PIL" 2>/dev/null || python3 -m pip install -r requirements.txt
if [ -n "${WHEELVIEW_BUILD_DIR:-}" ] || [ -d /workspace/wheelview/thumbs ]; then
  python3 scripts/fetch_images.py --from-dir "${WHEELVIEW_BUILD_DIR:-/workspace/wheelview}" --link
fi
python3 scripts/sync_events.py import
cd server && exec python3 -m uvicorn main:app --host "${HOST:-127.0.0.1}" --port "$PORT"
