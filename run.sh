#!/usr/bin/env bash
# H3 Studio launcher. ComfyUI must already be running (python main.py --reserve-vram 6).
cd "$(dirname "$0")/h3studio"
PY="${H3_PYTHON:-/home/hasitha/comfyui-venv/bin/python}"
PORT="${H3_PORT:-7860}"
if ! curl -s -m 2 -o /dev/null "http://127.0.0.1:${PORT:-7860}/api/status"; then :; else
  echo "H3 Studio is already running -> http://127.0.0.1:${PORT}"
  echo "(to restart: pkill -f 'h3studio/server.py|python server.py'  then ./run.sh)"; exit 0
fi
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "Port ${PORT} is used by another program. Run:  H3_PORT=7861 ./run.sh"; exit 1
fi
if ! curl -s -m 2 -o /dev/null http://127.0.0.1:8188/system_stats; then
  echo "Note: ComfyUI is not reachable on :8188 - the UI will start, but start ComfyUI before generating."
fi
exec "$PY" server.py
