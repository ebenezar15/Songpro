#!/usr/bin/env bash
# One-command setup + launch: creates a venv, installs pinned deps, builds the UI, starts the local server.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
. .venv/bin/activate
pip install -q -r backend/requirements.txt
if [ ! -d frontend/dist ] || [ "${REBUILD_UI:-0}" = "1" ]; then
  (cd frontend && npm ci --silent && npm run build --silent)
fi
echo "Kuralkoodu on http://127.0.0.1:${KURALKOODU_PORT:-8765}  (local only)"
cd backend && exec python -m kuralkoodu.server
