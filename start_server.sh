#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

if command -v pgrep >/dev/null 2>&1; then
    existing=$(pgrep -f "uvicorn tankbot_brain_server:app" || true)
    if [[ -n "${existing:-}" ]]; then
        echo "[SERVER] Terminating previous tankbot server (PIDs: $existing)"
        kill $existing 2>/dev/null || true
        sleep 1
    fi
fi

if command -v lsof >/dev/null 2>&1; then
    port_pids=$(lsof -ti tcp:"$PORT" || true)
    if [[ -n "${port_pids:-}" ]]; then
        echo "[SERVER] Port $PORT busy; terminating holders (PIDs: $port_pids)"
        kill $port_pids 2>/dev/null || true
        sleep 1
    fi
fi

source setup.sh
uvicorn tankbot_brain_server:app --host "$HOST" --port "$PORT"
