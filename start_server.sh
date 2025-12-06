#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
UVICORN_PATTERN=${UVICORN_PATTERN:-"uvicorn tankbot_brain_server:app"}
MAX_PORT_CLEAN_ATTEMPTS=5

format_pids() {
    local pids="${1:-}"
    [[ -z "$pids" ]] && return
    echo "$pids" | tr '\n' ' '
}

terminate_uvicorn_processes() {
    if ! command -v pgrep >/dev/null 2>&1; then
        return
    fi
    local running
    running=$(pgrep -f "$UVICORN_PATTERN" || true)
    if [[ -z "$running" ]]; then
        return
    fi
    echo "[SERVER] Terminating previous tankbot server (PIDs: $(format_pids "$running"))"
    kill $running 2>/dev/null || true
    sleep 1
    local lingering
    lingering=$(pgrep -f "$UVICORN_PATTERN" || true)
    if [[ -n "$lingering" ]]; then
        echo "[SERVER] Force killing lingering server (PIDs: $(format_pids "$lingering"))"
        kill -9 $lingering 2>/dev/null || true
        sleep 1
    fi
}

gather_port_pids() {
    local port="$1"
    local found=""
    if command -v lsof >/dev/null 2>&1; then
        found=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    fi
    if [[ -z "$found" ]] && command -v ss >/dev/null 2>&1; then
        found=$(ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p {print $NF}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p')
    fi
    if [[ -z "$found" ]] && command -v fuser >/dev/null 2>&1; then
        found=$(fuser "$port"/tcp 2>/dev/null | sed 's/.*: //' || true)
    fi
    echo "$found" | tr ' ' '\n' | grep -E '^[0-9]+$' || true
}

ensure_port_free() {
    local port="$1"
    local attempt=1
    while (( attempt <= MAX_PORT_CLEAN_ATTEMPTS )); do
        local busy_pids
        busy_pids=$(gather_port_pids "$port")
        if [[ -z "$busy_pids" ]]; then
            return 0
        fi
        local signal="-15"
        if (( attempt == MAX_PORT_CLEAN_ATTEMPTS )); then
            signal="-9"
        fi
        echo "[SERVER] Port $port busy (attempt $attempt/$MAX_PORT_CLEAN_ATTEMPTS); killing with signal ${signal#-}: $(format_pids "$busy_pids")"
        kill "$signal" $busy_pids 2>/dev/null || true
        sleep 1
        attempt=$((attempt + 1))
    done
    local still_busy
    still_busy=$(gather_port_pids "$port")
    if [[ -n "$still_busy" ]]; then
        echo "[SERVER][WARN] Port $port still busy after cleanup (PIDs: $(format_pids "$still_busy"))" >&2
        return 1
    fi
    return 0
}

terminate_uvicorn_processes
ensure_port_free "$PORT" || true

source setup.sh
uvicorn tankbot_brain_server:app --host "$HOST" --port "$PORT"
