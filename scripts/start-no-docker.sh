#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PID_FILE="$ROOT_DIR/scripts/.local-pids"

mkdir -p "$LOG_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    echo "Please install it and re-run this script." >&2
    case "$1" in
      cargo|rustup)
        echo "- Install Rust: https://rustup.rs/ (then restart your shell)" >&2 ;;
      python3)
        echo "- Install Python 3.10+ (e.g., pyenv, brew, apt)" >&2 ;;
      pip|pip3)
        echo "- Install pip for Python 3 (python3 -m ensurepip --upgrade)" >&2 ;;
      node|npm)
        echo "- Install Node.js LTS (https://nodejs.org/)" >&2 ;;
    esac
    exit 1
  fi
}

# Check prerequisites
require_cmd python3
require_cmd pip3 || require_cmd pip
require_cmd node
require_cmd npm
require_cmd cargo

# Start Python API Gateway (FastAPI)
(
  cd "$ROOT_DIR/backend/services/api_gateway"
  VENV_DIR=".venv_local"
  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
  fi
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt >>"$LOG_DIR/api_gateway.install.log" 2>&1
  echo "[api-gateway] Starting on :8090"
  # REQUIRE_AUTH=false for local demo; adjust as needed
  REQUIRE_AUTH=false python -m uvicorn main:app --host 0.0.0.0 --port 8090 --reload \
    >>"$LOG_DIR/api_gateway.log" 2>&1 &
  echo $! > "$ROOT_DIR/scripts/.api-gateway.pid"
) &
API_GATEWAY_WRAPPER_PID=$!

# Start GraphQL Gateway (Node)
(
  cd "$ROOT_DIR/graphql"
  if [ ! -d node_modules ]; then
    npm install >>"$LOG_DIR/graphql.install.log" 2>&1
  fi
  echo "[graphql] Starting on :4000"
  node gateway.js >>"$LOG_DIR/graphql.log" 2>&1 &
  echo $! > "$ROOT_DIR/scripts/.graphql.pid"
) &
GRAPHQL_WRAPPER_PID=$!

# Start Rust Core API (Axum)
(
  cd "$ROOT_DIR/core"
  echo "[core] Building (this may take a while the first time)..."
  cargo build >>"$LOG_DIR/core.build.log" 2>&1
  echo "[core] Starting on :8080"
  RUST_LOG=info DEEP_API_BASE="http://localhost:8090" cargo run \
    >>"$LOG_DIR/core.log" 2>&1 &
  echo $! > "$ROOT_DIR/scripts/.core.pid"
) &
CORE_WRAPPER_PID=$!

# Start Frontend (Next.js)
(
  cd "$ROOT_DIR/frontend"
  if [ ! -d node_modules ]; then
    npm install >>"$LOG_DIR/frontend.install.log" 2>&1
  fi
  echo "[frontend] Starting on :3000"
  # NEXT_PUBLIC_GRAPHQL_ENDPOINT defaults to localhost:4000/graphql in code if not set
  npm run dev >>"$LOG_DIR/frontend.log" 2>&1 &
  echo $! > "$ROOT_DIR/scripts/.frontend.pid"
) &
FRONTEND_WRAPPER_PID=$!

# Store wrapper PIDs for cleanup (not essential but helpful)
echo "API_GATEWAY_WRAPPER_PID=$API_GATEWAY_WRAPPER_PID" > "$PID_FILE"
echo "GRAPHQL_WRAPPER_PID=$GRAPHQL_WRAPPER_PID" >> "$PID_FILE"
echo "CORE_WRAPPER_PID=$CORE_WRAPPER_PID" >> "$PID_FILE"
echo "FRONTEND_WRAPPER_PID=$FRONTEND_WRAPPER_PID" >> "$PID_FILE"

# Give services time to boot
sleep 3

echo "\nHealth checks:"
# core
if command -v curl >/dev/null 2>&1; then
  curl -s http://localhost:8080/health || true
  echo
  curl -s http://localhost:4000/graphql?query=%7B__typename%7D || true
  echo
  curl -s http://localhost:3000 >/dev/null && echo "Frontend reachable" || echo "Frontend not ready yet"
else
  echo "- Core: http://localhost:8080/health"
  echo "- GraphQL: http://localhost:4000/graphql"
  echo "- Frontend: http://localhost:3000"
fi

echo "\nAll services starting (logs in $LOG_DIR). Press Ctrl+C to exit this script; services continue in background."
