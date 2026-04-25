#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -f backend/.env ] || [ ! -f frontend/.env ]; then
  python3 backend/scripts/bootstrap.py
fi

if [ ! -d backend/.venv ]; then
  python3 -m venv backend/.venv
fi
source backend/.venv/bin/activate
pip install -r backend/requirements.txt

cd frontend
npm install
cd "$ROOT_DIR"

cleanup() {
  if [ -n "${BACK_PID:-}" ] && ps -p "$BACK_PID" >/dev/null 2>&1; then
    kill "$BACK_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

(
  cd backend
  source .venv/bin/activate
  uvicorn app.main:app --reload --port 8000
) &
BACK_PID=$!

echo "Backend started on http://localhost:8000 (pid: $BACK_PID)"
echo "Starting frontend on http://localhost:5173"

cd frontend
npm run dev
