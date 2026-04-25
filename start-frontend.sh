#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -f backend/.env ] || [ ! -f frontend/.env ]; then
  python3 backend/scripts/bootstrap.py
fi

cd frontend
npm install
npm run dev
