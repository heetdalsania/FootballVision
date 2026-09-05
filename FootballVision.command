#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  echo "FootballVision is not set up yet."
  echo "Run the Setup commands in README.md first."
  read -r -p "Press Return to close…" _
  exit 1
fi

for model in football-ball-detection.pt football-player-detection.pt football-pitch-detection.pt; do
  if [[ ! -f "$APP_DIR/weights/$model" ]]; then
    echo "Missing model weights. Run: ./scripts/download_weights.sh"
    read -r -p "Press Return to close…" _
    exit 1
  fi
done

if curl -fsS http://localhost:8000/tactical >/dev/null 2>&1; then
  open http://localhost:8000/tactical
  exit 0
fi

"$APP_DIR/.venv/bin/python" main.py &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM

for _ in {1..40}; do
  if curl -fsS http://localhost:8000/tactical >/dev/null 2>&1; then
    if command -v open >/dev/null 2>&1; then
      open http://localhost:8000/tactical
    fi
    break
  fi
  sleep 0.25
done

wait "$SERVER_PID"
