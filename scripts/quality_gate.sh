#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m compileall -q src ui db tests scripts
"$PYTHON_BIN" -m pytest -q
if command -v node >/dev/null 2>&1; then
  node --check ui/static/js/tactical.js
fi
bash -n FootballVision.command scripts/build_macos_app.sh scripts/download_weights.sh scripts/quality_gate.sh
if [[ "${RUN_GOLDEN:-0}" == "1" ]]; then
  "$PYTHON_BIN" scripts/golden_video_benchmark.py
fi
