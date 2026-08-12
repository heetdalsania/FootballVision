#!/usr/bin/env bash
# Regenerate every benchmark artifact backing a resume claim.
#
# A claim is verified when a command reproduces it. This is that command.
#
# Runs entirely offline on the local weights in weights/. No API key, no
# account, no paid service. If any of this ever needs a signup, it is wrong.
#
#   ./scripts/run_benchmarks.sh [clip] [frames]
set -euo pipefail

cd "$(dirname "$0")/.."

CLIP="${1:-clips/highlight_12-37-34.mp4}"
FRAMES="${2:-60}"
PY="${PYTHON:-python3}"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

echo "=============================================="
echo " FootballVision benchmarks"
echo " clip   : $CLIP"
echo " frames : $FRAMES per arm"
echo " python : $PY"
echo "=============================================="

echo
echo ">>> 1/3  latency (naive vs optimized)"
"$PY" scripts/benchmark_latency.py --clip "$CLIP" --frames "$FRAMES"

echo
echo ">>> 2/3  calibration rate across every clip"
"$PY" scripts/benchmark_calibration.py --frames "$FRAMES"

echo
echo ">>> 3/3  COCO vs football-trained detector"
"$PY" scripts/benchmark_detectors.py --clip "$CLIP"

echo
echo "=============================================="
echo " Artifacts written to benchmarks/:"
ls -1 benchmarks/*.json 2>/dev/null | sed 's/^/   /'
echo
echo " If any number differs from what the resume claims,"
echo " the resume changes to match the measurement."
echo "=============================================="
