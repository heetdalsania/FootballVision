#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${ROOT_DIR}/weights"
mkdir -p "${WEIGHTS_DIR}"

download() {
  local filename="$1"
  local file_id="$2"
  local destination="${WEIGHTS_DIR}/${filename}"

  if [[ -s "${destination}" ]]; then
    echo "Already present: ${filename}"
    return
  fi

  echo "Downloading ${filename}..."
  python -m gdown --id "${file_id}" --output "${destination}.part"
  mv "${destination}.part" "${destination}"
}

download "football-ball-detection.pt" "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
download "football-player-detection.pt" "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
download "football-pitch-detection.pt" "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"

echo "FootballVision model weights are ready."
