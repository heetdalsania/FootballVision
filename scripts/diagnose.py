#!/usr/bin/env python3
"""Print actionable local FootballVision setup diagnostics."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    weights = {}
    for name in (
        "football-ball-detection.pt",
        "football-player-detection.pt",
        "football-pitch-detection.pt",
    ):
        path = ROOT / "weights" / name
        weights[name] = path.is_file() and path.stat().st_size > 1_000_000
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("cv2", "fastapi", "numpy", "torch", "ultralytics", "supervision")
    }
    folders = {}
    for name in ("uploads", "analysis", "clips"):
        path = ROOT / name
        try:
            path.mkdir(exist_ok=True)
            probe = path / ".diagnostic-write-test"
            probe.touch()
            probe.unlink()
            folders[name] = True
        except OSError:
            folders[name] = False
    report = {
        "ready": all(weights.values()) and all(packages.values()) and all(folders.values()),
        "python": sys.version.split()[0],
        "supported_python": sys.version_info[:2] in ((3, 11), (3, 12)),
        "weights": weights,
        "packages": packages,
        "folders_writable": folders,
        "ffmpeg_optional": bool(shutil.which("ffmpeg")),
        "free_disk_gb": round(shutil.disk_usage(ROOT).free / 1e9, 1),
        "signup_required": False,
    }
    print(json.dumps(report, indent=2))
    if not all(weights.values()):
        print("\nFix missing weights with: ./scripts/download_weights.sh")
    if not all(packages.values()):
        print("Fix missing packages with: .venv/bin/python -m pip install -r requirements.txt")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    project_python = ROOT / ".venv" / "bin" / "python"
    if project_python.is_file() and os.environ.get("FOOTBALLVISION_DIAGNOSE_VENV") != "1":
        environment = os.environ.copy()
        environment["FOOTBALLVISION_DIAGNOSE_VENV"] = "1"
        os.execve(str(project_python), [str(project_python), __file__], environment)
    raise SystemExit(main())
