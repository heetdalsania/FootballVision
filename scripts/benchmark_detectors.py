#!/usr/bin/env python3
"""Compare the general-purpose COCO detector with the football-trained one.

The resume claims: "the general-purpose COCO model returned 45 'person'
detections because it counts spectators, while the football-specific model
resolved 23 players, 2 referees and a goalkeeper". That comparison has no
artifact behind it. This script re-runs both detectors on one named frame and
commits the raw counts.

Both models are local .pt files already in the repo. No API key, no account.

Usage:
    python scripts/benchmark_detectors.py --clip clips/highlight_12-37-34.mp4 --frame 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

COCO_WEIGHTS = REPO / "yolo11x.pt"
FOOTBALL_WEIGHTS = REPO / "weights" / "football-player-detection.pt"


def grab_frame(clip: Path, index: int, width: int):
    import cv2

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {clip}")
    frame = None
    for i in range(index + 1):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    if frame is None:
        raise RuntimeError(f"clip has fewer than {index + 1} frames")
    if width and frame.shape[1] != width:
        import cv2 as _cv2
        h = int(frame.shape[0] * width / frame.shape[1])
        frame = _cv2.resize(frame, (width, h))
    return frame


def detect(weights: Path, frame, conf: float, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO

    if not weights.exists():
        return {"weights": str(weights.name), "error": "weights not present"}

    model = YOLO(str(weights)).to(device=device)
    res = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    names = res.names or {}
    counts = Counter()
    for c in (res.boxes.cls.tolist() if res.boxes is not None else []):
        counts[names.get(int(c), str(int(c)))] += 1
    return {
        "weights": weights.name,
        "total_detections": int(sum(counts.values())),
        "by_class": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="clips/highlight_12-37-34.mp4")
    ap.add_argument("--frame", type=int, default=30, help="0-based frame index")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--out-dir", default="benchmarks")
    args = ap.parse_args()

    clip = Path(args.clip) if Path(args.clip).is_absolute() else REPO / args.clip
    if not clip.exists():
        print(f"clip not found: {clip}", file=sys.stderr)
        return 2

    frame = grab_frame(clip, args.frame, args.width)
    print(f"frame {args.frame} of {clip.name}  shape={frame.shape}")

    out = {
        "clip": clip.name,
        "frame_index": args.frame,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "width": args.width,
        "coco": detect(COCO_WEIGHTS, frame, args.conf, args.imgsz, args.device),
        "football": detect(FOOTBALL_WEIGHTS, frame, args.conf, args.imgsz, args.device),
    }

    out_dir = REPO / args.out_dir
    out_dir.mkdir(exist_ok=True)
    (out_dir / "detector_comparison.json").write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== RESULT ===")
    for key in ("coco", "football"):
        d = out[key]
        if "error" in d:
            print(f"  {key:9s} {d['error']}")
            continue
        print(f"  {key:9s} {d['total_detections']:3d} detections  {d['by_class']}")
    print("\n  -> benchmarks/detector_comparison.json")
    print("  The claim being checked: COCO 45 'person' vs football 23 players,")
    print("  2 referees, 1 goalkeeper. Report what this run actually shows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
