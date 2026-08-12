#!/usr/bin/env python3
"""Measure the pitch-calibration rate across every clip, and persist it.

The engine already computes this (`match_report.extract` counts `calibrated`
and prints it at the end) and then throws it away, which is why the resume's
"camera calibrating on 100% of frames" has no artifact behind it.

Two traps this script exists to avoid:

1. `calibrated` is STICKY. `FootballEngine._pitch_transformer` keeps the
   previous homography when a solve fails, so `calibrated=True` can mean
   "solved this frame" OR "carried from an earlier frame". A rate that does
   not split those overstates what was measured. We report both, using the
   `homography_solved` flag.

2. "Frames with metrics rows" is NOT the calibration rate. In analysis/sample_*
   that ratio is 96.8%, but the missing frames are the opening ones where every
   player still has team=-1 — team-classifier warm-up, not a failed homography.

Runs offline on the local weights in weights/. No API key, no account.

Usage:
    python scripts/benchmark_calibration.py --frames 60
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def run_clip(clip: Path, frames: int, device: str, width: int, stride: int) -> dict:
    import cv2

    from src.fv_engine import FootballEngine

    engine = FootballEngine(device=device)
    engine.load()

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {clip}")

    per_frame: list[dict] = []
    frame_no = analysed = 0

    while analysed < frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if stride > 1 and frame_no % stride:
            continue
        if width and frame.shape[1] != width:
            h = int(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, h))

        r = engine.process(frame, annotate=False)
        analysed += 1
        per_frame.append({
            "frame_no": frame_no,
            "source_status": r.source_status,
            "calibrated": bool(r.calibrated),
            "homography_solved": bool(r.homography_solved),
            "n_keypoints": int(r.n_keypoints),
            "n_players": len(r.players or []),
        })

    cap.release()
    if not per_frame:
        raise RuntimeError(f"no frames analysed from {clip}")

    ok_frames = [f for f in per_frame if f["source_status"] == "ok"]
    n = len(ok_frames)
    calibrated = sum(f["calibrated"] for f in ok_frames)
    solved = sum(f["homography_solved"] for f in ok_frames)
    players = [f["n_players"] for f in ok_frames if f["n_players"] > 0]

    return {
        "clip": clip.name,
        "frames_analysed": n,
        "frames_calibrated": calibrated,
        # A frame is "carried" when it reports calibrated using a homography
        # solved on an earlier frame. This is by design, but it is a different
        # claim from "the camera calibrated on this frame".
        "frames_solved_fresh": solved,
        "frames_carried": calibrated - solved,
        "calibrated_pct": round(100.0 * calibrated / n, 1) if n else 0.0,
        "solved_fresh_pct": round(100.0 * solved / n, 1) if n else 0.0,
        "median_players_per_frame": statistics.median(players) if players else 0,
        "min_players": min(players) if players else 0,
        "max_players": max(players) if players else 0,
        "per_frame": per_frame,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", default="clips")
    ap.add_argument("--clip", default=None, help="single clip instead of the directory")
    ap.add_argument("--frames", type=int, default=60, help="analysed frames per clip")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--out-dir", default="benchmarks")
    args = ap.parse_args()

    if args.clip:
        clips = [Path(args.clip) if Path(args.clip).is_absolute() else REPO / args.clip]
    else:
        clips = sorted((REPO / args.clips_dir).glob("*.mp4"))
    if not clips:
        print("no clips found", file=sys.stderr)
        return 2

    out_dir = REPO / args.out_dir
    out_dir.mkdir(exist_ok=True)

    results = []
    for clip in clips:
        print(f"\n=== {clip.name} ===", flush=True)
        t0 = time.time()
        try:
            res = run_clip(clip, args.frames, args.device, args.width, args.stride)
        except Exception as exc:                     # keep going; report at the end
            print(f"  FAILED: {exc}", file=sys.stderr)
            results.append({"clip": clip.name, "error": str(exc)})
            continue
        results.append(res)
        print(f"  {res['frames_analysed']} frames in {time.time()-t0:.0f}s | "
              f"calibrated {res['frames_calibrated']}/{res['frames_analysed']} "
              f"({res['calibrated_pct']}%) | fresh solves {res['frames_solved_fresh']} "
              f"| median players {res['median_players_per_frame']}")

    good = [r for r in results if "error" not in r]
    total = sum(r["frames_analysed"] for r in good)
    cal = sum(r["frames_calibrated"] for r in good)
    fresh = sum(r["frames_solved_fresh"] for r in good)
    pooled = {
        "clips": len(good),
        "frames_analysed": total,
        "frames_calibrated": cal,
        "frames_solved_fresh": fresh,
        "calibrated_pct": round(100.0 * cal / total, 1) if total else 0.0,
        "solved_fresh_pct": round(100.0 * fresh / total, 1) if total else 0.0,
        "worst_clip_pct": min((r["calibrated_pct"] for r in good), default=0.0),
    }

    (out_dir / "calibration.json").write_text(
        json.dumps({"pooled": pooled, "per_clip": results}, indent=2) + "\n")

    print("\n=== POOLED ===")
    print(f"  {pooled['clips']} clips | {total} frames")
    print(f"  calibrated      : {cal}/{total} ({pooled['calibrated_pct']}%)")
    print(f"  fresh solves    : {fresh}/{total} ({pooled['solved_fresh_pct']}%)")
    print(f"  worst clip      : {pooled['worst_clip_pct']}%")
    print("\n  'Calibrated' includes frames carried on an earlier homography.")
    print("  Claim 'camera calibrating on 100% of frames' needs the FRESH number;")
    print("  the calibrated number supports 'a valid homography on 100% of frames'.")
    print(f"\n  -> benchmarks/calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
