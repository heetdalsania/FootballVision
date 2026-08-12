#!/usr/bin/env python3
"""Measure per-frame latency with and without the two caching optimisations.

The résumé claims "cut per-frame latency 4.5x, from 1,133 ms to 249 ms, by
caching per-track classifications that cannot change between frames and
re-solving the homography on an interval". The 1,133 ms baseline is recorded
(BRAIN, 2026-07-24, 60-frame run); the 249 ms figure is recorded nowhere. This
script produces both numbers from the same binary on the same clip.

Both optimisations are constructor parameters on FootballEngine, so the naive
arm needs no old code:

    naive      calibrate_every=1   team_refresh_every=1
    optimized  calibrate_every=15  team_refresh_every=90   (the defaults)

Runs entirely offline on the local weights in weights/. No API key, no account.

Usage:
    python scripts/benchmark_latency.py --clip clips/highlight_12-37-34.mp4 --frames 60
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ARMS = {
    "naive": dict(calibrate_every=1, team_refresh_every=1),
    "optimized": dict(calibrate_every=15, team_refresh_every=90),
}


def hardware() -> str:
    """Best-effort machine description, so a number can be compared later."""
    bits = [platform.machine(), platform.system(), platform.release()]
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if chip:
            bits.insert(0, chip)
    except Exception:
        pass
    return " | ".join(b for b in bits if b)


def run_arm(arm: str, clip: str, frames: int, device: str, width: int) -> dict:
    import cv2

    from src.fv_engine import FootballEngine

    cfg = ARMS[arm]
    engine = FootballEngine(device=device, **cfg)

    t_load = time.time()
    engine.load()
    load_s = time.time() - t_load

    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {clip}")

    per_frame_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {}
    analysed = 0

    while analysed < frames:
        ok, frame = cap.read()
        if not ok:
            break
        if width and frame.shape[1] != width:
            h = int(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, h))

        t0 = time.perf_counter()
        result = engine.process(frame, annotate=False)
        per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
        analysed += 1

        for stage, ms in (result.timings_ms or {}).items():
            stage_ms.setdefault(stage, []).append(ms)

        if analysed % 10 == 0:
            print(f"  {arm}: {analysed}/{frames} frames "
                  f"({per_frame_ms[-1]:.0f} ms last)", flush=True)

    cap.release()
    if not per_frame_ms:
        raise RuntimeError("no frames analysed")

    ordered = sorted(per_frame_ms)
    return {
        "arm": arm,
        "clip": clip,
        "frames": analysed,
        "calibrate_every": cfg["calibrate_every"],
        "team_refresh_every": cfg["team_refresh_every"],
        "device": device,
        "width": width,
        "hardware": hardware(),
        "model_load_s": round(load_s, 1),
        "per_frame_ms": [round(v, 1) for v in per_frame_ms],
        # Median, not mean: the first frames include team-classifier fitting
        # (TEAM_FIT_CROPS crops must accumulate first) and model warm-up, which
        # inflate a mean and are not what the claim is about.
        "median_ms": round(statistics.median(per_frame_ms), 1),
        "mean_ms": round(statistics.fmean(per_frame_ms), 1),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
        "median_stage_ms": {
            k: round(statistics.median(v), 1) for k, v in sorted(stage_ms.items())
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="clips/highlight_12-37-34.mp4")
    ap.add_argument("--frames", type=int, default=60,
                    help="frames per arm (60 matches the recorded baseline run)")
    ap.add_argument("--device", default="mps", help="mps | cuda | cpu")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--arms", default="naive,optimized")
    ap.add_argument("--out-dir", default="benchmarks")
    args = ap.parse_args()

    clip = args.clip if os.path.isabs(args.clip) else str(REPO / args.clip)
    if not os.path.exists(clip):
        print(f"clip not found: {clip}", file=sys.stderr)
        return 2

    out_dir = REPO / args.out_dir
    out_dir.mkdir(exist_ok=True)

    results = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            print(f"unknown arm {arm!r}", file=sys.stderr)
            return 2
        print(f"\n=== arm: {arm}  ({ARMS[arm]}) ===", flush=True)
        res = run_arm(arm, clip, args.frames, args.device, args.width)
        results[arm] = res
        path = out_dir / f"latency_{arm}.json"
        path.write_text(json.dumps(res, indent=2) + "\n")
        print(f"  -> {path.relative_to(REPO)}  median {res['median_ms']} ms")

    print("\n=== RESULT ===")
    for arm, r in results.items():
        print(f"  {arm:10s} median {r['median_ms']:8.1f} ms   "
              f"mean {r['mean_ms']:8.1f}   p95 {r['p95_ms']:8.1f}   "
              f"stages {r['median_stage_ms']}")

    if "naive" in results and "optimized" in results:
        a, b = results["naive"]["median_ms"], results["optimized"]["median_ms"]
        ratio = a / b if b else float("inf")
        print(f"\n  {a:.0f} ms -> {b:.0f} ms  =  {ratio:.2f}x reduction")
        print("\n  The ratio is the durable claim; absolute ms are hardware-bound.")
        print("  If this differs from the resume, the resume changes to match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
