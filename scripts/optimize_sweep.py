#!/usr/bin/env python3
"""Sweep engine configurations for the best latency that keeps detection quality.

benchmark_latency.py answers "did the caching work". This answers "what should
the defaults be". After the caching, `team` and `pitch` are zero and the cost
is detection: one player pass, plus a SECOND full pass by the ball model on
every frame the primary detector misses the ball, which is most of them.

Every config is scored on speed AND on what it detects, because a faster
setting that stops seeing players is not an optimisation. A candidate is only
acceptable if it holds player recall against the baseline.

Runs offline on the local weights in weights/. No API key, no account.

Usage:
    python scripts/optimize_sweep.py --clip data/sample.mp4 --frames 40
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

# name -> engine kwargs. All start from the post-caching defaults.
BASE = dict(calibrate_every=15, team_refresh_every=90)

CANDIDATES = {
    "baseline":            dict(),
    "ball_every_5":        dict(ball_every=5),
    "ball_every_15":       dict(ball_every=15),
    "ball_imgsz_640":      dict(ball_imgsz=640),
    "ball_every_5+640":    dict(ball_every=5, ball_imgsz=640),
    "player_imgsz_960":    dict(player_imgsz=960),
    "player_960+ball_5":   dict(player_imgsz=960, ball_every=5, ball_imgsz=640),
    "player_imgsz_640":    dict(player_imgsz=640),
}


def run(cfg: dict, clip: Path, frames: int, device: str, width: int) -> dict:
    import cv2

    from src.fv_engine import FootballEngine

    kwargs = dict(BASE)
    kwargs.update(cfg)
    engine = FootballEngine(device=device, **kwargs)
    engine.load()

    cap = cv2.VideoCapture(str(clip))
    per_frame, stage_ms, players, calibrated, n = [], {}, [], 0, 0

    while n < frames:
        ok, frame = cap.read()
        if not ok:
            break
        if width and frame.shape[1] != width:
            h = int(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, h))

        t0 = time.perf_counter()
        r = engine.process(frame, annotate=False)
        per_frame.append((time.perf_counter() - t0) * 1000.0)
        n += 1
        if r.source_status != "ok":
            continue
        calibrated += int(bool(r.calibrated))
        players.append(len(r.players or []))
        for k, v in (r.timings_ms or {}).items():
            stage_ms.setdefault(k, []).append(v)

    cap.release()
    if not per_frame:
        raise RuntimeError("no frames analysed")

    seen = [p for p in players if p > 0]
    return {
        "config": kwargs,
        "frames": n,
        "median_ms": round(statistics.median(per_frame), 1),
        "mean_ms": round(statistics.fmean(per_frame), 1),
        "median_players": statistics.median(seen) if seen else 0,
        "mean_players": round(statistics.fmean(seen), 1) if seen else 0.0,
        "frames_with_players": len(seen),
        "calibrated_pct": round(100.0 * calibrated / max(n, 1), 1),
        "median_stage_ms": {k: round(statistics.median(v), 1)
                            for k, v in sorted(stage_ms.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="data/sample.mp4")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--only", default=None, help="comma-separated candidate names")
    ap.add_argument("--out-dir", default="benchmarks")
    args = ap.parse_args()

    clip = Path(args.clip) if Path(args.clip).is_absolute() else REPO / args.clip
    if not clip.exists():
        print(f"clip not found: {clip}", file=sys.stderr)
        return 2

    names = ([s.strip() for s in args.only.split(",")] if args.only
             else list(CANDIDATES))

    results = {}
    for name in names:
        print(f"\n=== {name}  {CANDIDATES[name]} ===", flush=True)
        try:
            results[name] = run(CANDIDATES[name], clip, args.frames,
                                args.device, args.width)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        r = results[name]
        print(f"  median {r['median_ms']:7.1f} ms | players {r['median_players']} "
              f"| calibrated {r['calibrated_pct']}% | {r['median_stage_ms']}")

    base = results.get("baseline")
    print("\n" + "=" * 92)
    print(f"{'config':22s} {'median ms':>10s} {'vs base':>9s} "
          f"{'players':>8s} {'recall':>8s} {'calib%':>7s}   verdict")
    print("-" * 92)
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["median_ms"]):
        speed = f"{base['median_ms']/r['median_ms']:.2f}x" if base else "-"
        # Detection quality guard: a config that finds materially fewer players
        # is not faster, it is broken.
        if base and base["mean_players"]:
            recall = r["mean_players"] / base["mean_players"]
            verdict = "OK" if recall >= 0.95 else (
                "MARGINAL" if recall >= 0.90 else "REJECT - loses players")
        else:
            recall, verdict = 1.0, "?"
        print(f"{name:22s} {r['median_ms']:10.1f} {speed:>9s} "
              f"{r['median_players']:8} {recall:7.2f} {r['calibrated_pct']:6.1f}%   {verdict}")

    out_dir = REPO / args.out_dir
    out_dir.mkdir(exist_ok=True)
    (out_dir / "optimize_sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n  -> benchmarks/optimize_sweep.json")
    print("  Pick the fastest config whose recall stays at or above 0.95.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
