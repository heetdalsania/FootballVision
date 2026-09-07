#!/usr/bin/env python3
"""Run the local golden metrics over a folder of varied football clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.golden_video_benchmark import benchmark, evaluate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT / "data" / "reliability")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--thresholds", type=Path,
        default=ROOT / "benchmarks" / "golden_sample_thresholds.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "benchmarks" / "video_suite_latest.json",
    )
    args = parser.parse_args()
    videos = sorted(
        path for path in args.directory.glob("*")
        if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    )
    if not videos:
        parser.error(
            f"no videos found in {args.directory}; add clips with varied lighting, "
            "camera motion, resolution, and occlusion"
        )

    from src.fv_engine import FootballEngine
    engine = FootballEngine(
        device=args.device, player_imgsz=args.width, team_backend="local"
    )
    engine.load()
    thresholds = json.loads(args.thresholds.read_text())
    results = []
    for video in videos:
        metrics = benchmark(
            video, args.frames, args.stride, args.width, args.device, engine=engine
        )
        failures = evaluate(metrics, thresholds)
        results.append({
            "video": str(video.relative_to(ROOT) if video.is_relative_to(ROOT) else video),
            "passed": not failures,
            "failures": failures,
            "metrics": metrics,
        })
        print(f"{video.name}: {'PASS' if not failures else 'FAIL'}")

    output = {
        "passed": all(result["passed"] for result in results),
        "video_count": len(results),
        "thresholds": thresholds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
