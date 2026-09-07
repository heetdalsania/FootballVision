#!/usr/bin/env python3
"""Regression benchmark for the bundled golden match clip.

The command exits non-zero when detector quality or latency crosses a checked-in
threshold. It uses only local files and models; no account or API key is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CACHE_DIR = ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))


def evaluate(metrics: dict, thresholds: dict) -> list[str]:
    """Return human-readable threshold failures."""
    failures = []
    for name, rule in thresholds.items():
        if name not in metrics:
            failures.append(f"{name}: metric is missing")
            continue
        value = metrics[name]
        if "min" in rule and value < rule["min"]:
            failures.append(f"{name}: {value} < minimum {rule['min']}")
        if "max" in rule and value > rule["max"]:
            failures.append(f"{name}: {value} > maximum {rule['max']}")
    return failures


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return float(ordered[index])


def benchmark(
    video: Path,
    frames: int,
    stride: int,
    width: int,
    device: str,
    engine=None,
) -> dict:
    import cv2
    from src.fv_engine import FootballEngine
    from src.match_intelligence import MatchIntelligence

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    if engine is None:
        engine = FootballEngine(device=device, player_imgsz=width, team_backend="local")
        engine.load()
    else:
        engine.reset_session()
    intelligence = MatchIntelligence()

    timings, people_counts = [], []
    football = calibrated = projected = detected_people = resolved = outfield = 0
    team_flips = team_transitions = 0
    possession_frames = formation_frames = event_count = 0
    last_teams: dict[int, int] = {}
    read_frame = analysed = 0
    started = time.perf_counter()
    while analysed < frames:
        ok, frame = cap.read()
        if not ok:
            break
        read_frame += 1
        if stride > 1 and read_frame % stride:
            continue
        if width and frame.shape[1] != width:
            height = int(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, height))
        t0 = time.perf_counter()
        result = engine.process(frame, annotate=False)
        timings.append((time.perf_counter() - t0) * 1000)
        analysed += 1
        football += int(result.source_status == "ok")
        calibrated += int(result.calibrated)
        detected = sum(result.counts.get(k, 0) for k in ("player", "goalkeeper", "referee"))
        people_counts.append(detected)
        detected_people += detected
        projected += len(result.players)
        for player in result.players:
            if player.get("role") == "player":
                outfield += 1
                team = player.get("team")
                resolved += int(team in (0, 1))
                track_id = int(player.get("id", -1))
                if team in (0, 1) and track_id >= 0:
                    if track_id in last_teams:
                        team_transitions += 1
                        team_flips += int(last_teams[track_id] != team)
                    last_teams[track_id] = team
        video_time_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        intel = intelligence.update(
            result.players,
            result.ball,
            analysed,
            video_time_s,
            video_time_s,
        )
        possession_frames += int(intel["possession"] is not None)
        formation_frames += int(all(
            f.get("name") != "insufficient data"
            for f in intel["formations"].values()
        ))
        event_count += len(intel["new_events"])
    cap.release()
    elapsed = time.perf_counter() - started
    if not analysed:
        raise RuntimeError("video yielded no analysed frames")

    return {
        "video": str(video.relative_to(ROOT) if video.is_relative_to(ROOT) else video),
        "frames": analysed,
        "stride": stride,
        "width": width,
        "device": engine.device,
        "football_rate": round(football / analysed, 4),
        "calibrated_rate": round(calibrated / analysed, 4),
        "median_people": round(statistics.median(people_counts), 2),
        "projection_rate": round(projected / max(detected_people, 1), 4),
        "resolved_team_rate": round(resolved / max(outfield, 1), 4),
        "team_flip_rate": round(team_flips / max(team_transitions, 1), 4),
        "possession_rate": round(possession_frames / analysed, 4),
        "formation_ready_rate": round(formation_frames / analysed, 4),
        "events_emitted": event_count,
        "mean_latency_ms": round(statistics.mean(timings), 1),
        "p95_latency_ms": round(percentile(timings, .95), 1),
        "throughput_fps": round(analysed / max(elapsed, 1e-9), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "data/sample.mp4")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--thresholds", type=Path,
                        default=ROOT / "benchmarks/golden_sample_thresholds.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/golden_sample_latest.json")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"golden video not found: {args.video}")

    metrics = benchmark(args.video.resolve(), args.frames, args.stride, args.width, args.device)
    thresholds = json.loads(args.thresholds.read_text())
    failures = evaluate(metrics, thresholds)
    result = {"metrics": metrics, "thresholds": thresholds, "passed": not failures,
              "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
