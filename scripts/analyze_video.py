#!/usr/bin/env python3
"""
Run FootballVision over a video file and write an annotated result.

Screen capture can only see what macOS is actually rendering: a browser that
is fullscreen on a Space you have switched away from is not on any display,
so nothing can be captured from it. For testing, benchmarking, and for
producing a reproducible demo, a video file is strictly better — it always
contains the footage, it runs faster than real time, and the same input gives
the same output every run.

Usage:
    .venv/bin/python scripts/analyze_video.py match.mp4
    .venv/bin/python scripts/analyze_video.py match.mp4 --out annotated.mp4
    .venv/bin/python scripts/analyze_video.py match.mp4 --max-frames 300
"""

import argparse
import os
import statistics
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fv_engine import FootballEngine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse a football video")
    ap.add_argument("video", help="path to the input video")
    ap.add_argument("--out", default=None, help="annotated output video path")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = whole video)")
    ap.add_argument("--stride", type=int, default=1,
                    help="analyse every Nth frame")
    ap.add_argument("--device", default="mps", help="mps | cuda | cpu")
    ap.add_argument("--width", type=int, default=1920,
                    help="resize frames to this width before analysis")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"No such file: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"input   : {args.video}  ({total} frames @ {src_fps:.1f} fps)")

    engine = FootballEngine(device=args.device)
    engine.load()

    writer = None
    counts, calib, times = [], [], []
    frame_no = analysed = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if args.stride > 1 and frame_no % args.stride:
            continue

        if args.width and frame.shape[1] != args.width:
            h = int(frame.shape[0] * args.width / frame.shape[1])
            frame = cv2.resize(frame, (args.width, h))

        result = engine.process(frame, annotate=bool(args.out))
        analysed += 1

        if result.source_status != "ok":
            if analysed % 25 == 1:
                print(f"  frame {frame_no}: {result.source_status}")
        else:
            counts.append(result.counts.get("player", 0))
            calib.append(result.calibrated)
            times.append(sum(result.timings_ms.values()))

        if args.out and result.annotated is not None:
            if writer is None:
                h, w = result.annotated.shape[:2]
                writer = cv2.VideoWriter(
                    args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                    src_fps / max(1, args.stride), (w, h))
            writer.write(result.annotated)

        if analysed % 25 == 0:
            recent = counts[-25:] or [0]
            print(f"  {analysed:5d} frames | players≈{statistics.median(recent):.0f} "
                  f"| calibrated {sum(calib[-25:])}/{len(calib[-25:])} "
                  f"| {statistics.median(times[-25:] or [0]):.0f} ms/frame")

        if args.max_frames and analysed >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t_start
    print("\n" + "=" * 58)
    print(f"analysed          : {analysed} frames in {elapsed:.1f}s "
          f"({analysed / max(elapsed, 1e-6):.1f} fps)")
    if counts:
        print(f"players/frame     : median {statistics.median(counts):.0f}  "
              f"min {min(counts)}  max {max(counts)}")
        print(f"calibrated frames : {sum(calib)}/{len(calib)} "
              f"({100 * sum(calib) / len(calib):.0f}%)")
        print(f"latency           : median {statistics.median(times):.0f} ms/frame")
    else:
        print("no football detected in any frame — is this a match video?")
    if args.out:
        print(f"annotated output  : {args.out}")
    print("=" * 58)


if __name__ == "__main__":
    main()
