#!/usr/bin/env python3
"""
Build the static post-match analyst report.

The expensive pass (running the vision engine over the video) is separate
from the cheap one (drawing). Extract once, then re-render the figure as
often as you like from the saved CSVs:

    .venv/bin/python scripts/analyst_report.py data/sample.mp4
    .venv/bin/python scripts/analyst_report.py --from-csv analysis/sample
    .venv/bin/python scripts/analyst_report.py data/sample.mp4 --stride 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.match_report import extract, render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Static post-match analyst report")
    ap.add_argument("video", nargs="?", help="input match video")
    ap.add_argument("--from-csv", default=None, metavar="PREFIX",
                    help="skip extraction; re-render from PREFIX_tracking.csv "
                         "and PREFIX_metrics.csv")
    ap.add_argument("--outdir", default="analysis", help="output directory")
    ap.add_argument("--name", default=None, help="basename for outputs")
    ap.add_argument("--stride", type=int, default=3,
                    help="analyse every Nth frame (default 3)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N analysed frames (0 = all)")
    ap.add_argument("--device", default="mps", help="mps | cuda | cpu")
    ap.add_argument("--title", default=None, help="report title")
    args = ap.parse_args()

    if not args.video and not args.from_csv:
        ap.error("give a video, or --from-csv PREFIX")

    import pandas as pd

    if args.from_csv:
        prefix = args.from_csv
        tracking = pd.read_csv(f"{prefix}_tracking.csv")
        metrics = pd.read_csv(f"{prefix}_metrics.csv")
        name = args.name or os.path.basename(prefix)
        print(f"loaded   : {len(tracking)} tracking rows, {len(metrics)} metric rows")
    else:
        name = args.name or os.path.splitext(os.path.basename(args.video))[0]
        tracking, metrics = extract(
            args.video, stride=args.stride, max_frames=args.max_frames,
            device=args.device,
        )
        os.makedirs(args.outdir, exist_ok=True)
        prefix = os.path.join(args.outdir, name)
        tracking.to_csv(f"{prefix}_tracking.csv", index=False)
        metrics.to_csv(f"{prefix}_metrics.csv", index=False)
        print(f"saved    : {prefix}_tracking.csv")
        print(f"saved    : {prefix}_metrics.csv")

    if tracking.empty:
        sys.exit("no tracking data — was the pitch visible in this video?")

    frames = tracking["frame_id"].nunique()
    span = tracking["time_s"].max() - tracking["time_s"].min()
    subtitle = (f"{name}  ·  {frames} analysed frames over {span:.0f} s  ·  "
                f"{len(tracking):,} player positions  ·  "
                f"positions in metres on a 105 x 68 m pitch")

    out_png = os.path.join(args.outdir, f"{name}_report.png")
    os.makedirs(args.outdir, exist_ok=True)
    path = render(
        tracking, metrics, out_png,
        title=args.title or "Post-match tactical report",
        subtitle=subtitle,
    )
    print(f"report   : {path}")


if __name__ == "__main__":
    main()
