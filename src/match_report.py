"""
Static post-match analyst report.

The live `/tactical` canvas answers "what is happening now". This module
answers "what happened across the match" — the report an analyst puts in
front of a coach afterwards, rendered once to a PNG rather than streamed.

Two stages, deliberately separate:

    extract(video)  ->  tracking DataFrame + metrics DataFrame  (slow, GPU)
    render(frames)  ->  a figure                                (fast, cheap)

Keeping them apart means the expensive pass runs once and the plots can be
re-cut as many times as you like from the saved CSVs, which is how you
actually iterate on a figure.

Pitch plots come from `mplsoccer`, the standard football pitch-drawing
library — the markings, aspect ratio and coordinate handling are a solved
problem and not worth re-deriving.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Team colours. Index by team id (0 / 1) as the engine reports it.
TEAM_COLORS = {0: "#D81E3F", 1: "#1F4E9C"}
TEAM_NAMES = {0: "Team A", 1: "Team B"}

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


# --------------------------------------------------------------------------
# Stage 1 — extraction
# --------------------------------------------------------------------------

def extract(
    video_path: str,
    stride: int = 3,
    max_frames: int = 0,
    device: str = "mps",
    width: int = 1920,
    progress_every: int = 25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the engine over a video and collect per-frame tracking + metrics.

    Args:
        video_path: input match video.
        stride: analyse every Nth frame. 3 is plenty for positional summaries
            and keeps a 30 s clip to about a minute of compute.
        max_frames: stop after N analysed frames (0 = whole video).
        device: mps | cuda | cpu.
        width: resize frames to this width before analysis.

    Returns:
        (tracking, metrics) DataFrames. `tracking` is one row per player per
        analysed frame; `metrics` is one row per team per analysed frame.
    """
    import cv2

    from src.fv_engine import FootballEngine
    from src.tactical_metrics import compute_metrics

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    logger.info("input: %s (%d frames @ %.1f fps)", video_path, total, src_fps)
    print(f"input    : {video_path}  ({total} frames @ {src_fps:.1f} fps)")

    engine = FootballEngine(device=device)
    engine.load()

    track_rows: List[Dict] = []
    metric_rows: List[Dict] = []
    frame_no = analysed = calibrated = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if stride > 1 and frame_no % stride:
            continue

        if width and frame.shape[1] != width:
            h = int(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, h))

        result = engine.process(frame, annotate=False)
        analysed += 1
        if result.source_status != "ok":
            continue
        calibrated += int(bool(result.calibrated))

        t_s = frame_no / max(src_fps, 1e-6)
        players = result.players or []

        for p in players:
            track_rows.append({
                "frame_id": result.frame_id,
                "time_s": round(t_s, 3),
                "track_id": p.get("id"),
                "team": p.get("team", -1),
                "role": p.get("role", "player"),
                "x": p.get("x"),
                "y": p.get("y"),
            })

        # Metrics are defined on outfield players only: goalkeepers sit far
        # behind the line and would drag every shape statistic backwards.
        outfield = [p for p in players if p.get("role") == "player"]
        if outfield:
            concepts = compute_metrics(outfield, PITCH_LENGTH, PITCH_WIDTH)
            control = concepts.get("control") or {}
            for team_key, stats in (concepts.get("teams") or {}).items():
                # A team with too few players on the frame gets no stats block.
                if not stats:
                    continue
                metric_rows.append({
                    "frame_id": result.frame_id,
                    "time_s": round(t_s, 3),
                    "team": int(team_key),
                    "n": stats.get("n"),
                    "compactness_m": stats.get("compactness_m"),
                    "width_m": stats.get("width_m"),
                    "depth_m": stats.get("depth_m"),
                    "line_height_m": stats.get("line_height_m"),
                    "control_pct": control.get(str(team_key)),
                    "pressing_m": concepts.get("pressing_m"),
                })

        if progress_every and analysed % progress_every == 0:
            rate = analysed / max(time.time() - t_start, 1e-6)
            print(f"  {analysed:5d} frames | {len(track_rows):6d} detections "
                  f"| {rate:.1f} fps")

        if max_frames and analysed >= max_frames:
            break

    cap.release()
    elapsed = time.time() - t_start

    tracking = pd.DataFrame(track_rows)
    metrics = pd.DataFrame(metric_rows)

    print(f"\nanalysed : {analysed} frames in {elapsed:.1f}s "
          f"({analysed / max(elapsed, 1e-6):.1f} fps)")
    print(f"calibrated: {calibrated}/{analysed} "
          f"({100 * calibrated / max(analysed, 1):.0f}%)")
    print(f"tracking : {len(tracking)} rows | metrics: {len(metrics)} rows")

    return tracking, metrics


# --------------------------------------------------------------------------
# Stage 2 — rendering
# --------------------------------------------------------------------------

def _pitch(**kwargs):
    """A 105x68 m pitch in the report's visual style."""
    from mplsoccer import Pitch

    opts = dict(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH,
        pitch_width=PITCH_WIDTH,
        pitch_color="#FBFBF9",
        line_color="#BCBCB6",
        linewidth=0.9,
        goal_type="box",
    )
    opts.update(kwargs)
    return Pitch(**opts)


def _teams_present(tracking: pd.DataFrame) -> List[int]:
    outfield = tracking[tracking["role"] == "player"]
    return sorted(t for t in outfield["team"].unique() if t in (0, 1))


def _heatmap_panel(ax, tracking: pd.DataFrame, team: int) -> None:
    """Occupancy heatmap for one team, smoothed onto the pitch."""
    from scipy.ndimage import gaussian_filter

    pitch = _pitch()
    pitch.draw(ax=ax)

    sub = tracking[(tracking["team"] == team) & (tracking["role"] == "player")]
    sub = sub.dropna(subset=["x", "y"])
    if sub.empty:
        ax.set_title(f"{TEAM_NAMES[team]} — no data", fontsize=9)
        return

    stat = pitch.bin_statistic(
        sub["x"].to_numpy(), sub["y"].to_numpy(),
        statistic="count", bins=(42, 27),
    )
    stat["statistic"] = gaussian_filter(stat["statistic"], 1.4)

    base = TEAM_COLORS[team]
    cmap = _fade_cmap(base)
    # No `alpha=` here: it would override the colormap's own alpha ramp and
    # paint every cell solid, including the empty ones.
    pitch.heatmap(stat, ax=ax, cmap=cmap, edgecolors="none", zorder=0)
    pitch.draw(ax=ax)  # redraw markings over the heatmap

    ax.set_title(f"{TEAM_NAMES[team]} — occupancy   ·   {len(sub):,} positions",
                 fontsize=9.5, color="#222", pad=6)


def _fade_cmap(hex_color: str):
    """A transparent-to-colour ramp, so an empty pitch stays white."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    r, g, b = to_rgb(hex_color)
    return LinearSegmentedColormap.from_list(
        "fade", [(r, g, b, 0.0), (r, g, b, 0.45), (r, g, b, 1.0)]
    )


def _formation_panel(ax, tracking: pd.DataFrame, teams: List[int]) -> None:
    """Average position per tracked player, with each team's shape outlined."""
    pitch = _pitch()
    pitch.draw(ax=ax)

    outfield = tracking[tracking["role"] == "player"].dropna(subset=["x", "y"])
    for team in teams:
        sub = outfield[outfield["team"] == team]
        if sub.empty:
            continue
        # ByteTrack issues a fresh id whenever it loses and re-acquires a
        # player, so a clip yields more track ids than there are players on
        # the pitch. Take the most-tracked 11 per team: readable, and honest
        # about what it is. This is NOT a claim that these are the starting
        # eleven — a real formation plot needs identities stable across the
        # whole match, which single-clip tracking does not give you.
        seen = sub.groupby("track_id").size().sort_values(ascending=False)
        keep = seen.head(11).index
        avg = sub[sub["track_id"].isin(keep)].groupby("track_id")[["x", "y"]].mean()
        if avg.empty:
            continue

        color = TEAM_COLORS[team]
        if len(avg) >= 3:
            hull = _hull(avg[["x", "y"]].to_numpy())
            if hull is not None:
                ax.fill(hull[:, 0], hull[:, 1], color=color, alpha=0.10, zorder=1)
                ax.plot(np.append(hull[:, 0], hull[0, 0]),
                        np.append(hull[:, 1], hull[0, 1]),
                        color=color, lw=1.1, alpha=0.55, zorder=2)

        pitch.scatter(avg["x"], avg["y"], ax=ax, s=190, color=color,
                      edgecolors="white", linewidth=1.4, zorder=4)
        for tid, row in avg.iterrows():
            ax.text(row["x"], row["y"], str(int(tid)), color="white",
                    fontsize=6.2, ha="center", va="center",
                    zorder=5, fontweight="bold")

    ax.set_title("Average position and team shape  ·  11 most-tracked per team",
                 fontsize=9.5, color="#222", pad=6)


def _hull(pts: np.ndarray) -> Optional[np.ndarray]:
    try:
        from scipy.spatial import ConvexHull
        return pts[ConvexHull(pts).vertices]
    except Exception:
        return None


def _series_panel(ax, metrics: pd.DataFrame, column: str, teams: List[int],
                  title: str, ylabel: str, invert_note: str = "") -> None:
    """One metric over analysis elapsed time, per team, lightly smoothed."""
    for team in teams:
        sub = metrics[metrics["team"] == team].dropna(subset=[column])
        if sub.empty:
            continue
        sub = sub.sort_values("time_s")
        y = sub[column].rolling(9, center=True, min_periods=1).mean()
        ax.plot(sub["time_s"], y, color=TEAM_COLORS[team], lw=1.7,
                label=TEAM_NAMES[team], zorder=3)
        ax.plot(sub["time_s"], sub[column], color=TEAM_COLORS[team], lw=0.6,
                alpha=0.22, zorder=2)

    ax.set_title(title, fontsize=9.5, color="#222", pad=6)
    ax.set_xlabel("analysis elapsed (s)", fontsize=7.5, color="#666")
    ax.set_ylabel(ylabel, fontsize=7.5, color="#666")
    ax.tick_params(labelsize=7, colors="#666")
    ax.grid(True, color="#E8E8E4", lw=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D8D8D2")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    if invert_note:
        ax.text(0.01, 0.02, invert_note, transform=ax.transAxes,
                fontsize=6.5, color="#888", style="italic")


def render(
    tracking: pd.DataFrame,
    metrics: pd.DataFrame,
    out_path: str,
    title: str = "Match report",
    subtitle: str = "",
) -> str:
    """Render the six-panel report to `out_path`. Returns the path written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    teams = _teams_present(tracking)
    if not teams:
        raise ValueError("no outfield players with a resolved team in this data")

    fig = plt.figure(figsize=(13.5, 15.0), dpi=170)
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        4, 2, figure=fig,
        height_ratios=[0.20, 1.00, 1.00, 0.78],
        hspace=0.30, wspace=0.13,
        left=0.045, right=0.965, top=0.975, bottom=0.045,
    )

    # --- header -----------------------------------------------------------
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.set_xlim(0, 1)
    head.set_ylim(0, 1)
    # transAxes throughout: without it these land in data coordinates and
    # matplotlib autoscales the empty axis around them.
    head.text(0, 0.52, title, fontsize=21, fontweight="bold", color="#111",
              transform=head.transAxes, va="bottom")
    if subtitle:
        head.text(0, 0.34, subtitle, fontsize=9.5, color="#777",
                  transform=head.transAxes, va="top")
    head.plot([0, 1], [0.10, 0.10], transform=head.transAxes,
              color="#111", lw=1.3, clip_on=False)

    # --- row 1: occupancy heatmaps ---------------------------------------
    for i, team in enumerate(teams[:2]):
        _heatmap_panel(fig.add_subplot(gs[1, i]), tracking, team)

    # --- row 2: formation + territory ------------------------------------
    _formation_panel(fig.add_subplot(gs[2, 0]), tracking, teams)
    _series_panel(fig.add_subplot(gs[2, 1]), metrics, "control_pct", teams,
                  "Territory control (Voronoi)", "% of pitch controlled")

    # --- row 3: shape over time ------------------------------------------
    _series_panel(fig.add_subplot(gs[3, 0]), metrics, "compactness_m", teams,
                  "Compactness", "mean distance to centroid (m)",
                  invert_note="lower = more compact")
    _series_panel(fig.add_subplot(gs[3, 1]), metrics, "line_height_m", teams,
                  "Line height", "team centroid, along pitch (m)")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return out_path
