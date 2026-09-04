"""Build downloadable analyst artifacts from persisted tactical snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


TRACKING_COLUMNS = ["frame_id", "time_s", "track_id", "team", "role", "x", "y"]
METRIC_COLUMNS = [
    "frame_id", "time_s", "team", "n", "compactness_m", "width_m",
    "depth_m", "line_height_m", "control_pct", "pressing_m",
]


def snapshots_to_frames(snapshots: Iterable[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the JSON-friendly history representation into report tables."""
    tracking_rows, metric_rows = [], []
    for snap in snapshots:
        frame_id = int(snap.get("frame_id") or 0)
        time_s = float(snap.get("time_s") or 0)
        for player in snap.get("players") or []:
            tracking_rows.append({
                "frame_id": frame_id,
                "time_s": time_s,
                "track_id": player.get("id", -1),
                "team": player.get("team", -1),
                "role": player.get("role", "player"),
                "x": player.get("x"),
                "y": player.get("y"),
            })

        concepts = snap.get("concepts") or {}
        control = concepts.get("control") or {}
        for team_key, stats in (concepts.get("teams") or {}).items():
            if not stats or str(team_key) not in ("0", "1"):
                continue
            metric_rows.append({
                "frame_id": frame_id,
                "time_s": time_s,
                "team": int(team_key),
                "n": stats.get("n"),
                "compactness_m": stats.get("compactness_m"),
                "width_m": stats.get("width_m"),
                "depth_m": stats.get("depth_m"),
                "line_height_m": stats.get("line_height_m"),
                "control_pct": control.get(str(team_key)),
                "pressing_m": concepts.get("pressing_m"),
            })

    return (
        pd.DataFrame(tracking_rows, columns=TRACKING_COLUMNS),
        pd.DataFrame(metric_rows, columns=METRIC_COLUMNS),
    )


def build_session_artifacts(
    snapshots: list[dict],
    output_dir: str | Path,
    session_id: int,
    source_name: str,
) -> dict:
    """Write tracking CSV, metrics CSV and (when possible) a report PNG."""
    from src.match_report import render

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking, metrics = snapshots_to_frames(snapshots)
    if tracking.empty:
        raise ValueError("this session has no saved player positions yet")

    stem = f"session-{int(session_id)}"
    tracking_path = output_dir / f"{stem}-tracking.csv"
    metrics_path = output_dir / f"{stem}-metrics.csv"
    report_path = output_dir / f"{stem}-report.png"
    tracking.to_csv(tracking_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    resolved = tracking[(tracking["role"] == "player") & tracking["team"].isin([0, 1])]
    if resolved.empty or metrics.empty:
        report = None
    else:
        render(
            tracking,
            metrics,
            str(report_path),
            title="FootballVision match report",
            subtitle=f"Session {session_id} · {source_name} · {len(snapshots)} sampled moments",
        )
        report = str(report_path)

    return {
        "tracking": str(tracking_path),
        "metrics": str(metrics_path),
        "report": report,
        "tracking_rows": len(tracking),
        "metric_rows": len(metrics),
    }
