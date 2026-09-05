"""Build downloadable analyst artifacts from persisted tactical snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import json

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
    events: list[dict],
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
    events_path = output_dir / f"{stem}-events.csv"
    report_path = output_dir / f"{stem}-report.png"
    tracking.to_csv(tracking_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    event_rows = []
    for event in events:
        event_rows.append({
            "time_s": event.get("time_s"),
            "source_time_s": event.get("source_time_s"),
            "frame_id": event.get("frame_id"),
            "type": event.get("type"),
            "label": event.get("label"),
            "team": event.get("team"),
            "player_id": event.get("player_id"),
            "confidence": event.get("confidence"),
            "x": event.get("x"),
            "y": event.get("y"),
            "detail": json.dumps(event.get("detail") or {}, separators=(",", ":")),
            "clip_path": event.get("clip_path"),
        })
    pd.DataFrame(event_rows, columns=[
        "time_s", "source_time_s", "frame_id", "type", "label", "team",
        "player_id", "confidence", "x", "y", "detail", "clip_path",
    ]).to_csv(events_path, index=False)

    resolved = tracking[(tracking["role"] == "player") & tracking["team"].isin([0, 1])]
    if resolved.empty or metrics.empty:
        report = None
    else:
        counts = pd.Series([e.get("type") for e in events]).value_counts()
        labels = {
            "possession_start": ("possession start", "possession starts"),
            "pass": ("pass", "passes"),
            "turnover": ("turnover", "turnovers"),
            "carry": ("carry", "carries"),
            "restart": ("restart", "restarts"),
            "shot_candidate": ("shot candidate", "shot candidates"),
        }
        summary = ", ".join(
            f"{int(count)} {labels.get(kind, (kind.replace('_', ' '), kind.replace('_', ' ') + 's'))[count != 1]}"
            for kind, count in counts.items()
        ) or "no inferred events"
        render(
            tracking,
            metrics,
            str(report_path),
            title="FootballVision match report",
            subtitle=(f"Session {session_id} · {source_name} · "
                      f"{len(snapshots)} sampled moments · {summary}"),
        )
        report = str(report_path)

    return {
        "tracking": str(tracking_path),
        "metrics": str(metrics_path),
        "events": str(events_path),
        "report": report,
        "tracking_rows": len(tracking),
        "metric_rows": len(metrics),
    }
