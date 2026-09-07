"""Create a local tactical-overlay video from persisted session measurements."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


TEAM_COLORS = {0: (77, 72, 229), 1: (246, 130, 59), -1: (166, 160, 154)}


def _snapshot_time(snapshot: dict) -> float:
    value = snapshot.get("source_time_s")
    return float(value if value is not None else snapshot.get("time_s") or 0)


def build_annotated_video(
    source_path: str | Path,
    output_path: str | Path,
    snapshots: list[dict],
    events: list[dict],
    player_labels: Optional[list[dict]] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> str:
    """Render source footage with a measured top-down inset and event banner."""
    if not snapshots:
        raise ValueError("this session has no saved pitch states")
    source_path, output_path = Path(source_path), Path(output_path)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video dimensions are unavailable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create annotated video: {output_path}")

    ordered = sorted(snapshots, key=_snapshot_time)
    times = [_snapshot_time(snapshot) for snapshot in ordered]
    accepted_events = [
        event for event in events
        if event.get("review_status") != "rejected"
        and event.get("source_time_s") is not None
    ]
    labels = {
        int(label["track_id"]): label for label in (player_labels or [])
    }
    frame_no = 0
    last_progress = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            source_time = max(0.0, float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            index = max(0, min(len(ordered) - 1, bisect_right(times, source_time) - 1))
            snapshot = ordered[index]
            _draw_overlay(frame, snapshot, accepted_events, labels, source_time)
            writer.write(frame)
            pct = int(100 * frame_no / total) if total else 0
            if progress and pct != last_progress:
                progress(float(pct))
                last_progress = pct
    finally:
        cap.release()
        writer.release()
    if frame_no == 0 or not output_path.is_file():
        raise RuntimeError("No frames were written to the annotated video")
    if progress:
        progress(100.0)
    return str(output_path)


def _draw_overlay(
    frame: np.ndarray,
    snapshot: dict,
    events: list[dict],
    labels: dict[int, dict],
    source_time: float,
) -> None:
    height, width = frame.shape[:2]
    font_scale = max(.38, min(.7, width / 1600))
    bar_h = max(32, int(height * .055))
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (width, bar_h), (8, 10, 14), -1)
    cv2.addWeighted(shade, .78, frame, .22, 0, frame)

    intelligence = snapshot.get("intelligence") or {}
    possession = intelligence.get("possession") or {}
    team = possession.get("team")
    possession_text = (
        f"Team {'A' if team == 0 else 'B'} possession"
        if team in (0, 1) else "Possession unresolved"
    )
    forms = intelligence.get("formations") or {}
    form_a = (forms.get("0") or {}).get("name", "-")
    form_b = (forms.get("1") or {}).get("name", "-")
    title = (
        f"FootballVision   {source_time // 60:02.0f}:{source_time % 60:04.1f}   "
        f"{possession_text}   A {form_a} / B {form_b}"
    )
    cv2.putText(
        frame, title, (12, int(bar_h * .68)), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (244, 244, 244), 1, cv2.LINE_AA,
    )

    inset_w = max(180, int(width * .27))
    inset_h = max(116, int(inset_w * 68 / 105))
    inset_w = min(inset_w, width - 20)
    inset_h = min(inset_h, height - bar_h - 20)
    left, top = width - inset_w - 10, bar_h + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (left + inset_w, top + inset_h), (12, 48, 28), -1)
    cv2.addWeighted(overlay, .84, frame, .16, 0, frame)
    line_color = (190, 210, 195)
    cv2.rectangle(frame, (left, top), (left + inset_w, top + inset_h), line_color, 1)
    cv2.line(
        frame, (left + inset_w // 2, top),
        (left + inset_w // 2, top + inset_h), line_color, 1,
    )

    def point(x, y):
        return (
            left + int(float(x) / 105.0 * inset_w),
            top + inset_h - int(float(y) / 68.0 * inset_h),
        )

    for player in snapshot.get("players") or []:
        if player.get("role") not in ("player", "goalkeeper"):
            continue
        player_team = int(player.get("team", -1))
        center = point(player.get("x", 0), player.get("y", 0))
        cv2.circle(frame, center, 4, TEAM_COLORS.get(player_team, TEAM_COLORS[-1]), -1)
        label = labels.get(int(player.get("id", -1))) or {}
        shown = label.get("shirt_number")
        if shown is not None:
            cv2.putText(
                frame, str(shown), (center[0] + 4, center[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX, .28, (255, 255, 255), 1, cv2.LINE_AA,
            )
    ball = snapshot.get("ball")
    if ball:
        cv2.circle(frame, point(ball.get("x", 0), ball.get("y", 0)), 4, (255, 255, 255), -1)

    nearby = min(
        events,
        key=lambda event: abs(float(event["source_time_s"]) - source_time),
        default=None,
    )
    if nearby and abs(float(nearby["source_time_s"]) - source_time) <= .75:
        text = nearby.get("label") or nearby.get("type") or "Event"
        status = nearby.get("review_status") or "inferred"
        banner = f"{text}  |  {status}"
        size = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        y = height - 18
        cv2.rectangle(frame, (10, y - size[1] - 10), (24 + size[0], y + 6), (8, 10, 14), -1)
        cv2.putText(
            frame, banner, (17, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
