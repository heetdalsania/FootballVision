"""Create short, silent event clips from a local source video with OpenCV."""

from __future__ import annotations

from pathlib import Path


def build_event_clip(
    video_path: str | Path,
    output_path: str | Path,
    event_time_s: float,
    before_s: float = 3.0,
    after_s: float = 3.0,
) -> str:
    import cv2

    video_path, output_path = Path(video_path), Path(output_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open source video: {video_path.name}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("source video dimensions are unavailable")

    start_s = max(0.0, float(event_time_s) - max(0.0, before_s))
    end_s = max(start_s + 0.25, float(event_time_s) + max(0.0, after_s))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("could not create MP4 event clip")

    written = 0
    while cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 <= end_s:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    if not written:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("the selected event is outside the source video")
    return str(output_path)
