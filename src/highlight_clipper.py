"""
Highlight Clipper for FootballVision.

Maintains a ring buffer of recent frames. When a snap is detected,
it flushes the pre-snap buffer + captures the next N seconds and
writes an MP4 clip to disk.
"""

import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLIPS_DIR = Path(__file__).resolve().parent.parent / "clips"
CLIPS_DIR.mkdir(exist_ok=True)


class HighlightClipper:
    """
    Ring-buffer based highlight clipper.

    Usage:
        clipper = HighlightClipper(fps=10, pre_snap_secs=3, post_snap_secs=5)
        clipper.push_frame(frame)          # call every frame
        if snap_detected:
            clipper.trigger_clip()         # start recording post-snap
        path = clipper.tick()              # call every frame; returns clip path when done
    """

    def __init__(
        self,
        fps: int = 10,
        pre_snap_secs: float = 3.0,
        post_snap_secs: float = 5.0,
        frame_width: int = 1280,
        frame_height: int = 720,
    ):
        self.fps = fps
        self.pre_frames  = int(pre_snap_secs  * fps)
        self.post_frames = int(post_snap_secs * fps)
        self.frame_width  = frame_width
        self.frame_height = frame_height

        self._ring: deque = deque(maxlen=self.pre_frames)
        self._recording = False
        self._post_buffer: list = []
        self._clip_path: Optional[Path] = None

    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray):
        """Add a frame to the ring buffer (always called)."""
        small = cv2.resize(frame, (self.frame_width, self.frame_height))
        self._ring.append(small)

        if self._recording:
            self._post_buffer.append(small)

    def trigger_clip(self):
        """Start recording post-snap frames. Safe to call multiple times."""
        if self._recording:
            return  # Already recording
        logger.info("[HighlightClipper] Snap detected — capturing clip")
        self._recording = True
        self._post_buffer = []

    def tick(self) -> Optional[str]:
        """
        Call every frame while recording. Returns clip path when the
        post-snap buffer is full and the clip has been written.
        """
        if not self._recording:
            return None

        if len(self._post_buffer) >= self.post_frames:
            path = self._write_clip()
            self._recording = False
            self._post_buffer = []
            return path

        return None

    # ------------------------------------------------------------------

    def _write_clip(self) -> str:
        """Write pre+post frames to an MP4 file and return its path."""
        ts = time.strftime("%H-%M-%S")
        out_path = CLIPS_DIR / f"highlight_{ts}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )

        frames = list(self._ring) + self._post_buffer
        for f in frames:
            writer.write(f)

        writer.release()
        logger.info(f"[HighlightClipper] Clip saved: {out_path}")
        return str(out_path.name)
