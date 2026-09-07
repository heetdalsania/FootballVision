"""Lightweight constant-velocity ball recovery in pitch coordinates."""

from __future__ import annotations

from math import hypot
from typing import Optional, Tuple


class BallMotionTracker:
    """Smooth detections and predict through short visual occlusions."""

    def __init__(
        self,
        max_age_frames: int = 12,
        smoothing: float = .48,
        velocity_smoothing: float = .55,
    ):
        self.max_age_frames = max(1, int(max_age_frames))
        self.smoothing = float(smoothing)
        self.velocity_smoothing = float(velocity_smoothing)
        self.reset()

    def reset(self) -> None:
        self.position: Optional[Tuple[float, float]] = None
        self.velocity = (0.0, 0.0)
        self.last_frame = 0

    def observe(self, point: Tuple[float, float], frame_id: int) -> dict:
        point = (float(point[0]), float(point[1]))
        frame_id = int(frame_id)
        if self.position is None:
            self.position, self.last_frame = point, frame_id
            return self._payload(False, 0, 1.0)

        dt = max(1, frame_id - self.last_frame)
        # A one-frame teleport across a quarter of the pitch is almost always
        # a false white-object detection. Keep predicting instead.
        if hypot(point[0] - self.position[0], point[1] - self.position[1]) > 25 * dt:
            predicted = self.predict(frame_id)
            return predicted or self._payload(True, dt, .1)

        previous = self.position
        alpha = self.smoothing
        smoothed = (
            previous[0] + alpha * (point[0] - previous[0]),
            previous[1] + alpha * (point[1] - previous[1]),
        )
        measured_velocity = (
            (smoothed[0] - previous[0]) / dt,
            (smoothed[1] - previous[1]) / dt,
        )
        beta = self.velocity_smoothing
        self.velocity = (
            beta * measured_velocity[0] + (1 - beta) * self.velocity[0],
            beta * measured_velocity[1] + (1 - beta) * self.velocity[1],
        )
        self.position, self.last_frame = smoothed, frame_id
        return self._payload(False, 0, 1.0)

    def predict(self, frame_id: int) -> Optional[dict]:
        if self.position is None:
            return None
        age = int(frame_id) - self.last_frame
        if age <= 0:
            return self._payload(False, 0, 1.0)
        if age > self.max_age_frames:
            return None
        x = min(105.0, max(0.0, self.position[0] + self.velocity[0] * age))
        y = min(68.0, max(0.0, self.position[1] + self.velocity[1] * age))
        confidence = max(.05, 1.0 - age / (self.max_age_frames + 1))
        return {
            "x": round(x, 2), "y": round(y, 2), "stale": True,
            "predicted": True, "age_frames": age,
            "confidence": round(confidence, 2),
        }

    def _payload(self, stale: bool, age: int, confidence: float) -> dict:
        return {
            "x": round(self.position[0], 2),
            "y": round(self.position[1], 2),
            "stale": stale,
            "predicted": False,
            "age_frames": age,
            "confidence": round(confidence, 2),
        }
