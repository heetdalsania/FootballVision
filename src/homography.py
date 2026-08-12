"""
Pitch Homography for FootballVision

Maps player positions from broadcast-image pixel space to real-world
pitch coordinates on a standard 105 x 68 m football pitch. This is the
foundation of every real football-analysis metric: once tracks live in
metres on a top-down pitch (not perspective-warped pixels), you can
compute team shape, compactness, pressing, pitch control, and retrieve
similar situations.

A homography H is a 3x3 projective transform between two planes. For a
roughly planar surface viewed from a fixed camera (a football pitch on a
fixed screen-capture feed), a single H maps image pixels to pitch metres:

        [x_pitch]        [px]
        [y_pitch]  ~  H  [py]
        [   1   ]        [ 1]

Calibration is a 4+ point correspondence between known image landmarks
(centre mark, penalty-box corners, penalty spots, halfway-line ends) and
their known pitch coordinates. For a fixed capture angle one calibration
holds for the whole session, so H is computed once and stored.

Pitch coordinate convention (metres):
    origin (0, 0)      = bottom-left corner
    (PITCH_LENGTH, 0)  = bottom-right corner
    x in [0, 105]      = along the length (goal to goal)
    y in [0, 68]       = across the width
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard football pitch dimensions in metres (Emirates Stadium is 105 x 68).
PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Named pitch landmarks (metres) — used to build calibration correspondences.
# Coordinates follow the convention above: x along length, y across width.
# ---------------------------------------------------------------------------
PITCH_LANDMARKS: Dict[str, Point] = {
    "corner_bl": (0.0, 0.0),
    "corner_br": (PITCH_LENGTH, 0.0),
    "corner_tl": (0.0, PITCH_WIDTH),
    "corner_tr": (PITCH_LENGTH, PITCH_WIDTH),
    "center": (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
    "halfway_bottom": (PITCH_LENGTH / 2, 0.0),
    "halfway_top": (PITCH_LENGTH / 2, PITCH_WIDTH),
    # Penalty area is 16.5 m deep, 40.32 m wide (centred on the width).
    "penalty_box_l_top": (16.5, PITCH_WIDTH / 2 + 20.16),
    "penalty_box_l_bottom": (16.5, PITCH_WIDTH / 2 - 20.16),
    "penalty_box_r_top": (PITCH_LENGTH - 16.5, PITCH_WIDTH / 2 + 20.16),
    "penalty_box_r_bottom": (PITCH_LENGTH - 16.5, PITCH_WIDTH / 2 - 20.16),
    "penalty_spot_l": (11.0, PITCH_WIDTH / 2),
    "penalty_spot_r": (PITCH_LENGTH - 11.0, PITCH_WIDTH / 2),
}


@dataclass
class PitchHomography:
    """
    Perspective mapping from image pixels to pitch metres.

    Typical use:
        homo = PitchHomography()
        homo.calibrate(image_pts=[...4+ pixel points...],
                       pitch_pts=[...matching metre points...])
        x_m, y_m = homo.to_pitch(px, py)

    If never calibrated, `to_pitch` falls back to a naive linear map from the
    frame rectangle onto the pitch rectangle (see `set_frame_fallback`). This
    keeps the top-down view populated before the user calibrates, while
    `is_calibrated` stays False so the UI can flag it as approximate.
    """

    H: Optional[np.ndarray] = None
    frame_w: int = 1280
    frame_h: int = 720
    _reproj_error: float = field(default=0.0)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(
        self,
        image_pts: Sequence[Point],
        pitch_pts: Sequence[Point],
    ) -> float:
        """
        Compute the homography from >= 4 point correspondences.

        Args:
            image_pts: pixel (px, py) coordinates clicked on the frame.
            pitch_pts: matching (x, y) coordinates in metres on the pitch.

        Returns:
            Mean reprojection error in metres (lower is better).

        Raises:
            ValueError: if fewer than 4 correspondences are given, or the
                        points are degenerate (collinear) so no H is found.
        """
        if len(image_pts) != len(pitch_pts):
            raise ValueError(
                f"image_pts ({len(image_pts)}) and pitch_pts "
                f"({len(pitch_pts)}) must be the same length"
            )
        if len(image_pts) < 4:
            raise ValueError("Need at least 4 point correspondences to calibrate")

        import cv2

        src = np.asarray(image_pts, dtype=np.float64).reshape(-1, 1, 2)
        dst = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 1, 2)

        # RANSAC tolerates a mis-clicked point when > 4 correspondences given.
        method = cv2.RANSAC if len(image_pts) > 4 else 0
        H, _ = cv2.findHomography(src, dst, method, 5.0)
        if H is None:
            raise ValueError(
                "Homography could not be computed — points may be collinear "
                "or degenerate. Pick landmarks that span the pitch."
            )

        self.H = H
        self._reproj_error = self._compute_reproj_error(image_pts, pitch_pts)
        logger.info(
            "Pitch homography calibrated from %d points (reproj err %.2f m)",
            len(image_pts),
            self._reproj_error,
        )
        return self._reproj_error

    def calibrate_from_landmarks(
        self, correspondences: Dict[str, Point]
    ) -> float:
        """
        Convenience calibration keyed by named landmark.

        Args:
            correspondences: {landmark_name: (px, py)} where landmark_name is
                a key in PITCH_LANDMARKS. e.g. {"center": (640, 360), ...}
        """
        image_pts: List[Point] = []
        pitch_pts: List[Point] = []
        for name, px in correspondences.items():
            if name not in PITCH_LANDMARKS:
                raise ValueError(
                    f"Unknown landmark '{name}'. Valid: {sorted(PITCH_LANDMARKS)}"
                )
            image_pts.append(px)
            pitch_pts.append(PITCH_LANDMARKS[name])
        return self.calibrate(image_pts, pitch_pts)

    def set_frame_fallback(self, frame_w: int, frame_h: int) -> None:
        """Record frame size used by the uncalibrated linear fallback."""
        self.frame_w = int(frame_w) or 1280
        self.frame_h = int(frame_h) or 720

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def to_pitch(self, px: float, py: float) -> Point:
        """
        Map an image pixel to pitch coordinates in metres.

        Falls back to a linear frame->pitch map (perspective-naive) when no
        calibration is present, so callers always get a usable coordinate.
        """
        if self.H is None:
            return self._fallback_to_pitch(px, py)

        import cv2

        pt = np.array([[[float(px), float(py)]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(pt, self.H)
        x, y = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
        return x, y

    def to_pitch_batch(self, points: Sequence[Point]) -> List[Point]:
        """Vectorised `to_pitch` for many points at once."""
        if not points:
            return []
        if self.H is None:
            return [self._fallback_to_pitch(px, py) for px, py in points]

        import cv2

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in mapped]

    def _fallback_to_pitch(self, px: float, py: float) -> Point:
        """Naive linear map from frame rectangle to pitch rectangle."""
        x = (px / self.frame_w) * PITCH_LENGTH
        y = (1.0 - py / self.frame_h) * PITCH_WIDTH  # image y grows downward
        return x, y

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def is_calibrated(self) -> bool:
        return self.H is not None

    @property
    def reprojection_error(self) -> float:
        return self._reproj_error

    @staticmethod
    def on_pitch(x: float, y: float, margin: float = 3.0) -> bool:
        """True if (x, y) metres lands within the pitch (+ small margin)."""
        return (-margin <= x <= PITCH_LENGTH + margin) and (
            -margin <= y <= PITCH_WIDTH + margin
        )

    @staticmethod
    def clamp_to_pitch(x: float, y: float) -> Point:
        """Clamp a metre coordinate to the pitch rectangle."""
        return (
            min(max(x, 0.0), PITCH_LENGTH),
            min(max(y, 0.0), PITCH_WIDTH),
        )

    def _compute_reproj_error(
        self, image_pts: Sequence[Point], pitch_pts: Sequence[Point]
    ) -> float:
        mapped = self.to_pitch_batch(list(image_pts))
        errs = [
            float(np.hypot(mx - tx, my - ty))
            for (mx, my), (tx, ty) in zip(mapped, pitch_pts)
        ]
        return float(np.mean(errs)) if errs else 0.0

    # ------------------------------------------------------------------
    # Persistence — a fixed capture angle only needs calibrating once.
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.H is None:
            raise ValueError("Cannot save an uncalibrated homography")
        payload = {
            "H": self.H.tolist(),
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "reproj_error": self._reproj_error,
            "pitch_length": PITCH_LENGTH,
            "pitch_width": PITCH_WIDTH,
        }
        Path(path).write_text(json.dumps(payload, indent=2))
        logger.info("Saved pitch homography to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PitchHomography":
        data = json.loads(Path(path).read_text())
        homo = cls(
            H=np.asarray(data["H"], dtype=np.float64),
            frame_w=int(data.get("frame_w", 1280)),
            frame_h=int(data.get("frame_h", 720)),
        )
        homo._reproj_error = float(data.get("reproj_error", 0.0))
        return homo


if __name__ == "__main__":
    # Sanity demo: a synthetic camera looking at a pitch. We fabricate a
    # ground-truth homography, project the four corners to "pixels", then
    # recover it from those correspondences and confirm round-trip accuracy.
    import cv2

    true_pitch = [
        (0.0, 0.0),
        (PITCH_LENGTH, 0.0),
        (PITCH_LENGTH, PITCH_WIDTH),
        (0.0, PITCH_WIDTH),
    ]
    # A plausible broadcast-ish quadrilateral in a 1280x720 frame.
    image = [
        (300.0, 650.0),
        (980.0, 650.0),
        (1180.0, 250.0),
        (100.0, 250.0),
    ]
    homo = PitchHomography(frame_w=1280, frame_h=720)
    err = homo.calibrate(image, true_pitch)
    print(f"Reprojection error: {err:.4f} m  (calibrated={homo.is_calibrated})")

    cx, cy = homo.to_pitch(640, 450)
    print(f"Frame centre-ish (640,450) -> pitch ({cx:.1f}, {cy:.1f}) m")
    print(f"On pitch? {PitchHomography.on_pitch(cx, cy)}")
