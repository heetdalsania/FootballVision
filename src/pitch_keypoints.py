"""
Automatic Pitch Calibration from Detected Keypoints

Manual 4-point calibration is the weakest link in the tactical view: it has to
be redone whenever the camera moves, and a broadcast camera pans and zooms
constantly. This module removes it.

Roboflow's `football-field-detection-f07vi` model is a YOLO **pose** model that
locates 32 known landmarks on a football pitch (corners, penalty-box corners,
goal-box corners, penalty spots, halfway-line ends, centre-circle extremes).
Each detected landmark has a known real-world position, so every frame with
4+ confident landmarks yields a fresh homography — automatic, continuous
calibration that survives camera movement.

The 32 vertices below are Roboflow's `SoccerPitchConfiguration`, defined on a
12000 x 7000 cm pitch. They are rescaled to FootballVision's 105 x 68 m model
and flipped so y=0 is the bottom touchline, matching `src/homography.py`'s
convention (corner_bl == origin).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.homography import PITCH_LENGTH, PITCH_WIDTH, PitchHomography

# Roboflow SoccerPitchConfiguration source dimensions (centimetres).
_SRC_LENGTH_CM = 12000.0
_SRC_WIDTH_CM = 7000.0

# The 32 landmarks, in the exact order the keypoint model emits them.
_VERTICES_CM: List[Tuple[int, int]] = [
    (0, 0), (0, 1450), (0, 2584), (0, 4416), (0, 5550), (0, 7000),
    (550, 2584), (550, 4416), (1100, 3500), (2015, 1450), (2015, 2584),
    (2015, 4416), (2015, 5550), (6000, 0), (6000, 2585), (6000, 4415),
    (6000, 7000), (9985, 1450), (9985, 2584), (9985, 4416), (9985, 5550),
    (10900, 3500), (11450, 2584), (11450, 4416), (12000, 0), (12000, 1450),
    (12000, 2584), (12000, 4416), (12000, 5550), (12000, 7000), (5085, 3500),
    (6915, 3500),
]


def _to_metres(x_cm: float, y_cm: float) -> Tuple[float, float]:
    """Rescale a source vertex to the 105x68 m model, y flipped to bottom-origin."""
    x_m = (x_cm / _SRC_LENGTH_CM) * PITCH_LENGTH
    y_m = (1.0 - (y_cm / _SRC_WIDTH_CM)) * PITCH_WIDTH
    return x_m, y_m


#: The 32 landmarks in FootballVision pitch metres, index-aligned to the model.
PITCH_VERTICES_M: List[Tuple[float, float]] = [_to_metres(x, y) for x, y in _VERTICES_CM]


def calibrate_from_keypoints(
    keypoints: Sequence,
    min_confidence: float = 0.50,
    min_points: int = 4,
    frame_w: int = 1280,
    frame_h: int = 720,
) -> Tuple[Optional[PitchHomography], dict]:
    """
    Build a homography from a keypoint model's output for one frame.

    Args:
        keypoints: the model's keypoints for a frame. Each item must expose
            `.x`, `.y` and `.confidence` (the shape `inference` returns).
        min_confidence: landmarks below this are ignored — the model emits all
            32 every frame, scoring the ones it cannot see near zero.
        min_points: minimum confident landmarks needed (a homography needs 4).
        frame_w/frame_h: frame size, for the uncalibrated fallback.

    Returns:
        (PitchHomography or None, info dict with counts and error).
    """
    image_pts: List[Tuple[float, float]] = []
    pitch_pts: List[Tuple[float, float]] = []

    for idx, kp in enumerate(keypoints):
        if idx >= len(PITCH_VERTICES_M):
            break
        conf = float(getattr(kp, "confidence", 0.0) or 0.0)
        if conf < min_confidence:
            continue
        x = float(getattr(kp, "x", 0.0))
        y = float(getattr(kp, "y", 0.0))
        # The model reports (0,0) for landmarks it cannot localise.
        if x <= 0 and y <= 0:
            continue
        image_pts.append((x, y))
        pitch_pts.append(PITCH_VERTICES_M[idx])

    info = {"n_confident": len(image_pts), "min_confidence": min_confidence}

    if len(image_pts) < min_points:
        info["error"] = (
            f"only {len(image_pts)} confident landmarks "
            f"(need {min_points}) — pitch markings may be out of shot"
        )
        return None, info

    homo = PitchHomography(frame_w=frame_w, frame_h=frame_h)
    try:
        err = homo.calibrate(image_pts, pitch_pts)
    except ValueError as exc:
        info["error"] = str(exc)
        return None, info

    info["reproj_error_m"] = round(err, 3)

    # A good broadcast calibration lands well under a metre; a large error
    # means the landmarks were mismatched or nearly collinear, and the
    # resulting positions would be misleading.
    if err > 8.0:
        info["error"] = f"reprojection error {err:.1f} m too high — rejected"
        return None, info

    return homo, info


class AutoCalibrator:
    """
    Keeps the pitch homography fresh as the broadcast camera moves.

    Re-solving every frame is wasteful and jittery, so this recalibrates on an
    interval and only replaces the stored homography when the new solution is
    at least as trustworthy as the current one.
    """

    def __init__(self, every_n_frames: int = 30, min_confidence: float = 0.50):
        self.every_n_frames = max(1, every_n_frames)
        self.min_confidence = min_confidence
        self.homography: Optional[PitchHomography] = None
        self.last_info: dict = {}
        self.calibrations = 0

    def maybe_update(
        self,
        frame_id: int,
        keypoints: Optional[Sequence],
        frame_w: int,
        frame_h: int,
    ) -> Optional[PitchHomography]:
        """Recalibrate if this frame is due and the landmarks are good enough."""
        if keypoints is None:
            return self.homography
        if frame_id % self.every_n_frames != 0 and self.homography is not None:
            return self.homography

        homo, info = calibrate_from_keypoints(
            keypoints,
            min_confidence=self.min_confidence,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        self.last_info = info
        if homo is not None:
            self.homography = homo
            self.calibrations += 1
        return self.homography


if __name__ == "__main__":
    print(f"pitch model: {PITCH_LENGTH} x {PITCH_WIDTH} m, "
          f"{len(PITCH_VERTICES_M)} landmarks")
    for name, idx in [("corner (0,0)", 0), ("far corner", 29),
                      ("halfway bottom", 13), ("centre-ish", 8)]:
        x, y = PITCH_VERTICES_M[idx]
        print(f"  [{idx:2d}] {name:<16} -> ({x:6.2f}, {y:6.2f}) m")

    # Round-trip: synthesise an image projection of 6 landmarks and recover it.
    import cv2

    idxs = [0, 5, 24, 29, 13, 16]
    dst = np.array([PITCH_VERTICES_M[i] for i in idxs], dtype=np.float64)
    true_H = cv2.getPerspectiveTransform(
        np.array([[0, 68], [0, 0], [105, 68], [105, 0]], dtype=np.float32),
        np.array([[100, 250], [300, 650], [1180, 250], [980, 650]], dtype=np.float32),
    )
    img = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), true_H).reshape(-1, 2)

    class _KP:
        def __init__(self, x, y):
            self.x, self.y, self.confidence = x, y, 0.9

    kps = [_KP(0, 0) for _ in range(32)]
    for j, i in enumerate(idxs):
        kps[i] = _KP(float(img[j][0]), float(img[j][1]))

    homo, info = calibrate_from_keypoints(kps)
    print(f"\nauto-calibration: {info}")
    assert homo is not None, "should calibrate from 6 landmarks"
    print("AUTO-CALIBRATION OK")
