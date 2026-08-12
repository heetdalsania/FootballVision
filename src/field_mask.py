"""
Playing-Surface (Field) Mask for FootballVision

A general-purpose person detector finds *everyone* in a broadcast frame —
players, but also the crowd in the stands, staff on the touchline, and people
in advertising boards. On a stadium wide shot the crowd can outnumber the
players several times over, which wrecks every downstream metric (team shape,
pitch control, formation).

This module segments the green playing surface and provides an `on_field()`
test so the pipeline can keep only detections that actually stand on the
pitch. Method:

    1. Threshold the frame in HSV for grass-green (hue band, with generous
       saturation/value ranges so it survives floodlights, shadow, and the
       mown light/dark stripes).
    2. Morphological close + open to fill line markings and remove speckle.
    3. Keep the largest connected component — the pitch is by far the biggest
       contiguous green region; stray green (advertising, tunnel, crowd
       banners) is discarded.
    4. Fill that component's convex hull so players standing on grass (who
       themselves are *not* green) fall inside the mask.

A player is then judged by their **feet** (bottom-centre of the bbox), which
is where they contact the ground — the same point used for pitch homography.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Grass hue in OpenCV's 0-179 hue scale. Real pitches (and synthetic turf)
# sit roughly in 35-90; the range is wide to tolerate colour grading.
DEFAULT_HUE_LO = 30
DEFAULT_HUE_HI = 95


@dataclass
class FieldMask:
    """Binary mask of the playing surface, plus helpers to test points."""

    mask: np.ndarray          # uint8 {0,255}, same HxW as the frame
    coverage: float           # fraction of the frame that is field
    frame_shape: Tuple[int, int]

    def on_field(self, x: float, y: float, tolerance_px: int = 0) -> bool:
        """True if the point lies on the playing surface."""
        h, w = self.frame_shape
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            return False
        if self.mask[yi, xi] > 0:
            return True
        if tolerance_px > 0:
            y0, y1 = max(0, yi - tolerance_px), min(h, yi + tolerance_px + 1)
            x0, x1 = max(0, xi - tolerance_px), min(w, xi + tolerance_px + 1)
            return bool(self.mask[y0:y1, x0:x1].any())
        return False

    def bbox_on_field(self, bbox, tolerance_px: int = 12) -> bool:
        """
        True if a detection's ground-contact point is on the pitch.

        Uses the bottom-centre of the box (the feet) rather than the centre,
        so a tall player whose head overlaps the crowd still counts as on the
        field — and a spectator in the stands does not.
        """
        x1, y1, x2, y2 = bbox
        feet_x = (float(x1) + float(x2)) / 2.0
        feet_y = float(y2)
        return self.on_field(feet_x, feet_y, tolerance_px=tolerance_px)

    @property
    def is_usable(self) -> bool:
        """
        Whether the mask found a plausible pitch.

        If the frame isn't a football scene (or the grass wasn't found), the
        caller should fall back to keeping every detection rather than
        filtering everything away.
        """
        return 0.10 <= self.coverage <= 0.98


def build_field_mask(
    frame: np.ndarray,
    hue_lo: int = DEFAULT_HUE_LO,
    hue_hi: int = DEFAULT_HUE_HI,
    downscale: int = 4,
    use_convex_hull: bool = True,
) -> FieldMask:
    """
    Segment the playing surface from a BGR frame.

    Args:
        frame: BGR image.
        hue_lo/hue_hi: grass hue band (OpenCV 0-179 scale).
        downscale: work at 1/N resolution for speed, then upscale the mask.
        use_convex_hull: fill the pitch component's hull so players and lines
            inside the pitch are included.
    """
    import cv2

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // downscale), max(1, h // downscale)),
                       interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (hue_lo, 40, 40), (hue_hi, 255, 255))

    # Close gaps (pitch lines, players standing on grass), then despeckle.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k, iterations=2)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k, iterations=1)

    mask_small = _largest_component(green, use_convex_hull)

    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    coverage = float((mask > 0).mean())
    return FieldMask(mask=mask, coverage=coverage, frame_shape=(h, w))


def _largest_component(binary: np.ndarray, use_convex_hull: bool) -> np.ndarray:
    """Keep only the biggest blob; optionally fill its convex hull."""
    import cv2

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary

    biggest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(binary)
    shape = cv2.convexHull(biggest) if use_convex_hull else biggest
    cv2.drawContours(out, [shape], -1, 255, thickness=cv2.FILLED)
    return out


if __name__ == "__main__":
    import sys
    import cv2

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: python -m src.field_mask <image>")
        raise SystemExit(1)

    frame = cv2.imread(path)
    fm = build_field_mask(frame)
    print(f"frame={frame.shape} coverage={fm.coverage:.3f} usable={fm.is_usable}")
