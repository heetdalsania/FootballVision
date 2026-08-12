"""
Team Assignment for FootballVision

Assigns each tracked player to one of the two teams by clustering jersey
colour. A top-down tactical view is only football-credible once dots are
team-coloured — otherwise it is just a scatter of anonymous players.

Method:
    1. For each track, sample the torso region of its bounding box (the
       middle band, avoiding head/shorts/grass) and take the mean colour.
    2. Fit KMeans(2) over the torso colours of a frame's players → two team
       centroids in colour space.
    3. Cache the centroids and assign every subsequent player to the nearest
       one, so team labels stay stable frame-to-frame. Periodically refit as
       lighting / camera changes.

Colour space: HSV, which separates hue (kit colour) from brightness better
than raw BGR under stadium lighting. Referees and goalkeepers are colour
outliers; for the MVP they simply fall into whichever team centroid is
nearer — good enough for team shape. A dedicated outlier class is a later
refinement.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TEAM_A = 0
TEAM_B = 1
TEAM_UNKNOWN = -1


class TeamAssigner:
    """
    Two-team jersey-colour classifier.

    Usage:
        assigner = TeamAssigner()
        assigner.assign(tracks, frame)   # sets track.team on each track
    """

    def __init__(self, refit_every: int = 60, min_players: int = 6):
        """
        Args:
            refit_every: refit the KMeans centroids every N calls (0 = fit
                once and never refit). Refitting adapts to lighting drift.
            min_players: minimum torso samples needed before a fit is trusted.
        """
        self.refit_every = refit_every
        self.min_players = min_players
        self._centroids: Optional[np.ndarray] = None  # shape (2, 3) in HSV
        self._call_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    #: Detected classes that must never influence the two team centroids.
    #: Referees wear deliberately distinct colours and goalkeepers wear a kit
    #: different from their own outfield team, so including them drags a
    #: centroid away from the kit it is supposed to represent.
    NON_OUTFIELD = {"referee", "goalkeeper"}

    def assign(self, tracks: Sequence, frame: np.ndarray) -> None:
        """
        Assign a team to every track in-place (sets `track.team`).

        Args:
            tracks: iterable of Track objects exposing `.bbox` and a writable
                `.team` attribute.
            frame: the BGR frame the tracks were detected in.
        """
        if frame is None or len(tracks) == 0:
            return

        colours = [self._torso_colour(t.bbox, frame) for t in tracks]
        valid = [(t, c) for t, c in zip(tracks, colours) if c is not None]
        if not valid:
            return

        # Fit the two team centroids on outfield players only.
        fit_samples = [
            c for t, c in valid
            if str(getattr(t, "class_name", "")).lower() not in self.NON_OUTFIELD
        ] or [c for _, c in valid]

        self._call_count += 1
        need_fit = self._centroids is None or (
            self.refit_every and self._call_count % self.refit_every == 0
        )
        if need_fit and len(fit_samples) >= self.min_players:
            self._fit(fit_samples)

        if self._centroids is None:
            # Not enough data to fit yet — leave teams unknown.
            for t, _ in valid:
                t.team = TEAM_UNKNOWN
            return

        for t, c in valid:
            t.team = self._nearest_team(c)

    def team_counts(self, tracks: Sequence) -> Dict[int, int]:
        """Count players per team among the given tracks."""
        counts = {TEAM_A: 0, TEAM_B: 0, TEAM_UNKNOWN: 0}
        for t in tracks:
            counts[getattr(t, "team", TEAM_UNKNOWN)] = (
                counts.get(getattr(t, "team", TEAM_UNKNOWN), 0) + 1
            )
        return counts

    @property
    def is_fitted(self) -> bool:
        return self._centroids is not None

    def reset(self) -> None:
        self._centroids = None
        self._call_count = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _torso_colour(
        self, bbox: Tuple[int, int, int, int], frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Mean HSV colour of the torso band of a player's bbox."""
        import cv2

        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 3 or y2 - y1 < 6:
            return None

        # Torso = central 50% horizontally, upper-middle band vertically
        # (skip ~20% head at top and ~40% legs at bottom).
        bw, bh = x2 - x1, y2 - y1
        tx1 = x1 + int(0.25 * bw)
        tx2 = x2 - int(0.25 * bw)
        ty1 = y1 + int(0.20 * bh)
        ty2 = y1 + int(0.60 * bh)
        if tx2 <= tx1 or ty2 <= ty1:
            return None

        patch = frame[ty1:ty2, tx1:tx2]
        if patch.size == 0:
            return None

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return hsv.reshape(-1, 3).mean(axis=0)

    def _fit(self, colours: List[np.ndarray]) -> None:
        """Fit two team centroids from torso colours."""
        data = np.asarray(colours, dtype=np.float64)
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=2, n_init=5, random_state=42)
            km.fit(self._encode_hue(data))
            centroids = km.cluster_centers_
        except Exception as exc:  # sklearn missing or degenerate input
            logger.debug("KMeans unavailable/failed (%s); using median split", exc)
            centroids = self._median_split(self._encode_hue(data))

        # Keep a stable ordering so TEAM_A/TEAM_B don't swap on refit.
        if self._centroids is not None:
            centroids = self._align_to_previous(centroids)
        self._centroids = centroids

    def _nearest_team(self, colour: np.ndarray) -> int:
        enc = self._encode_hue(colour.reshape(1, 3))[0]
        d0 = np.linalg.norm(enc - self._centroids[0])
        d1 = np.linalg.norm(enc - self._centroids[1])
        return TEAM_A if d0 <= d1 else TEAM_B

    @staticmethod
    def _encode_hue(hsv: np.ndarray) -> np.ndarray:
        """
        Encode HSV so both coloured and achromatic kits separate cleanly.

        Hue (0-180 in OpenCV) is an angle, so 179 and 1 are close; it is mapped
        to (cos, sin) on the unit circle scaled by saturation, because grey and
        white pixels carry no reliable hue.

        Saturation and value are then given real weight, not a token 0.3. The
        motivating case is Liverpool red vs Real Madrid white: white is
        *achromatic* (saturation ~0), so a hue-dominated encoding places every
        white shirt near the origin alongside anything else desaturated, and
        the two teams fail to separate. Saturation is the discriminating axis
        there, and brightness separates white from black kits.
        """
        h = hsv[:, 0] * (np.pi / 90.0)  # 0..180 -> 0..2pi
        s = hsv[:, 1] / 255.0
        v = hsv[:, 2] / 255.0
        return np.stack(
            [np.cos(h) * s, np.sin(h) * s, s * 1.2, v * 0.9], axis=1
        )

    @staticmethod
    def _median_split(enc: np.ndarray) -> np.ndarray:
        """Fallback 2-cluster split along the highest-variance axis."""
        axis = int(np.argmax(enc.var(axis=0)))
        med = np.median(enc[:, axis])
        low = enc[enc[:, axis] <= med]
        high = enc[enc[:, axis] > med]
        c0 = low.mean(axis=0) if len(low) else enc.mean(axis=0)
        c1 = high.mean(axis=0) if len(high) else enc.mean(axis=0)
        return np.stack([c0, c1], axis=0)

    def _align_to_previous(self, centroids: np.ndarray) -> np.ndarray:
        """Reorder new centroids to minimise drift from cached labels."""
        keep = np.linalg.norm(centroids - self._centroids, axis=1).sum()
        swap = (
            np.linalg.norm(centroids[0] - self._centroids[1])
            + np.linalg.norm(centroids[1] - self._centroids[0])
        )
        return centroids[::-1] if swap < keep else centroids


if __name__ == "__main__":
    # Smoke test with two synthetic teams (red vs blue jerseys) on green grass.
    import cv2

    class _T:
        def __init__(self, bbox):
            self.bbox = bbox
            self.team = TEAM_UNKNOWN

    frame = np.full((720, 1280, 3), (40, 120, 40), dtype=np.uint8)  # green
    tracks = []
    for i in range(6):  # red team
        x = 100 + i * 40
        cv2.rectangle(frame, (x, 300), (x + 24, 380), (0, 0, 200), -1)
        tracks.append(_T((x, 300, x + 24, 380)))
    for i in range(6):  # blue team
        x = 700 + i * 40
        cv2.rectangle(frame, (x, 300), (x + 24, 380), (200, 0, 0), -1)
        tracks.append(_T((x, 300, x + 24, 380)))

    TeamAssigner(min_players=4).assign(tracks, frame)
    reds = [t.team for t in tracks[:6]]
    blues = [t.team for t in tracks[6:]]
    print("red team labels :", reds)
    print("blue team labels:", blues)
    ok = len(set(reds)) == 1 and len(set(blues)) == 1 and reds[0] != blues[0]
    print("separated cleanly:", ok)
