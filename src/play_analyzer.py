"""
Play Analyzer Module for FootballVision
Heuristic run/pass probability, snap detection, and play-level stats.
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class PlayResult:
    """Outcome of a single play analysis."""
    play_type: str          # "run", "pass", or "unknown"
    run_prob: float
    pass_prob: float
    snap_detected: bool
    snap_frame: Optional[int]
    formation_hint: str     # offensive hint derived from positions
    alerts: List[str]       # coaching alerts (zone coverage, blitz, etc.)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "play_type": self.play_type,
            "run_prob": round(self.run_prob, 3),
            "pass_prob": round(self.pass_prob, 3),
            "snap_detected": self.snap_detected,
            "snap_frame": self.snap_frame,
            "formation_hint": self.formation_hint,
            "alerts": self.alerts,
            "timestamp": self.timestamp,
        }


class PlayAnalyzer:
    """
    Heuristic play analyzer.

    Detects snaps via collective velocity burst, estimates run/pass probability
    from player spread and depth, and generates coaching alerts.

    Example:
        analyzer = PlayAnalyzer()
        result = analyzer.update(tracks, frame_id=100, frame_w=1920, frame_h=1080)
        print(result.play_type, result.alerts)
    """

    # Snap detection: avg player velocity magnitude must exceed this (px/frame)
    SNAP_VELOCITY_THRESHOLD = 6.0
    # Minimum number of tracks moving fast together to confirm snap
    SNAP_MIN_MOVING_PLAYERS = 5
    # Frames to suppress re-detection after a snap
    SNAP_COOLDOWN_FRAMES = 90

    def __init__(self, history_len: int = 30):
        self._velocity_history: deque = deque(maxlen=history_len)
        self._frames_since_snap: int = self.SNAP_COOLDOWN_FRAMES  # start ready
        self._snap_frame: Optional[int] = None
        self._play_count: int = 0
        self._play_type_history: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self,
               tracks: list,
               frame_id: int,
               frame_w: int = 1920,
               frame_h: int = 1080) -> PlayResult:
        """
        Process a new frame of tracks and return play analysis.

        Args:
            tracks: List of Track objects (must have .velocity and .center).
            frame_id: Monotonically increasing frame counter.
            frame_w: Frame width for normalization.
            frame_h: Frame height for normalization.

        Returns:
            PlayResult with snap status, probabilities, and alerts.
        """
        snap_detected = self._detect_snap(tracks, frame_id)

        run_prob, pass_prob = self._estimate_play_type(tracks, frame_w, frame_h)
        formation_hint = self._formation_hint(tracks, frame_w, frame_h)
        alerts = self._generate_alerts(tracks, frame_w, frame_h)

        play_type = "unknown"
        if snap_detected:
            play_type = "pass" if pass_prob > run_prob else "run"
            self._play_type_history.append(play_type)
            self._play_count += 1

        return PlayResult(
            play_type=play_type,
            run_prob=run_prob,
            pass_prob=pass_prob,
            snap_detected=snap_detected,
            snap_frame=self._snap_frame if snap_detected else None,
            formation_hint=formation_hint,
            alerts=alerts,
        )

    def reset(self):
        """Reset state for a new game or session."""
        self._velocity_history.clear()
        self._frames_since_snap = self.SNAP_COOLDOWN_FRAMES
        self._snap_frame = None
        self._play_count = 0
        self._play_type_history.clear()

    @property
    def play_count(self) -> int:
        return self._play_count

    def tendency_stats(self) -> Dict[str, float]:
        """Run/pass tendency over the game so far."""
        if not self._play_type_history:
            return {"run": 0.0, "pass": 0.0, "total": 0}
        total = len(self._play_type_history)
        runs = self._play_type_history.count("run")
        return {
            "run": round(runs / total, 3),
            "pass": round(1 - runs / total, 3),
            "total": total,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_snap(self, tracks: list, frame_id: int) -> bool:
        """Snap = sudden burst of collective motion after a quiet pre-snap."""
        self._frames_since_snap += 1

        if not tracks:
            return False

        # Compute per-player velocity magnitudes
        mags = [
            float(np.sqrt(t.velocity[0] ** 2 + t.velocity[1] ** 2))
            for t in tracks
            if hasattr(t, "velocity")
        ]
        if not mags:
            return False

        avg_mag = float(np.mean(mags))
        moving = sum(1 for m in mags if m > self.SNAP_VELOCITY_THRESHOLD * 0.5)
        self._velocity_history.append(avg_mag)

        # Need enough history and cooldown elapsed
        if (
            len(self._velocity_history) < 5
            or self._frames_since_snap < self.SNAP_COOLDOWN_FRAMES
        ):
            return False

        # Recent avg vs baseline: burst detection
        recent_avg = float(np.mean(list(self._velocity_history)[-5:]))
        baseline = float(np.mean(list(self._velocity_history)[:-5])) if len(self._velocity_history) > 5 else 0.0

        is_burst = (
            recent_avg > self.SNAP_VELOCITY_THRESHOLD
            and recent_avg > baseline * 2.0
            and moving >= self.SNAP_MIN_MOVING_PLAYERS
        )

        if is_burst:
            self._frames_since_snap = 0
            self._snap_frame = frame_id
            return True

        return False

    def _estimate_play_type(self,
                             tracks: list,
                             frame_w: int,
                             frame_h: int) -> Tuple[float, float]:
        """
        Heuristic run/pass probability.

        Wider spread → more likely pass (spread offense / routes).
        Tight formation + blockers bunched → more likely run.
        """
        if not tracks:
            return 0.5, 0.5

        xs = [t.center[0] / frame_w for t in tracks]
        ys = [t.center[1] / frame_h for t in tracks]

        x_spread = float(np.std(xs)) if len(xs) > 1 else 0.0
        y_depth = max(ys) - min(ys) if ys else 0.0
        player_count = len(tracks)

        # Wide spread + deep pocket → pass tendency
        pass_score = 0.0
        pass_score += min(x_spread / 0.35, 1.0) * 0.5
        pass_score += min(y_depth / 0.5, 1.0) * 0.3
        # Fewer players visible (spread out, some off-screen) → pass
        if player_count < 15:
            pass_score += 0.2

        pass_score = float(np.clip(pass_score, 0.05, 0.95))
        run_score = 1.0 - pass_score
        return round(run_score, 3), round(pass_score, 3)

    def _formation_hint(self,
                        tracks: list,
                        frame_w: int,
                        frame_h: int) -> str:
        """Light offensive formation hint for the fan display."""
        if not tracks:
            return "Unknown"

        xs = [t.center[0] / frame_w for t in tracks]
        x_range = max(xs) - min(xs) if xs else 0.0
        wide = sum(1 for x in xs if x < 0.25 or x > 0.75)

        if x_range > 0.75 and wide >= 4:
            return "Shotgun Empty"
        if x_range > 0.65 and wide >= 3:
            return "Shotgun Spread"
        if x_range > 0.55:
            return "Shotgun"
        if x_range < 0.40:
            return "I-Formation"
        return "Pro Set"

    def _generate_alerts(self,
                          tracks: list,
                          frame_w: int,
                          frame_h: int) -> List[str]:
        """Generate real-time coaching alerts based on player geometry."""
        alerts: List[str] = []
        if not tracks:
            return alerts

        xs = [t.center[0] / frame_w for t in tracks]
        ys = [t.center[1] / frame_h for t in tracks]
        player_count = len(tracks)

        y_centroid = float(np.mean(ys))
        # Players clustered near line of scrimmage (within 10% of centroid y)
        near_los = sum(1 for y in ys if abs(y - y_centroid) < 0.10)

        # Blitz indicator: many players packed near LOS
        if near_los >= 8 and player_count >= 14:
            alerts.append("Possible blitz — heavy box")

        # Zone vs man: if defensive players spread evenly across width
        x_spread = float(np.std(xs)) if len(xs) > 1 else 0.0
        if x_spread > 0.28:
            alerts.append("Zone coverage detected")
        elif x_spread < 0.18 and player_count >= 12:
            alerts.append("Man coverage — tight alignment")

        # Nickel/dime indicator: many spread DBs
        wide_players = sum(1 for x in xs if x < 0.20 or x > 0.80)
        if wide_players >= 5:
            alerts.append("Dime package — 6 DBs")
        elif wide_players >= 4:
            alerts.append("Nickel package — 5 DBs")

        return alerts
