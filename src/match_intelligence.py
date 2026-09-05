"""Temporal match intelligence derived from projected players and ball states.

The detector answers *where objects are*. This module adds conservative match
semantics while keeping uncertainty visible: possession needs repeated spatial
evidence, owner changes use hysteresis, and visually inferred shots are labelled
as candidates rather than claimed as facts.
"""

from __future__ import annotations

from collections import Counter, deque
from math import hypot
from typing import Dict, Iterable, Optional

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
POSSESSION_RADIUS_M = 3.5
RELEASE_RADIUS_M = 5.5
SWITCH_CONFIRM_FRAMES = 2
# Ball detections are deliberately sparse (the dedicated pass runs every few
# frames). Retain the last owner through a short evidence gap; a newly
# confirmed opponent still switches immediately, so this does not delay real
# turnovers—it only prevents repeated "possession start" noise.
OWNER_HOLD_S = 5.0
CARRY_DISTANCE_M = 8.0
SHOT_SPEED_MPS = 7.0


class MatchIntelligence:
    """Stateful possession, event, direction, phase, and formation estimator."""

    def __init__(self, max_events: int = 200):
        self.max_events = max_events
        self.reset()

    def reset(self) -> None:
        self._events: deque[dict] = deque(maxlen=self.max_events)
        self._event_seq = 0
        self._direction_score = 0
        self._directions = {0: 1, 1: -1}
        self._formation_history = {0: deque(maxlen=12), 1: deque(maxlen=12)}
        self.start_new_period()

    def start_new_period(self) -> None:
        """Clear transient evidence without renumbering persisted events.

        Video-file analysis can loop at EOF. The source timestamp then jumps
        backwards, so ownership and ball-velocity evidence must not bridge the
        end of the file and its first frame.
        """
        self.owner_id: Optional[int] = None
        self.owner_team: Optional[int] = None
        self.owner_confidence = 0.0
        self.owner_last_seen_s = 0.0
        self._candidate: Optional[tuple[int, int]] = None
        self._candidate_count = 0
        self._candidate_confidence = 0.0
        self._last_fresh_ball: Optional[tuple[float, float, float]] = None
        self._last_fresh_ball_seen_s: Optional[float] = None
        self._carry_origin: Optional[tuple[float, float, float]] = None
        self._last_shot_s = -999.0

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def update(
        self,
        players: Iterable[dict],
        ball: Optional[dict],
        frame_id: int,
        time_s: float,
        source_time_s: Optional[float] = None,
    ) -> dict:
        players = [p for p in players if p.get("team") in (0, 1)]
        time_s = float(time_s)
        self._update_directions(players)
        formations = self._update_formations(players)
        new_events: list[dict] = []

        fresh_ball = bool(ball and not ball.get("stale"))
        ball_xy = ((float(ball["x"]), float(ball["y"])) if ball else None)
        nearest = self._nearest_owner(players, ball_xy)

        if fresh_ball and ball_xy is not None:
            if (self._last_fresh_ball_seen_s is not None
                    and time_s - self._last_fresh_ball_seen_s >= 2.5
                    and self._near_restart_boundary(ball_xy)):
                new_events.append(self._emit(
                    "restart", "Restart", frame_id, time_s, source_time_s,
                    self.owner_team, self.owner_id, .62, ball_xy,
                    {"evidence": "ball reappeared near a pitch boundary"},
                ))
            shot = self._shot_candidate(ball_xy, frame_id, time_s, source_time_s)
            if shot:
                new_events.append(shot)
            self._last_fresh_ball_seen_s = time_s

        accepted = self._update_owner_candidate(nearest, ball, time_s)
        if accepted is not None:
            new_events.extend(self._accept_owner(
                accepted, ball_xy, frame_id, time_s, source_time_s
            ))
        elif (self.owner_id is not None
              and time_s - self.owner_last_seen_s > OWNER_HOLD_S):
            self.owner_id = self.owner_team = None
            self.owner_confidence = 0.0
            self._carry_origin = None

        carry = self._carry_event(ball_xy, frame_id, time_s, source_time_s)
        if carry:
            new_events.append(carry)

        possession = None
        if self.owner_team in (0, 1):
            possession = {
                "team": self.owner_team,
                "player_id": self.owner_id,
                "confidence": round(self.owner_confidence, 2),
            }
        phase = self._phase(possession, ball_xy)
        return {
            "possession": possession,
            "directions": {str(k): v for k, v in self._directions.items()},
            "formations": formations,
            "phase": phase,
            "new_events": new_events,
            "events": self.events[-50:],
        }

    def _nearest_owner(self, players: list[dict], ball_xy):
        if ball_xy is None:
            return None
        choices = []
        for p in players:
            if p.get("role") not in ("player", "goalkeeper"):
                continue
            distance = hypot(float(p["x"]) - ball_xy[0], float(p["y"]) - ball_xy[1])
            choices.append((distance, int(p.get("id", -1)), int(p["team"])))
        if not choices:
            return None
        distance, player_id, team = min(choices)
        radius = RELEASE_RADIUS_M if player_id == self.owner_id else POSSESSION_RADIUS_M
        if distance > radius:
            return None
        confidence = max(.05, min(.99, 1.0 - distance / RELEASE_RADIUS_M))
        return player_id, team, confidence, distance

    def _update_owner_candidate(self, nearest, ball, time_s):
        if nearest is None:
            self._candidate = None
            self._candidate_count = 0
            return None
        player_id, team, confidence, _distance = nearest
        key = (player_id, team)
        if key == (self.owner_id, self.owner_team):
            self.owner_last_seen_s = time_s
            self.owner_confidence = .65 * self.owner_confidence + .35 * confidence
            self._candidate = None
            self._candidate_count = 0
            return None
        if key == self._candidate:
            self._candidate_count += 1
            self._candidate_confidence = (
                .5 * self._candidate_confidence + .5 * confidence
            )
        else:
            self._candidate = key
            self._candidate_count = 1
            self._candidate_confidence = confidence
        # A carried/held ball is useful but weaker evidence than a fresh hit.
        needed = SWITCH_CONFIRM_FRAMES + (1 if ball and ball.get("stale") else 0)
        return (player_id, team, self._candidate_confidence) if self._candidate_count >= needed else None

    def _accept_owner(self, accepted, ball_xy, frame_id, time_s, source_time_s):
        player_id, team, confidence = accepted
        previous_id, previous_team = self.owner_id, self.owner_team
        self.owner_id, self.owner_team = player_id, team
        self.owner_confidence = confidence
        self.owner_last_seen_s = time_s
        self._candidate = None
        self._candidate_count = 0
        self._carry_origin = ((*ball_xy, time_s) if ball_xy else None)

        if previous_id is None:
            return [self._emit(
                "possession_start", "Possession", frame_id, time_s, source_time_s,
                team, player_id, confidence, ball_xy,
                {"evidence": "nearest player confirmed across frames"},
            )]
        if previous_team == team:
            return [self._emit(
                "pass", "Pass", frame_id, time_s, source_time_s,
                team, player_id, confidence, ball_xy,
                {"from_player_id": previous_id, "to_player_id": player_id},
            )]
        return [self._emit(
            "turnover", "Turnover", frame_id, time_s, source_time_s,
            team, player_id, confidence, ball_xy,
            {"won_from_team": previous_team, "from_player_id": previous_id},
        )]

    def _carry_event(self, ball_xy, frame_id, time_s, source_time_s):
        if self.owner_id is None or ball_xy is None:
            return None
        if self._carry_origin is None:
            self._carry_origin = (*ball_xy, time_s)
            return None
        ox, oy, started = self._carry_origin
        distance = hypot(ball_xy[0] - ox, ball_xy[1] - oy)
        if distance < CARRY_DISTANCE_M or time_s - started < 1.0:
            return None
        self._carry_origin = (*ball_xy, time_s)
        return self._emit(
            "carry", "Carry", frame_id, time_s, source_time_s,
            self.owner_team, self.owner_id, min(.9, .5 + distance / 30), ball_xy,
            {"distance_m": round(distance, 1)},
        )

    def _shot_candidate(self, ball_xy, frame_id, time_s, source_time_s):
        previous = self._last_fresh_ball
        self._last_fresh_ball = (ball_xy[0], ball_xy[1], time_s)
        if previous is None or self.owner_team not in (0, 1):
            return None
        dt = time_s - previous[2]
        if dt <= 0 or dt > 2.5 or time_s - self._last_shot_s < 3.0:
            return None
        vx = (ball_xy[0] - previous[0]) / dt
        direction = self._directions[self.owner_team]
        final_zone = ball_xy[0] >= 88 if direction > 0 else ball_xy[0] <= 17
        toward_goal = vx * direction >= SHOT_SPEED_MPS
        central = abs(ball_xy[1] - PITCH_WIDTH / 2) <= 24
        if not (final_zone and toward_goal and central):
            return None
        self._last_shot_s = time_s
        confidence = min(.92, .55 + abs(vx) / 50)
        return self._emit(
            "shot_candidate", "Shot candidate", frame_id, time_s, source_time_s,
            self.owner_team, self.owner_id, confidence, ball_xy,
            {"ball_speed_toward_goal_mps": round(abs(vx), 1)},
        )

    def _update_directions(self, players: list[dict]) -> None:
        xs = {team: [float(p["x"]) for p in players if p["team"] == team
                     and p.get("role") == "player"] for team in (0, 1)}
        if len(xs[0]) < 3 or len(xs[1]) < 3:
            return
        vote = 1 if np.median(xs[0]) < np.median(xs[1]) else -1
        self._direction_score = max(-20, min(20, self._direction_score + vote))
        if abs(self._direction_score) >= 3:
            self._directions[0] = 1 if self._direction_score > 0 else -1
            self._directions[1] = -self._directions[0]

    def _update_formations(self, players: list[dict]) -> dict:
        result = {}
        for team in (0, 1):
            team_players = [p for p in players if p["team"] == team
                            and p.get("role") == "player"]
            estimate = estimate_formation(team_players, self._directions[team])
            if estimate["name"] != "insufficient data":
                self._formation_history[team].append(estimate["name"])
                stable, votes = Counter(self._formation_history[team]).most_common(1)[0]
                estimate["name"] = stable
                estimate["stability"] = round(votes / len(self._formation_history[team]), 2)
            result[str(team)] = estimate
        return result

    def _phase(self, possession, ball_xy) -> dict:
        if not possession or ball_xy is None:
            return {"label": "transition", "team": None, "zone": None}
        team = possession["team"]
        progress = ball_xy[0] if self._directions[team] > 0 else PITCH_LENGTH - ball_xy[0]
        if progress < 35:
            label, zone = "build-up", "defensive third"
        elif progress < 70:
            label, zone = "progression", "middle third"
        else:
            label, zone = "final-third attack", "attacking third"
        return {"label": label, "team": team, "zone": zone}

    @staticmethod
    def _near_restart_boundary(ball_xy) -> bool:
        x, y = ball_xy
        return x < 8 or x > PITCH_LENGTH - 8 or y < 6 or y > PITCH_WIDTH - 6

    def _emit(self, event_type, label, frame_id, time_s, source_time_s,
              team, player_id, confidence, ball_xy, detail):
        self._event_seq += 1
        event = {
            "event_seq": self._event_seq,
            "type": event_type,
            "label": label,
            "frame_id": int(frame_id),
            "time_s": round(float(time_s), 2),
            "source_time_s": (round(float(source_time_s), 3)
                              if source_time_s is not None else None),
            "team": team,
            "player_id": player_id,
            "confidence": round(float(confidence), 2),
            "x": round(float(ball_xy[0]), 2) if ball_xy else None,
            "y": round(float(ball_xy[1]), 2) if ball_xy else None,
            "detail": detail or {},
        }
        self._events.append(event)
        return event


def estimate_formation(players: list[dict], attack_direction: int) -> dict:
    """Estimate three outfield lines with deterministic 1-D k-means."""
    if len(players) < 6:
        return {"name": "insufficient data", "confidence": 0.0, "players": len(players)}
    progress = np.asarray([
        float(p["x"]) if attack_direction > 0 else PITCH_LENGTH - float(p["x"])
        for p in players
    ])
    centers = np.quantile(progress, [.2, .55, .85])
    labels = np.zeros(len(progress), dtype=int)
    for _ in range(10):
        labels = np.argmin(abs(progress[:, None] - centers[None, :]), axis=1)
        updated = np.asarray([
            progress[labels == i].mean() if np.any(labels == i) else centers[i]
            for i in range(3)
        ])
        if np.allclose(updated, centers):
            break
        centers = updated
    order = np.argsort(centers)
    counts = [int(np.sum(labels == idx)) for idx in order]
    separation = float(np.diff(np.sort(centers)).min()) if len(set(labels)) == 3 else 0.0
    confidence = min(1.0, len(players) / 10.0) * min(1.0, separation / 12.0)
    return {
        "name": "-".join(str(n) for n in counts),
        "confidence": round(confidence, 2),
        "players": len(players),
        "lines": counts,
    }
