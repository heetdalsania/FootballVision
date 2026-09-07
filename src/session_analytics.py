"""Deterministic, local post-match analytics from persisted pitch states."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def _round(value, digits=1):
    return round(float(value), digits) if value is not None else None


def _average(values: Iterable[float]):
    values = [float(value) for value in values if value is not None]
    return _round(sum(values) / len(values)) if values else None


def build_session_analytics(snapshots: list[dict], events: list[dict]) -> dict:
    """Build JSON-friendly analytics without external services or models."""
    usable_events = [
        event for event in events if event.get("review_status") != "rejected"
    ]
    review_counts = Counter(
        event.get("review_status") or "inferred" for event in events
    )
    event_counts = Counter(event.get("type") or "unknown" for event in usable_events)
    possession_samples = Counter()
    formations = {0: Counter(), 1: Counter()}
    team_metrics = {team: defaultdict(list) for team in (0, 1)}
    final_third = Counter()
    player_samples = Counter()
    heatmaps = {
        0: [[0 for _ in range(12)] for _ in range(8)],
        1: [[0 for _ in range(12)] for _ in range(8)],
    }
    momentum = []

    for snap in snapshots:
        intelligence = snap.get("intelligence") or {}
        possession = intelligence.get("possession") or {}
        owner_team = possession.get("team")
        if owner_team in (0, 1):
            possession_samples[int(owner_team)] += 1

        directions = intelligence.get("directions") or {"0": 1, "1": -1}
        for team in (0, 1):
            formation = (intelligence.get("formations") or {}).get(str(team)) or {}
            name = formation.get("name")
            if name and name != "insufficient data":
                formations[team][name] += 1

        concepts = snap.get("concepts") or {}
        control = concepts.get("control") or {}
        for team in (0, 1):
            shape = (concepts.get("teams") or {}).get(str(team)) or {}
            for key in ("compactness_m", "width_m", "depth_m", "line_height_m"):
                if shape.get(key) is not None:
                    team_metrics[team][key].append(shape[key])
            if control.get(str(team)) is not None:
                team_metrics[team]["control_pct"].append(control[str(team)])

        for player in snap.get("players") or []:
            team = player.get("team")
            if team not in (0, 1) or player.get("role") != "player":
                continue
            team = int(team)
            x = min(PITCH_LENGTH, max(0.0, float(player.get("x") or 0)))
            y = min(PITCH_WIDTH, max(0.0, float(player.get("y") or 0)))
            player_samples[team] += 1
            direction = directions.get(str(team), 1 if team == 0 else -1)
            progress = x if direction > 0 else PITCH_LENGTH - x
            final_third[team] += int(progress >= 70.0)
            col = min(11, int(12 * x / PITCH_LENGTH))
            row = min(7, int(8 * y / PITCH_WIDTH))
            heatmaps[team][row][col] += 1

        if control:
            a = float(control.get("0") or 0)
            b = float(control.get("1") or 0)
            possession_bias = 8 if owner_team == 0 else -8 if owner_team == 1 else 0
            shown_time = (
                snap.get("source_time_s")
                if snap.get("source_time_s") is not None
                else snap.get("time_s")
            )
            momentum.append({
                "time_s": _round(shown_time, 2),
                "team_a_dominance": _round(
                    max(-100, min(100, a - b + possession_bias))
                ),
            })

    total_possession = sum(possession_samples.values())
    pass_edges = defaultdict(int)
    for event in usable_events:
        if event.get("type") != "pass":
            continue
        detail = event.get("detail") or {}
        source, target = detail.get("from_player_id"), detail.get("to_player_id")
        if source is not None and target is not None:
            pass_edges[(event.get("team"), source, target)] += 1

    duration_values = [
        snap.get("source_time_s")
        if snap.get("source_time_s") is not None
        else snap.get("time_s")
        for snap in snapshots
    ]
    duration_s = max(
        (float(value) for value in duration_values if value is not None),
        default=0,
    )
    teams = {}
    for team in (0, 1):
        metrics = team_metrics[team]
        teams[str(team)] = {
            "possession_sample_pct": (
                _round(100 * possession_samples[team] / total_possession)
                if total_possession else None
            ),
            "avg_control_pct": _average(metrics["control_pct"]),
            "avg_compactness_m": _average(metrics["compactness_m"]),
            "avg_width_m": _average(metrics["width_m"]),
            "avg_depth_m": _average(metrics["depth_m"]),
            "avg_line_height_m": _average(metrics["line_height_m"]),
            "final_third_presence_pct": (
                _round(100 * final_third[team] / player_samples[team])
                if player_samples[team] else None
            ),
            "passes": sum(
                count
                for (edge_team, _source, _target), count in pass_edges.items()
                if edge_team == team
            ),
            "turnovers_won": sum(
                1 for event in usable_events
                if event.get("type") == "turnover" and event.get("team") == team
            ),
            "top_formations": [
                {"name": name, "samples": count}
                for name, count in formations[team].most_common(3)
            ],
            "heatmap": heatmaps[team],
        }

    return {
        "duration_s": _round(duration_s, 2),
        "snapshot_count": len(snapshots),
        "event_count": len(usable_events),
        "event_counts": dict(sorted(event_counts.items())),
        "review_counts": dict(sorted(review_counts.items())),
        "teams": teams,
        "pass_network": [
            {
                "team": team,
                "from_player_id": source,
                "to_player_id": target,
                "passes": count,
            }
            for (team, source, target), count in sorted(
                pass_edges.items(), key=lambda item: (-item[1], str(item[0]))
            )
        ],
        "momentum": momentum,
    }


def compare_session_analytics(first: dict, second: dict) -> dict:
    """Return second-minus-first deltas for comparable team metrics."""
    metric_keys = (
        "possession_sample_pct", "avg_control_pct", "avg_compactness_m",
        "avg_width_m", "avg_depth_m", "avg_line_height_m",
        "final_third_presence_pct", "passes", "turnovers_won",
    )
    deltas = {}
    for team in ("0", "1"):
        left = (first.get("teams") or {}).get(team) or {}
        right = (second.get("teams") or {}).get(team) or {}
        deltas[team] = {}
        for key in metric_keys:
            if left.get(key) is None or right.get(key) is None:
                deltas[team][key] = None
            else:
                deltas[team][key] = _round(right[key] - left[key])
    return {"teams": deltas}
