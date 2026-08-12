"""Tests for the static analyst report.

Only the rendering half is covered here. Extraction needs the vision models
and a video, so it is exercised by actually running
`scripts/analyst_report.py` rather than in the unit suite.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.match_report import (  # noqa: E402
    PITCH_LENGTH,
    PITCH_WIDTH,
    _hull,
    _teams_present,
    render,
)


def _tracking(n_frames=40, tracks_per_team=11, seed=0):
    """Synthetic tracking data: two teams, each clustered on its own half."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        for team in (0, 1):
            base_x = 35.0 if team == 0 else 70.0
            for t in range(tracks_per_team):
                rows.append({
                    "frame_id": f,
                    "time_s": f * 0.12,
                    "track_id": team * 100 + t,
                    "team": team,
                    "role": "player",
                    "x": float(np.clip(base_x + rng.normal(0, 6), 0, PITCH_LENGTH)),
                    "y": float(np.clip(10 + t * 4.5 + rng.normal(0, 2), 0, PITCH_WIDTH)),
                })
        # A referee, which must never be counted as a team player.
        rows.append({"frame_id": f, "time_s": f * 0.12, "track_id": 900,
                     "team": -1, "role": "referee", "x": 52.0, "y": 34.0})
    return pd.DataFrame(rows)


def _metrics(n_frames=40):
    rows = []
    for f in range(n_frames):
        for team in (0, 1):
            rows.append({
                "frame_id": f,
                "time_s": f * 0.12,
                "team": team,
                "n": 11,
                "compactness_m": 13.0 + team * 3.0,
                "width_m": 40.0,
                "depth_m": 30.0,
                "line_height_m": 45.0 + team * 9.0,
                "control_pct": 45.0 + team * 10.0,
                "pressing_m": 6.5,
            })
    return pd.DataFrame(rows)


def test_teams_present_ignores_referees_and_unknown():
    tracking = _tracking()
    assert _teams_present(tracking) == [0, 1]


def test_teams_present_excludes_unassigned_team():
    tracking = _tracking()
    tracking.loc[tracking.index[:20], "team"] = -1
    # -1 is "team not resolved" and must never become a series in the report.
    assert -1 not in _teams_present(tracking)


def test_hull_of_a_square_has_four_vertices():
    pts = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0],
                    [5.0, 5.0]])  # interior point must be dropped
    hull = _hull(pts)
    assert hull is not None
    assert len(hull) == 4


def test_render_writes_a_png():
    tracking, metrics = _tracking(), _metrics()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "report.png")
        path = render(tracking, metrics, out, title="Test", subtitle="synthetic")
        assert path == out
        assert os.path.exists(out)
        # A real multi-panel figure is not a few hundred bytes.
        assert os.path.getsize(out) > 50_000


def test_render_raises_when_no_team_is_resolved():
    tracking = _tracking()
    tracking["team"] = -1
    with pytest.raises(ValueError, match="no outfield players"):
        render(tracking, _metrics(), "unused.png")


def test_render_survives_more_track_ids_than_players():
    """ByteTrack re-issues ids, so a clip yields more tracks than players."""
    tracking = _tracking(tracks_per_team=26)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "report.png")
        render(tracking, _metrics(), out)
        assert os.path.exists(out)


def test_render_handles_a_team_missing_from_metrics():
    """One team can drop out of the metrics when too few players are on frame."""
    metrics = _metrics()
    metrics = metrics[metrics["team"] == 0]
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "report.png")
        render(_tracking(), metrics, out)
        assert os.path.exists(out)
