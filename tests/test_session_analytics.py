from src.session_analytics import build_session_analytics, compare_session_analytics


def snapshot(time_s, possession):
    return {
        "frame_id": int(time_s * 10),
        "time_s": time_s,
        "source_time_s": time_s,
        "players": [
            {"id": 1, "team": 0, "role": "player", "x": 80, "y": 20},
            {"id": 2, "team": 1, "role": "player", "x": 20, "y": 40},
        ],
        "concepts": {
            "control": {"0": 60, "1": 40},
            "teams": {
                "0": {
                    "compactness_m": 10, "width_m": 30, "depth_m": 20,
                    "line_height_m": 60,
                },
                "1": {
                    "compactness_m": 12, "width_m": 28, "depth_m": 18,
                    "line_height_m": 45,
                },
            },
        },
        "intelligence": {
            "possession": {"team": possession, "player_id": possession + 1},
            "directions": {"0": 1, "1": -1},
            "formations": {
                "0": {"name": "4-4-2"}, "1": {"name": "4-3-3"},
            },
        },
    }


def test_session_analytics_respect_rejected_events():
    events = [
        {
            "type": "pass", "team": 0, "review_status": "confirmed",
            "detail": {"from_player_id": 1, "to_player_id": 3},
        },
        {
            "type": "turnover", "team": 1, "review_status": "rejected",
            "detail": {},
        },
    ]
    result = build_session_analytics([snapshot(1, 0), snapshot(2, 1)], events)
    assert result["teams"]["0"]["possession_sample_pct"] == 50.0
    assert result["teams"]["0"]["passes"] == 1
    assert result["teams"]["1"]["turnovers_won"] == 0
    assert result["event_count"] == 1
    assert result["review_counts"]["rejected"] == 1
    assert result["pass_network"][0]["from_player_id"] == 1
    assert result["teams"]["0"]["final_third_presence_pct"] == 100.0


def test_comparison_returns_second_minus_first():
    first = build_session_analytics([snapshot(1, 0)], [])
    second = build_session_analytics([snapshot(1, 1)], [])
    result = compare_session_analytics(first, second)
    assert result["teams"]["0"]["possession_sample_pct"] == -100.0
    assert result["teams"]["1"]["possession_sample_pct"] == 100.0
