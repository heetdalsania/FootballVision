from src.grounded_ask import answer, build_facts_block


PITCH = {
    "calibrated": True,
    "n_keypoints": 8,
    "counts": {"player": 2},
    "team_counts": {"0": 1, "1": 1},
    "players": [
        {"id": 1, "role": "player", "team": 0, "x": 30, "y": 20},
        {"id": 2, "role": "player", "team": 1, "x": 70, "y": 40},
    ],
    "concepts": {
        "teams": {
            "0": {"n": 1, "centroid": [30, 20], "compactness_m": 8,
                  "width_m": 20, "depth_m": 15, "line_height_m": 45},
            "1": {"n": 1, "centroid": [70, 40], "compactness_m": 12,
                  "width_m": 30, "depth_m": 18, "line_height_m": 65},
        },
        "control": {"0": 55, "1": 45},
        "pressing_m": 5.5,
    },
    "intelligence": {
        "possession": {"team": 0, "player_id": 1, "confidence": .8},
        "directions": {"0": 1, "1": -1},
        "phase": {"label": "progression", "zone": "middle third", "team": 0},
        "formations": {
            "0": {"name": "4-4-2", "confidence": .75},
            "1": {"name": "4-3-3", "confidence": .7},
        },
        "events": [{"label": "Pass", "time_s": 4, "confidence": .8}],
    },
}


def test_facts_include_temporal_intelligence():
    facts, ok = build_facts_block(PITCH)
    assert ok
    assert "possession: team A" in facts
    assert "estimated formation team A: 4-4-2" in facts


def test_local_rules_answer_without_any_model():
    result = answer("Who has possession?", PITCH, backend="rules")
    assert result["ok"] and result["backend"] == "local rules"
    assert "Team A" in result["answer"] and "80%" in result["answer"]


def test_local_rules_compare_metrics_and_refuse_unmeasured_claims():
    assert "Team A" in answer("Which team is more compact?", PITCH, "rules")["answer"]
    refusal = answer("Who scored the opening goal?", PITCH, "rules")["answer"]
    assert "does not cover" in refusal


def test_auto_uses_deterministic_possession_answer_before_a_model():
    result = answer("Who has possession?", PITCH, backend="auto")
    assert result["backend"] == "local rules"
    assert "Team A" in result["answer"]
    assert "player 1" in result["answer"]
