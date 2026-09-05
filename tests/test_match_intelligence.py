from src.match_intelligence import MatchIntelligence, estimate_formation


def player(pid, team, x, y, role="player"):
    return {"id": pid, "team": team, "x": x, "y": y, "role": role}


def balanced_players():
    return [player(i, 0, 15 + (i % 3) * 18, 8 + i * 5) for i in range(10)] + [
        player(20 + i, 1, 90 - (i % 3) * 18, 8 + i * 5) for i in range(10)
    ]


def test_possession_needs_confirmation_and_pass_is_same_team_switch():
    intel = MatchIntelligence()
    players = [player(1, 0, 20, 20), player(2, 0, 30, 20), player(3, 1, 70, 20)]
    first = intel.update(players, {"x": 20.5, "y": 20}, 1, 0.0)
    assert first["possession"] is None
    second = intel.update(players, {"x": 20.5, "y": 20}, 2, .25)
    assert second["possession"]["player_id"] == 1
    intel.update(players, {"x": 30, "y": 20}, 3, .5)
    switched = intel.update(players, {"x": 30, "y": 20}, 4, .75)
    assert switched["new_events"][-1]["type"] == "pass"
    assert switched["new_events"][-1]["detail"]["from_player_id"] == 1


def test_opponent_switch_is_turnover():
    intel = MatchIntelligence()
    players = [player(1, 0, 20, 20), player(3, 1, 70, 20)]
    intel.update(players, {"x": 20, "y": 20}, 1, 0)
    intel.update(players, {"x": 20, "y": 20}, 2, .25)
    intel.update(players, {"x": 70, "y": 20}, 3, .5)
    result = intel.update(players, {"x": 70, "y": 20}, 4, .75)
    assert result["new_events"][-1]["type"] == "turnover"
    assert result["possession"]["team"] == 1


def test_stale_ball_requires_extra_switch_evidence():
    intel = MatchIntelligence()
    players = [player(1, 0, 20, 20)]
    intel.update(players, {"x": 20, "y": 20, "stale": True}, 1, 0)
    intel.update(players, {"x": 20, "y": 20, "stale": True}, 2, .25)
    assert intel.owner_id is None
    intel.update(players, {"x": 20, "y": 20, "stale": True}, 3, .5)
    assert intel.owner_id == 1


def test_short_ball_gap_keeps_possession_without_creating_duplicate_start():
    intel = MatchIntelligence()
    players = [player(1, 0, 20, 20)]
    intel.update(players, {"x": 20, "y": 20}, 1, 0)
    intel.update(players, {"x": 20, "y": 20}, 2, .25)
    state = intel.update(players, None, 3, 4.5)
    assert state["possession"]["player_id"] == 1
    assert state["new_events"] == []


def test_new_video_period_clears_owner_but_keeps_event_sequence():
    intel = MatchIntelligence()
    players = [player(1, 0, 20, 20)]
    intel.update(players, {"x": 20, "y": 20}, 1, 1.0)
    first = intel.update(players, {"x": 20, "y": 20}, 2, 1.1)
    assert first["new_events"][0]["event_seq"] == 1
    intel.start_new_period()
    assert intel.owner_id is None
    intel.update(players, {"x": 20, "y": 20}, 3, 0.0)
    second = intel.update(players, {"x": 20, "y": 20}, 4, .1)
    assert second["new_events"][0]["event_seq"] == 2


def test_formation_estimate_counts_three_lines():
    players = []
    for i, x in enumerate([18] * 4 + [48] * 4 + [78] * 2):
        players.append(player(i, 0, x, 5 + i * 5))
    result = estimate_formation(players, 1)
    assert result["name"] == "4-4-2"
    assert result["confidence"] > .8


def test_direction_phase_and_shot_candidate_are_conservative():
    intel = MatchIntelligence()
    players = balanced_players()
    # Lock Team A toward +x, then confirm an owner near the final third.
    for frame in range(1, 5):
        intel.update(players, {"x": 72, "y": 34}, frame, frame * .25)
    players.append(player(99, 0, 72, 34))
    intel.update(players, {"x": 72, "y": 34}, 5, 1.25)
    state = intel.update(players, {"x": 72, "y": 34}, 6, 1.5)
    assert state["phase"]["label"] == "final-third attack"
    # Two fresh observations imply a fast ball toward the +x goal.
    intel.update(players, {"x": 78, "y": 34}, 7, 2.0)
    result = intel.update(players, {"x": 90, "y": 34}, 8, 2.5)
    assert any(e["type"] == "shot_candidate" for e in result["new_events"])
