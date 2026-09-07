from src.ball_tracking import BallMotionTracker


def test_ball_motion_predicts_through_short_gap():
    tracker = BallMotionTracker(max_age_frames=3, smoothing=1)
    tracker.observe((10, 20), 1)
    tracker.observe((12, 20), 2)
    predicted = tracker.predict(3)
    assert predicted["stale"] and predicted["predicted"]
    assert predicted["x"] > 12
    assert tracker.predict(6) is None


def test_ball_motion_rejects_impossible_teleport():
    tracker = BallMotionTracker(max_age_frames=4, smoothing=1)
    tracker.observe((10, 10), 1)
    result = tracker.observe((90, 60), 2)
    assert result["stale"]
    assert result["x"] < 20
