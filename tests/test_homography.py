"""
Tests for pitch homography + team assignment (the tactical-view spine).

Run:
    python -m tests.test_homography
    # or: pytest tests/test_homography.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.homography import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    PitchHomography,
)
from src.team_assignment import TEAM_UNKNOWN, TeamAssigner


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------
def _make_calibrated():
    """Calibrate from the four pitch corners projected into a fake frame."""
    pitch = [
        (0.0, 0.0),
        (PITCH_LENGTH, 0.0),
        (PITCH_LENGTH, PITCH_WIDTH),
        (0.0, PITCH_WIDTH),
    ]
    image = [(300.0, 650.0), (980.0, 650.0), (1180.0, 250.0), (100.0, 250.0)]
    homo = PitchHomography(frame_w=1280, frame_h=720)
    err = homo.calibrate(image, pitch)
    return homo, image, pitch, err


def test_calibration_is_exact_for_consistent_points():
    homo, image, pitch, err = _make_calibrated()
    assert homo.is_calibrated
    assert err < 1e-6, f"expected near-zero reprojection error, got {err}"
    # Each calibration pixel maps back to its pitch point.
    for (px, py), (tx, ty) in zip(image, pitch):
        x, y = homo.to_pitch(px, py)
        assert abs(x - tx) < 1e-4 and abs(y - ty) < 1e-4


def test_requires_four_points():
    homo = PitchHomography()
    try:
        homo.calibrate([(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 1), (2, 2)])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for < 4 correspondences")


def test_uncalibrated_fallback_maps_into_pitch():
    homo = PitchHomography(frame_w=1280, frame_h=720)
    assert not homo.is_calibrated
    # Frame centre should land roughly at pitch centre with the linear map.
    x, y = homo.to_pitch(640, 360)
    assert abs(x - PITCH_LENGTH / 2) < 1.0
    assert abs(y - PITCH_WIDTH / 2) < 1.0


def test_on_pitch_and_clamp():
    assert PitchHomography.on_pitch(52.5, 34.0)
    assert not PitchHomography.on_pitch(200.0, 34.0)
    cx, cy = PitchHomography.clamp_to_pitch(200.0, -5.0)
    assert cx == PITCH_LENGTH and cy == 0.0


def test_batch_matches_scalar():
    homo, image, _, _ = _make_calibrated()
    batch = homo.to_pitch_batch(image)
    for (px, py), (bx, by) in zip(image, batch):
        sx, sy = homo.to_pitch(px, py)
        assert abs(sx - bx) < 1e-9 and abs(sy - by) < 1e-9


def test_save_load_roundtrip(tmp_path=None):
    import tempfile

    homo, _, _, _ = _make_calibrated()
    d = tmp_path or tempfile.mkdtemp()
    path = os.path.join(str(d), "H.json")
    homo.save(path)
    loaded = PitchHomography.load(path)
    assert loaded.is_calibrated
    assert np.allclose(loaded.H, homo.H)


def test_calibrate_from_landmarks():
    homo = PitchHomography()
    corr = {
        "corner_bl": (300.0, 650.0),
        "corner_br": (980.0, 650.0),
        "corner_tr": (1180.0, 250.0),
        "corner_tl": (100.0, 250.0),
    }
    err = homo.calibrate_from_landmarks(corr)
    assert homo.is_calibrated and err < 1e-6


# ---------------------------------------------------------------------------
# Team assignment
# ---------------------------------------------------------------------------
class _T:
    def __init__(self, bbox):
        self.bbox = bbox
        self.team = TEAM_UNKNOWN


def test_two_teams_separate_cleanly():
    import cv2

    frame = np.full((720, 1280, 3), (40, 120, 40), dtype=np.uint8)
    tracks = []
    for i in range(6):  # red jerseys (BGR)
        x = 100 + i * 40
        cv2.rectangle(frame, (x, 300), (x + 24, 380), (0, 0, 200), -1)
        tracks.append(_T((x, 300, x + 24, 380)))
    for i in range(6):  # blue jerseys
        x = 700 + i * 40
        cv2.rectangle(frame, (x, 300), (x + 24, 380), (200, 0, 0), -1)
        tracks.append(_T((x, 300, x + 24, 380)))

    TeamAssigner(min_players=4).assign(tracks, frame)
    reds = {t.team for t in tracks[:6]}
    blues = {t.team for t in tracks[6:]}
    assert len(reds) == 1 and len(blues) == 1
    assert reds != blues


def test_assign_noop_without_frame():
    tracks = [_T((0, 0, 10, 20))]
    TeamAssigner().assign(tracks, None)
    assert tracks[0].team == TEAM_UNKNOWN


# ---------------------------------------------------------------------------
# Structured football concepts (tactical metrics)
# ---------------------------------------------------------------------------
def _two_blocks():
    from src.tactical_metrics import compute_metrics

    players = []
    for i in range(5):
        players.append({"id": i, "team": 0, "x": 30 + i, "y": 20 + i * 6})
    for i in range(5):
        players.append({"id": 10 + i, "team": 1, "x": 75 - i, "y": 20 + i * 6})
    return compute_metrics(players, PITCH_LENGTH, PITCH_WIDTH)


def test_control_sums_to_100():
    m = _two_blocks()
    assert abs(m["control"]["0"] + m["control"]["1"] - 100.0) < 0.2


def test_centroids_on_correct_halves():
    m = _two_blocks()
    assert m["teams"]["0"]["centroid"][0] < m["teams"]["1"]["centroid"][0]


def test_compactness_and_shape_fields():
    m = _two_blocks()
    a = m["teams"]["0"]
    assert a["n"] == 5 and a["compactness_m"] > 0
    assert a["width_m"] > 0 and a["depth_m"] > 0
    assert len(a["hull"]) >= 3            # a hull polygon for 5 spread points
    assert m["pressing_m"] is not None


def test_control_grid_shape():
    m = _two_blocks()
    g = m["control_grid"]
    assert len(g["cells"]) == g["cols"] * g["rows"]
    assert set(g["cells"]) <= {0, 1}


def test_metrics_handle_one_team_only():
    from src.tactical_metrics import compute_metrics

    players = [{"id": i, "team": 0, "x": 40 + i, "y": 30} for i in range(4)]
    m = compute_metrics(players, PITCH_LENGTH, PITCH_WIDTH)
    assert m["control"]["0"] == 100.0 and m["control"]["1"] == 0.0
    assert m["teams"]["1"] is None
    assert m["pressing_m"] is None       # needs both teams


def test_metrics_empty_is_safe():
    from src.tactical_metrics import compute_metrics

    m = compute_metrics([], PITCH_LENGTH, PITCH_WIDTH)
    assert m["teams"]["0"] is None and m["teams"]["1"] is None
    assert m["control"] == {"0": 0.0, "1": 0.0}


# ---------------------------------------------------------------------------
# Game-state embedding + situation retrieval
# ---------------------------------------------------------------------------
def _block(ax, bx, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    players = []
    for i in range(11):
        players.append({"team": 0, "x": ax + rng.normal(0, jitter), "y": 8 + i * 5})
        players.append({"team": 1, "x": bx + rng.normal(0, jitter), "y": 8 + i * 5})
    return players


def test_embedding_is_unit_norm_and_fixed_dim():
    from src.state_embedding import StateEmbedder

    emb = StateEmbedder()
    v = emb.embed(_block(30, 70))
    assert v.shape == (emb.dim,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_embedding_permutation_invariant():
    from src.state_embedding import StateEmbedder

    emb = StateEmbedder()
    players = _block(30, 70, jitter=2.0, seed=1)
    v1 = emb.embed(players)
    shuffled = list(reversed(players))
    v2 = emb.embed(shuffled)
    assert float(np.linalg.norm(v1 - v2)) < 1e-6  # ordering must not matter


def test_similar_states_more_similar_than_dissimilar():
    from src.state_embedding import StateEmbedder

    emb = StateEmbedder()
    low = emb.embed(_block(25, 55, seed=2))
    low2 = emb.embed(_block(27, 57, seed=3))
    high = emb.embed(_block(75, 95, seed=4))
    assert float(low @ low2) > float(low @ high)


def test_store_retrieves_correct_family():
    from src.state_embedding import StateEmbedder, SituationStore

    emb = StateEmbedder()
    store = SituationStore(dim=emb.dim)
    for t in range(8):
        store.add(emb.embed(_block(25, 55, 1.5, seed=t)),
                  {"frame_id": t, "t": float(t), "family": "low"})
    for t in range(8, 16):
        store.add(emb.embed(_block(72, 95, 1.5, seed=t)),
                  {"frame_id": t, "t": float(t), "family": "high"})

    hits = store.query(emb.embed(_block(26, 56, 0.5, seed=99)), k=3)
    assert len(hits) == 3
    assert all(h["meta"]["family"] == "low" for h in hits)
    assert hits[0]["similarity"] >= hits[-1]["similarity"]  # sorted desc


def test_store_exclusion_window():
    from src.state_embedding import StateEmbedder, SituationStore

    emb = StateEmbedder()
    store = SituationStore(dim=emb.dim)
    for t in range(5):
        store.add(emb.embed(_block(25, 55, seed=t)), {"frame_id": t, "t": float(t)})
    # Exclude everything within 100s of t=2 -> no results survive.
    hits = store.query(emb.embed(_block(25, 55)), k=3,
                       exclude_within_s=100.0, query_time_s=2.0)
    assert hits == []


def test_store_capacity_drops_oldest():
    from src.state_embedding import StateEmbedder, SituationStore

    emb = StateEmbedder()
    store = SituationStore(dim=emb.dim, capacity=5)
    for t in range(8):
        store.add(emb.embed(_block(25, 55, seed=t)), {"frame_id": t, "t": float(t)})
    assert len(store) == 5


def test_store_save_load_roundtrip():
    import tempfile
    from src.state_embedding import StateEmbedder, SituationStore

    emb = StateEmbedder()
    store = SituationStore(dim=emb.dim)
    for t in range(4):
        store.add(emb.embed(_block(25, 55, seed=t)), {"frame_id": t, "t": float(t)})
    d = tempfile.mkdtemp()
    path = os.path.join(d, "situations")
    store.save(path)
    loaded = SituationStore.load(path)
    assert len(loaded) == 4
    hits = loaded.query(emb.embed(_block(25, 55)), k=1)
    assert hits and hits[0]["meta"]["frame_id"] in {0, 1, 2, 3}


if __name__ == "__main__":
    passed = 0
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")
