import asyncio
from io import BytesIO

import numpy as np

from db import session as db_session
from src.fv_engine import smooth_point
from src.session_report import snapshots_to_frames
from src.team_assignment import CropTeamClassifier
from scripts.golden_video_benchmark import evaluate
from starlette.datastructures import UploadFile
from ui import api


def test_smooth_point_reduces_jitter():
    assert smooth_point(None, (10, 20), .4) == (10.0, 20.0)
    assert smooth_point((10, 20), (20, 30), .4) == (14.0, 24.0)


def test_local_crop_classifier_separates_kits():
    red = np.full((80, 30, 3), (0, 0, 220), dtype=np.uint8)
    blue = np.full((80, 30, 3), (220, 0, 0), dtype=np.uint8)
    classifier = CropTeamClassifier().fit([red] * 6 + [blue] * 6)
    prediction = classifier.predict([red, blue])
    assert prediction[0] in (0, 1)
    assert prediction[1] in (0, 1)
    assert prediction[0] != prediction[1]


def test_snapshot_report_conversion():
    tracking, metrics = snapshots_to_frames([{
        "frame_id": 15,
        "time_s": 2.5,
        "players": [{"id": 7, "team": 0, "role": "player", "x": 20, "y": 30}],
        "concepts": {"control": {"0": 55}, "pressing_m": 7,
                     "teams": {"0": {"n": 1, "compactness_m": 0,
                                        "width_m": 0, "depth_m": 0,
                                        "line_height_m": 20}}},
    }])
    assert tracking.iloc[0].to_dict()["track_id"] == 7
    assert metrics.iloc[0].to_dict()["control_pct"] == 55


def test_golden_threshold_evaluation():
    assert evaluate({"rate": .9}, {"rate": {"min": .8, "max": 1}}) == []
    assert "minimum" in evaluate({"rate": .4}, {"rate": {"min": .8}})[0]


def test_tactical_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(db_session, "DB_PATH", tmp_path / "test.db")

    async def run():
        await db_session.init_db()
        sid = await db_session.create_tactical_session("sample.mp4", "video", "data/sample.mp4")
        await db_session.save_tactical_snapshot(sid, {
            "frame_id": 15, "elapsed_s": 2.5,
            "pitch": {"players": [{"id": 1}], "ball": {"x": 2},
                      "concepts": None, "counts": {"player": 1}},
        })
        await db_session.finish_tactical_session(sid, 15, 2.5)
        return sid, await db_session.get_tactical_session(sid), await db_session.get_tactical_snapshots(sid)

    sid, session, snapshots = asyncio.run(run())
    assert sid > 0 and session["status"] == "complete"
    assert snapshots[0]["players"] == [{"id": 1}]


def test_local_upload_sanitises_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    upload = UploadFile(filename="../../my match.mp4", file=BytesIO(b"video-bytes"))
    response = asyncio.run(api.tactical_upload(upload))
    data = __import__("json").loads(response.body)
    assert response.status_code == 200
    assert data["video"]["path"] == "uploads/my-match.mp4"
    assert (tmp_path / data["video"]["path"]).read_bytes() == b"video-bytes"
