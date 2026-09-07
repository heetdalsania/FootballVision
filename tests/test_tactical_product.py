import asyncio
from io import BytesIO

import numpy as np

from db import session as db_session
from src.fv_engine import FootballEngine, smooth_point
from src.event_clip import build_event_clip
from src.annotated_video import build_annotated_video
from src.session_report import snapshots_to_frames
from src.team_assignment import CropTeamClassifier
from scripts.golden_video_benchmark import evaluate
from starlette.datastructures import UploadFile
from ui import api


def test_smooth_point_reduces_jitter():
    assert smooth_point(None, (10, 20), .4) == (10.0, 20.0)
    assert smooth_point((10, 20), (20, 30), .4) == (14.0, 24.0)


def test_manual_four_corner_calibration():
    engine = FootballEngine(enable_team_classifier=False)
    result = engine.set_manual_calibration([
        {"x": .1, "y": .9}, {"x": .9, "y": .9},
        {"x": .8, "y": .2}, {"x": .2, "y": .2},
    ], (1000, 600))
    assert result["ok"] and result["mode"] == "manual"
    assert result["reprojection_error_m"] < .001
    assert engine._manual_calibration


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
            "frame_id": 15, "elapsed_s": 2.5, "source_time_s": 1.25,
            "pitch": {"players": [{"id": 1}], "ball": {"x": 2},
                      "concepts": None, "counts": {"player": 1},
                      "intelligence": {"possession": {"team": 0, "player_id": 1}}},
        })
        await db_session.save_tactical_events(sid, [{
            "event_seq": 1, "frame_id": 12, "time_s": 2.0,
            "source_time_s": 1.5, "type": "pass", "label": "Pass",
            "team": 0, "player_id": 2, "confidence": .8,
            "x": 20, "y": 30, "detail": {"from_player_id": 1},
        }])
        stored_events = await db_session.get_tactical_events(sid)
        await db_session.update_tactical_event(sid, stored_events[0]["id"], {
            "label": "Reviewed pass", "review_status": "confirmed",
            "notes": "Checked against the source video",
        })
        await db_session.set_tactical_player_label(
            sid, 1, "Alex", 9, 0, "Captain"
        )
        await db_session.finish_tactical_session(sid, 15, 2.5)
        return (sid, await db_session.get_tactical_session(sid),
                await db_session.get_tactical_snapshots(sid),
                await db_session.get_tactical_events(sid),
                await db_session.get_tactical_player_labels(sid))

    sid, session, snapshots, events, player_labels = asyncio.run(run())
    assert sid > 0 and session["status"] == "complete"
    assert snapshots[0]["players"] == [{"id": 1}]
    assert snapshots[0]["source_time_s"] == 1.25
    assert snapshots[0]["intelligence"]["possession"]["player_id"] == 1
    assert events[0]["type"] == "pass"
    assert events[0]["label"] == "Reviewed pass"
    assert events[0]["review_status"] == "confirmed"
    assert events[0]["notes"] == "Checked against the source video"
    assert events[0]["detail"]["from_player_id"] == 1
    assert player_labels[0]["display_name"] == "Alex"
    assert player_labels[0]["shirt_number"] == 9


def test_local_upload_sanitises_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    upload = UploadFile(filename="../../my match.mp4", file=BytesIO(b"video-bytes"))
    response = asyncio.run(api.tactical_upload(upload))
    data = __import__("json").loads(response.body)
    assert response.status_code == 200
    assert data["video"]["path"] == "uploads/my-match.mp4"
    assert (tmp_path / data["video"]["path"]).read_bytes() == b"video-bytes"


def test_event_clip_export(tmp_path):
    import cv2

    source, output = tmp_path / "source.mp4", tmp_path / "event.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
    )
    for value in range(40):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    assert build_event_clip(source, output, 2.0, 1.0, 1.0) == str(output)
    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 15
    cap.release()


def test_tactical_overlay_video_export(tmp_path):
    import cv2

    source, output = tmp_path / "source.mp4", tmp_path / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180)
    )
    for value in range(20):
        writer.write(np.full((180, 320, 3), value + 30, dtype=np.uint8))
    writer.release()
    snapshots = [{
        "source_time_s": 0,
        "players": [
            {"id": 1, "team": 0, "role": "player", "x": 30, "y": 20},
            {"id": 2, "team": 1, "role": "player", "x": 70, "y": 40},
        ],
        "ball": {"x": 50, "y": 34},
        "intelligence": {
            "possession": {"team": 0, "player_id": 1},
            "formations": {"0": {"name": "4-4-2"}, "1": {"name": "4-3-3"}},
        },
    }]
    progress = []
    result = build_annotated_video(
        source, output, snapshots,
        [{"source_time_s": .5, "label": "Pass", "review_status": "confirmed"}],
        [{"track_id": 1, "shirt_number": 9}],
        progress.append,
    )
    assert result == str(output)
    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    cap.release()
    assert progress[-1] == 100.0


def test_saved_pitch_can_ground_ask_when_pipeline_is_idle():
    body = api.AskRequest(question="Who has possession?", pitch={
        "calibrated": True,
        "players": [{"id": 7, "role": "player", "team": 0, "x": 50, "y": 34}],
        "intelligence": {
            "possession": {"team": 0, "player_id": 7, "confidence": .82},
        },
    })
    response = asyncio.run(api.tactical_ask(body))
    data = __import__("json").loads(response.body)
    assert data["backend"] == "local rules"
    assert "player 7" in data["answer"]
