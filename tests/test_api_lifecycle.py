"""Regression tests for tactical API and background-task lifecycle behavior."""

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np

from src.tactical_pipeline import TacticalPipeline
from ui import api


def _json(response):
    return json.loads(response.body)


class _FakePipeline:
    def __init__(self, running=False):
        self.is_running = running
        self.config = None
        self.stop_calls = 0
        self.start_calls = 0
        self.paused = False

    async def stop(self):
        self.stop_calls += 1
        self.is_running = False
        return {"ok": True}

    async def start(self):
        self.start_calls += 1
        self.is_running = True
        return {"ok": True, "source": "test"}

    async def pause(self):
        self.paused = True
        return {"ok": True, "paused": True}

    async def resume(self):
        self.paused = False
        return {"ok": True, "paused": False}


def _install_test_video_root(monkeypatch, tmp_path):
    """Give API tests an isolated media tree instead of relying on local footage."""
    root = tmp_path / "project"
    data_dir = root / "data"
    uploads_dir = root / "uploads"
    data_dir.mkdir(parents=True)
    uploads_dir.mkdir()
    video = data_dir / "sample.mp4"
    video.write_bytes(b"test video placeholder")
    monkeypatch.setattr(api, "ROOT", root)
    monkeypatch.setattr(api, "TACTICAL_VIDEO_DIRS", (data_dir, uploads_dir))
    return video


def test_tactical_start_reuses_loaded_pipeline(monkeypatch, tmp_path):
    _install_test_video_root(monkeypatch, tmp_path)
    pipeline = _FakePipeline(running=True)
    monkeypatch.setattr(api, "_tac", pipeline)
    async def create(*_args):
        return 123
    async def finish(*_args):
        return None
    monkeypatch.setattr(api, "create_tactical_session", create)
    monkeypatch.setattr(api, "finish_tactical_session", finish)

    response = asyncio.run(api.tactical_start(api.TacticalStartRequest(
        video_path="data/sample.mp4", target_fps=99
    )))

    assert response.status_code == 200
    assert api._tac is pipeline
    assert pipeline.stop_calls == 1 and pipeline.start_calls == 1
    assert pipeline.config.target_fps == 15
    assert pipeline.config.loop_video is False
    assert Path(pipeline.config.video_path).name == "sample.mp4"


def test_tactical_start_rejects_files_outside_media_dirs(monkeypatch, tmp_path):
    pipeline = _FakePipeline()
    monkeypatch.setattr(api, "_tac", pipeline)
    outside_video = tmp_path / "outside.mp4"
    outside_video.write_bytes(b"not a real video, but a real path")

    response = asyncio.run(api.tactical_start(api.TacticalStartRequest(
        video_path=str(outside_video)
    )))

    assert response.status_code == 400
    assert not pipeline.start_calls
    assert "data/ or uploads/" in _json(response)["error"]


def test_tactical_start_rejects_ambiguous_source(monkeypatch):
    pipeline = _FakePipeline()
    monkeypatch.setattr(api, "_tac", pipeline)

    response = asyncio.run(api.tactical_start(api.TacticalStartRequest(
        video_path="data/sample.mp4", display_index=0
    )))

    assert response.status_code == 400
    assert not pipeline.start_calls


def test_video_listing_uses_project_relative_identifiers(monkeypatch, tmp_path):
    _install_test_video_root(monkeypatch, tmp_path)
    response = asyncio.run(api.tactical_videos())
    videos = _json(response)["videos"]
    assert videos
    assert all(not Path(item["path"]).is_absolute() for item in videos)


def test_favicon_does_not_generate_a_browser_404():
    response = asyncio.run(api.favicon())
    assert response.status_code == 204


def test_demo_flag_redirects_to_working_synthetic_view(monkeypatch):
    monkeypatch.setenv("FOOTBALLVISION_DEMO", "1")
    response = asyncio.run(api.index())
    assert response.headers["location"] == "/coach?demo"


def test_primary_pages_and_assets_render_without_server_errors():
    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://test",
            follow_redirects=False,
        ) as client:
            expected = {
                "/": 307,
                "/tactical": 200,
                "/modes": 200,
                "/coach": 200,
                "/fan": 200,
                "/tactical/videos": 200,
                "/tactical/stats": 200,
                "/clips": 200,
                "/favicon.ico": 204,
                "/static/js/tactical.js?v=15": 200,
            }
            for path, status in expected.items():
                response = await client.get(path)
                assert response.status_code == status, path

    asyncio.run(check())


def test_tactical_stats_exposes_session_for_companion_monitor(monkeypatch):
    class RunningPipeline:
        def stats(self):
            return {"running": True, "paused": False, "source": "window 42"}

    monkeypatch.setattr(api, "_tac", RunningPipeline())
    monkeypatch.setattr(api, "_tactical_session_id", 456)

    response = asyncio.run(api.tactical_stats())

    assert _json(response) == {
        "running": True,
        "paused": False,
        "source": "window 42",
        "session_id": 456,
    }


def test_tactical_live_exposes_latest_frame_for_native_monitor(monkeypatch):
    class RunningPipeline:
        is_running = True
        _paused = True
        _last_payload = {"type": "tactical", "frame_id": 17, "frame_b64": "abc"}

    monkeypatch.setattr(api, "_tac", RunningPipeline())
    monkeypatch.setattr(api, "_tactical_session_id", 789)

    response = asyncio.run(api.tactical_live())

    assert _json(response) == {
        "running": True,
        "paused": True,
        "session_id": 789,
        "payload": {"type": "tactical", "frame_id": 17, "frame_b64": "abc"},
    }


def test_shutdown_stops_both_products(monkeypatch):
    legacy = _FakePipeline(running=True)
    tactical = _FakePipeline(running=True)
    monkeypatch.setattr(api, "_pipeline", legacy)
    monkeypatch.setattr(api, "_tac", tactical)

    asyncio.run(api._shutdown_pipelines())

    assert legacy.stop_calls == 1
    assert tactical.stop_calls == 1


def test_analysis_failure_is_broadcast_and_cleans_up(monkeypatch):
    class Source:
        released = False

        def release(self):
            self.released = True

    class Client:
        def __init__(self):
            self.messages = []

        async def send_text(self, message):
            self.messages.append(json.loads(message))

    pipeline = TacticalPipeline()
    source = Source()
    client = Client()
    pipeline._source = source
    pipeline._running = True
    pipeline.add_client(client)
    monkeypatch.setattr(
        pipeline, "_read_frame", lambda: np.zeros((32, 32, 3), dtype=np.uint8)
    )

    def fail(_frame):
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(pipeline, "_analyze", fail)
    asyncio.run(pipeline._run())

    assert not pipeline.is_running
    assert source.released
    assert pipeline._source is None
    assert client.messages[-1] == {
        "type": "error",
        "error": "Analysis stopped: synthetic inference failure",
    }


def test_video_eof_finalises_and_broadcasts_completion(monkeypatch):
    class Source:
        released = False

        def release(self):
            self.released = True

    class Client:
        def __init__(self):
            self.messages = []

        async def send_text(self, message):
            self.messages.append(json.loads(message))

    async def run():
        pipeline = TacticalPipeline()
        pipeline.config.video_path = "sample.mp4"
        pipeline._source = Source()
        pipeline._running = True
        client = Client()
        pipeline.add_client(client)
        finalised = []

        def eof():
            pipeline._source_eof = True
            return None

        async def finish(frames, elapsed, error):
            finalised.append((frames, elapsed, error))

        monkeypatch.setattr(pipeline, "_read_frame", eof)
        pipeline.stopped_sink = finish
        await pipeline._run()
        return pipeline, client, finalised

    pipeline, client, finalised = asyncio.run(run())
    assert not pipeline.is_running
    assert finalised and finalised[0][2] is None
    assert client.messages[-1]["type"] == "complete"
    assert client.messages[-1]["progress_pct"] == 100.0


def test_pause_and_resume_endpoints(monkeypatch):
    pipeline = _FakePipeline(running=True)
    monkeypatch.setattr(api, "_tac", pipeline)
    paused = _json(asyncio.run(api.tactical_pause()))
    resumed = _json(asyncio.run(api.tactical_resume()))
    assert paused["paused"] is True
    assert resumed["paused"] is False
