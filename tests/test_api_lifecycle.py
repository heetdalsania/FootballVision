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

    async def stop(self):
        self.stop_calls += 1
        self.is_running = False
        return {"ok": True}

    async def start(self):
        self.start_calls += 1
        self.is_running = True
        return {"ok": True, "source": "test"}


def test_tactical_start_reuses_loaded_pipeline(monkeypatch):
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


def test_video_listing_uses_project_relative_identifiers():
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
