"""
FootballVision FastAPI Server

Routes:
    GET  /                → mode selector (index.html)
    GET  /coach           → coach dashboard
    GET  /fan             → fan dashboard
    POST /session/start   → start capture + pipeline
    POST /session/stop    → stop pipeline, persist game
    WS   /ws/coach        → real-time tactical feed (~10 fps)
    WS   /ws/fan          → real-time fan feed (~2 fps)
    POST /fan/predict     → submit a fan play prediction
    GET  /clips           → list highlight clip filenames
    GET  /heatmap/{game_id} → heatmap PNG (Phase 5)
    GET  /stats           → current session stats (JSON)
"""

import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import AnalysisPipeline, PipelineConfig
from db.session import (
    create_tactical_session,
    finish_tactical_session,
    get_tactical_event,
    get_tactical_events,
    get_tactical_session,
    get_tactical_snapshots,
    init_db,
    list_tactical_sessions,
    save_game,
    save_play,
    save_prediction,
    save_tactical_snapshot,
    save_tactical_events,
    set_tactical_event_clip,
    set_tactical_artifacts,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


async def _shutdown_pipelines():
    if _pipeline.is_running:
        await _pipeline.stop()
    if _tac.is_running:
        await _tac.stop()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()
    logger.info("FootballVision API ready.")
    try:
        yield
    finally:
        await _shutdown_pipelines()


app = FastAPI(title="FootballVision", version="2.0.0", lifespan=lifespan)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
CLIPS_DIR = ROOT / "clips"
CLIPS_DIR.mkdir(exist_ok=True)
TACTICAL_VIDEO_DIRS = (ROOT / "data", ROOT / "uploads")
TACTICAL_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
TACTICAL_UPLOAD_LIMIT = 5 * 1024 * 1024 * 1024
ANALYSIS_DIR = ROOT / "analysis"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")

# Global pipeline singleton (legacy Coach/Fan views)
_pipeline: AnalysisPipeline = AnalysisPipeline()

# The football product runs on its own purpose-built loop.
from src.tactical_pipeline import TacticalPipeline, TacticalConfig
_tac: TacticalPipeline = TacticalPipeline()
_game_id: int = 0
_session_start: float = 0.0
_tactical_session_id: int = 0

# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """
    Send the root to the tactical view, or to the synthetic legacy demo when
    the server was explicitly started with ``main.py --demo``.

    Coach/Fan were the original NFL-era modes; the football-analysis product
    is the tactical view, so there is one entry point rather than a menu.
    """
    from fastapi.responses import RedirectResponse
    if os.environ.get("FOOTBALLVISION_DEMO") == "1":
        return RedirectResponse(url="/coach?demo")
    return RedirectResponse(url="/tactical")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Avoid a noisy browser 404 until a branded icon is added."""
    return Response(status_code=204)

@app.get("/modes", response_class=HTMLResponse)
async def modes(request: Request):
    """The original mode selector, kept for the Coach/Fan views."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/coach", response_class=HTMLResponse)
async def coach_page(request: Request):
    return templates.TemplateResponse(request=request, name="coach.html")

@app.get("/fan", response_class=HTMLResponse)
async def fan_page(request: Request):
    return templates.TemplateResponse(request=request, name="fan.html")

@app.get("/tactical", response_class=HTMLResponse)
async def tactical_page(request: Request):
    return templates.TemplateResponse(request=request, name="tactical.html")

# ---------------------------------------------------------------------------
# Session control
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    use_demo: bool = False
    coach_fps: int = 10
    fan_fps: int = 2
    jpeg_quality: int = 75
    display_index: int = 0
    window_id: Optional[int] = None

@app.get("/displays")
async def displays():
    """List attached displays so the UI can choose which screen to capture."""
    try:
        from src.capture import list_displays, QUARTZ_AVAILABLE
    except Exception as e:
        return JSONResponse({"available": False, "error": str(e), "displays": []})
    return JSONResponse({"available": QUARTZ_AVAILABLE, "displays": list_displays()})

@app.get("/windows")
async def windows():
    """List capturable application windows (e.g. the browser playing a match)."""
    try:
        from src.capture import list_windows, QUARTZ_AVAILABLE
    except Exception as e:
        return JSONResponse({"available": False, "error": str(e), "windows": []})
    return JSONResponse({"available": QUARTZ_AVAILABLE, "windows": list_windows()})

@app.get("/sources")
async def sources():
    """Everything capturable — windows first, then whole displays."""
    try:
        from src.capture import (list_displays, list_windows,
                                 has_screen_permission, QUARTZ_AVAILABLE,
                                 SCK_AVAILABLE)
    except Exception as e:
        return JSONResponse({"available": False, "error": str(e),
                             "windows": [], "displays": []})
    return JSONResponse({
        "available": QUARTZ_AVAILABLE,
        "permission": has_screen_permission(),
        # With ScreenCaptureKit the list also carries windows that are
        # minimised or on another Space, flagged `on_screen: false`. They are
        # raised automatically when analysis starts, because macOS renders no
        # pixels for them until then and no API can capture what is not drawn.
        "screencapturekit": SCK_AVAILABLE,
        "windows": list_windows(),
        "displays": list_displays(),
    })


@app.post("/capture/activate")
async def capture_activate(payload: dict):
    """Bring a window to the front so macOS renders (and can capture) it."""
    window_id = payload.get("window_id")
    if window_id is None:
        return JSONResponse({"ok": False, "error": "window_id required"},
                            status_code=400)
    try:
        from src.capture import activate_window
        return JSONResponse({"ok": bool(activate_window(int(window_id)))})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/capture/permission")
async def capture_permission():
    """
    Screen Recording permission status.

    Without it macOS returns a wallpaper-only image rather than an error, so
    the app must check explicitly or it will silently analyse an empty desktop.
    """
    from src.capture import has_screen_permission
    return JSONResponse({"granted": has_screen_permission()})

@app.post("/session/start")
async def session_start(body: StartRequest):
    global _pipeline, _game_id, _session_start

    if _pipeline.is_running:
        return JSONResponse({"ok": False, "error": "Session already running"})

    config = PipelineConfig(
        use_demo_source=body.use_demo,
        coach_fps=body.coach_fps,
        fan_fps=body.fan_fps,
        jpeg_quality=body.jpeg_quality,
        display_index=body.display_index,
        window_id=body.window_id,
    )
    _pipeline = AnalysisPipeline(config)
    _session_start = time.time()

    try:
        await _pipeline.start()
    except Exception as e:
        logger.exception("Pipeline start failed")
        return JSONResponse({"ok": False, "error": str(e)})

    _game_id = await save_game(source="demo" if body.use_demo else "screen")
    logger.info(f"Session started. game_id={_game_id}")
    return JSONResponse({"ok": True, "game_id": _game_id})

@app.post("/session/stop")
async def session_stop():
    global _game_id

    if not _pipeline.is_running:
        return JSONResponse({"ok": False, "error": "No active session"})

    await _pipeline.stop()

    # Persist game end time
    await save_game(game_id=_game_id, ended=True)

    stats = {
        "game_id": _game_id,
        "plays": _pipeline.play_count,
        "duration_s": round(time.time() - _session_start, 1),
        "formation_freq": _pipeline.formation_freq,
    }
    logger.info(f"Session stopped. stats={stats}")
    return JSONResponse({"ok": True, **stats})

@app.get("/stats")
async def get_stats():
    if not _pipeline.is_running:
        return JSONResponse({"running": False})
    return JSONResponse({
        "running": True,
        "frame_id": _pipeline.frame_id,
        "play_count": _pipeline.play_count,
        "formation_freq": _pipeline.formation_freq,
        "tendency": _pipeline._play_analyzer.tendency_stats() if _pipeline._play_analyzer else {},
    })

# ---------------------------------------------------------------------------
# Pitch calibration (pixel -> pitch-metre homography for the tactical view)
# ---------------------------------------------------------------------------

class Correspondence(BaseModel):
    image: list                    # [px, py] clicked on the frame
    pitch: Optional[list] = None   # [x_m, y_m] on the pitch, OR ...
    landmark: Optional[str] = None # ... a named PITCH_LANDMARKS key

class CalibrateRequest(BaseModel):
    points: list                   # list of Correspondence dicts

@app.post("/session/calibrate")
async def session_calibrate(body: CalibrateRequest):
    """
    Calibrate the pixel->pitch homography from 4+ point correspondences.

    Each point is {"image": [px, py], "pitch": [x_m, y_m]} or
    {"image": [px, py], "landmark": "center"}.
    """
    if _pipeline._homography is None and not _pipeline.is_running:
        return JSONResponse(
            {"ok": False, "error": "Start a session before calibrating"}
        )
    result = _pipeline.calibrate_homography(body.points)
    status = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status)

@app.get("/pitch/landmarks")
async def pitch_landmarks():
    """Return the named pitch landmarks (metres) for the calibration UI."""
    from src.homography import PITCH_LANDMARKS, PITCH_LENGTH, PITCH_WIDTH
    return JSONResponse({
        "pitch_length": PITCH_LENGTH,
        "pitch_width": PITCH_WIDTH,
        "landmarks": {k: list(v) for k, v in PITCH_LANDMARKS.items()},
    })

# ---------------------------------------------------------------------------
# Situation retrieval ("find similar situations")
# ---------------------------------------------------------------------------

@app.get("/situations/similar")
async def situations_similar(k: int = 4):
    """Return the past game-states most similar to the current one."""
    if not _pipeline.is_running:
        return JSONResponse({"ok": False, "error": "No active session"})
    k = max(1, min(int(k), 12))
    result = _pipeline.find_similar(k=k)
    return JSONResponse(result)

@app.get("/situations/count")
async def situations_count():
    """How many game-state snapshots are currently stored."""
    n = len(_pipeline._situations) if _pipeline._situations else 0
    return JSONResponse({"count": n})


# ---------------------------------------------------------------------------
# Tactical analysis (the football product)
# ---------------------------------------------------------------------------

class TacticalStartRequest(BaseModel):
    video_path: Optional[str] = None
    display_index: Optional[int] = None
    window_id: Optional[int] = None
    target_fps: int = 6


def _resolve_tactical_video(video_path: str) -> Path:
    """Resolve a user-selected video without allowing arbitrary file access."""
    candidate = Path(video_path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate

    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise ValueError("The selected video no longer exists")

    if not resolved.is_file() or resolved.suffix.lower() not in TACTICAL_VIDEO_EXTENSIONS:
        raise ValueError("The selected source is not a supported video file")

    for folder in TACTICAL_VIDEO_DIRS:
        try:
            resolved.relative_to(folder.resolve())
            return resolved
        except ValueError:
            continue
    raise ValueError("Videos must be selected from the data/ or uploads/ folder")

@app.post("/tactical/start")
async def tactical_start(body: TacticalStartRequest):
    global _tactical_session_id
    selected_sources = sum(value is not None for value in (
        body.video_path, body.display_index, body.window_id
    ))
    if selected_sources > 1:
        return JSONResponse(
            {"ok": False, "error": "Select exactly one video, window, or display"},
            status_code=400,
        )

    video_path = None
    if body.video_path is not None:
        try:
            video_path = str(_resolve_tactical_video(body.video_path))
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    if _tac.is_running:
        await _tac.stop()
    cfg = TacticalConfig(
        video_path=video_path,
        display_index=body.display_index,
        window_id=body.window_id,
        target_fps=max(1, min(body.target_fps, 15)),
    )
    # Keep the pipeline instance so its loaded ML engine can be reused when a
    # user stops and starts another source. Reloading all three models made a
    # routine source change look like the app had hung.
    _tac.config = cfg
    if video_path:
        source_type = "video"
        source_name = Path(video_path).name
        stored_path = str(Path(video_path).resolve().relative_to(ROOT))
    elif body.window_id is not None:
        source_type, source_name, stored_path = "window", f"Window {body.window_id}", None
    else:
        index = body.display_index if body.display_index is not None else 0
        source_type, source_name, stored_path = "display", f"Display {index}", None

    session_id = await create_tactical_session(source_name, source_type, stored_path)
    _tactical_session_id = session_id

    async def persist(payload, sid=session_id):
        await save_tactical_snapshot(sid, payload)

    async def finalise(frames, elapsed_s, error, sid=session_id):
        await finish_tactical_session(sid, frames, elapsed_s, error)

    async def persist_events(events, sid=session_id):
        await save_tactical_events(sid, events)

    _tac.snapshot_sink = persist
    _tac.stopped_sink = finalise
    _tac.event_sink = persist_events
    result = await _tac.start()
    result["session_id"] = session_id
    if not result.get("ok"):
        await finish_tactical_session(session_id, 0, 0, result.get("error"))
    return JSONResponse(result, status_code=200 if result.get("ok") else 400)

@app.post("/tactical/stop")
async def tactical_stop():
    result = await _tac.stop()
    result["session_id"] = _tactical_session_id
    return JSONResponse(result)

@app.get("/tactical/stats")
async def tactical_stats():
    return JSONResponse(_tac.stats())

@app.get("/tactical/similar")
async def tactical_similar(k: int = 4):
    return JSONResponse(_tac.find_similar(k=max(1, min(k, 12))))

@app.get("/tactical/videos")
async def tactical_videos():
    """Video files available to analyse, from data/ and uploads/."""
    out = []
    for folder in TACTICAL_VIDEO_DIRS:
        if not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in TACTICAL_VIDEO_EXTENSIONS:
                try:
                    resolved = _resolve_tactical_video(str(f))
                except ValueError:
                    continue
                out.append({
                    # A project-relative identifier avoids exposing the local
                    # account path in the browser and survives moving the repo.
                    "path": str(resolved.relative_to(ROOT)),
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / 1e6, 1),
                })
    return JSONResponse({"videos": out})


@app.post("/tactical/upload")
async def tactical_upload(video: UploadFile = File(...)):
    """Save a local match upload into the only directory analysis may read."""
    original = Path(video.filename or "match.mp4").name
    suffix = Path(original).suffix.lower()
    if suffix not in TACTICAL_VIDEO_EXTENSIONS:
        await video.close()
        return JSONResponse(
            {"ok": False, "error": "Use MP4, MOV, MKV, AVI, or WebM video"},
            status_code=400,
        )

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original).stem).strip("-.")
    safe_stem = safe_stem[:80] or "match"
    upload_dir = ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)
    target = upload_dir / f"{safe_stem}{suffix}"
    if target.exists():
        target = upload_dir / f"{safe_stem}-{uuid.uuid4().hex[:8]}{suffix}"
    partial = target.with_suffix(target.suffix + ".part")

    size = 0
    try:
        with partial.open("wb") as dst:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > TACTICAL_UPLOAD_LIMIT:
                    raise ValueError("Video exceeds the 5 GB upload limit")
                dst.write(chunk)
        partial.replace(target)
    except ValueError as exc:
        partial.unlink(missing_ok=True)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=413)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        logger.exception("Upload failed")
        return JSONResponse({"ok": False, "error": f"Could not save upload: {exc}"}, status_code=500)
    finally:
        await video.close()

    return JSONResponse({
        "ok": True,
        "video": {
            "path": str(target.relative_to(ROOT)),
            "name": target.name,
            "size_mb": round(size / 1e6, 1),
        },
    })


@app.get("/tactical/sessions")
async def tactical_sessions(limit: int = 30):
    return JSONResponse({"sessions": await list_tactical_sessions(limit)})


@app.get("/tactical/sessions/{session_id}")
async def tactical_session_detail(session_id: int):
    session = await get_tactical_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    snapshots = await get_tactical_snapshots(session_id)
    events = await get_tactical_events(session_id)
    for event in events:
        if event.get("clip_path"):
            event["clip_url"] = "/" + event["clip_path"].lstrip("/")
    return JSONResponse({"session": session, "snapshots": snapshots, "events": events})


@app.post("/tactical/sessions/{session_id}/events/{event_id}/clip")
async def tactical_event_clip(session_id: int, event_id: int):
    """Build a six-second local clip around a persisted video event."""
    session = await get_tactical_session(session_id)
    event = await get_tactical_event(session_id, event_id)
    if not session or not event:
        raise HTTPException(status_code=404, detail="Event not found")
    if session.get("source_type") != "video" or not session.get("source_path"):
        return JSONResponse(
            {"ok": False, "error": "Clip export is available for video-file sessions"},
            status_code=409,
        )
    if event.get("source_time_s") is None:
        return JSONResponse(
            {"ok": False, "error": "This event has no source-video timestamp"},
            status_code=409,
        )
    try:
        source = _resolve_tactical_video(session["source_path"])
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)

    filename = f"session-{session_id}-event-{event_id}.mp4"
    output = CLIPS_DIR / filename
    from src.event_clip import build_event_clip
    import asyncio
    try:
        await asyncio.get_running_loop().run_in_executor(
            None, build_event_clip, source, output, event["source_time_s"]
        )
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    clip_path = str(output.relative_to(ROOT))
    await set_tactical_event_clip(session_id, event_id, clip_path)
    return JSONResponse({"ok": True, "clip_url": f"/clips/{filename}"})


@app.post("/tactical/sessions/{session_id}/artifacts")
async def tactical_build_artifacts(session_id: int):
    session = await get_tactical_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    snapshots = await get_tactical_snapshots(session_id)
    events = await get_tactical_events(session_id)
    from src.session_report import build_session_artifacts
    import asyncio
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            build_session_artifacts,
            snapshots,
            events,
            ANALYSIS_DIR,
            session_id,
            session["source_name"],
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)

    def rel(path):
        return str(Path(path).resolve().relative_to(ROOT)) if path else None

    await set_tactical_artifacts(
        session_id,
        report_path=rel(result["report"]),
        tracking_path=rel(result["tracking"]),
        metrics_path=rel(result["metrics"]),
        events_path=rel(result["events"]),
    )
    downloads = {
        kind: f"/tactical/sessions/{session_id}/download/{kind}"
        for kind in ("tracking", "metrics", "events")
    }
    if result["report"]:
        downloads["report"] = f"/tactical/sessions/{session_id}/download/report"
    return JSONResponse({
        "ok": True,
        "report_available": bool(result["report"]),
        "tracking_rows": result["tracking_rows"],
        "metric_rows": result["metric_rows"],
        "downloads": downloads,
    })


@app.get("/tactical/sessions/{session_id}/download/{kind}")
async def tactical_download(session_id: int, kind: str):
    if kind not in {"report", "tracking", "metrics", "events"}:
        raise HTTPException(status_code=404, detail="Artifact not found")
    session = await get_tactical_session(session_id)
    if not session or not session.get(f"{kind}_path"):
        raise HTTPException(status_code=404, detail="Build the report first")
    path = (ROOT / session[f"{kind}_path"]).resolve()
    try:
        path.relative_to(ANALYSIS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file is missing")
    media = "image/png" if kind == "report" else "text/csv"
    return FileResponse(path, media_type=media, filename=path.name)


class AskRequest(BaseModel):
    question: str
    backend: str = "auto"
    pitch: Optional[dict] = None

@app.post("/tactical/ask")
async def tactical_ask(body: AskRequest):
    """
    Answer a question using only what the vision pipeline measured.

    The grounding source is detector output, not text, so the model is being
    arbitrated by a different modality. The response always carries the facts
    block so the caller can show exactly what the model was allowed to see.
    """
    from src.grounded_ask import answer
    # A reviewed historical moment lives in the browser rather than in the
    # currently idle pipeline. Accept that exact displayed measurement state
    # so Ask remains grounded while the user scrubs a saved session.
    pitch = body.pitch or (_tac._last_payload or {}).get("pitch") or {}
    loop = __import__("asyncio").get_event_loop()
    result = await loop.run_in_executor(None, answer, body.question, pitch, body.backend)
    return JSONResponse(result, status_code=200 if result.get("ok") else 400)

@app.websocket("/ws/tactical")
async def ws_tactical(websocket: WebSocket):
    await websocket.accept()
    _tac.add_client(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _tac.remove_client(websocket)

# ---------------------------------------------------------------------------
# Fan prediction endpoint
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    session_id: str
    prediction: str          # "run", "pass", or a formation name
    play_id: int = 0

@app.post("/fan/predict")
async def fan_predict(body: PredictRequest):
    # Evaluate against last resolved play (simple demo logic)
    last_payload = _pipeline._last_fan_payload
    if not last_payload:
        return JSONResponse({"ok": False, "error": "No play data yet"})

    correct = False
    points = 0
    actual_type = "unknown"

    if _pipeline._play_analyzer:
        tendency = _pipeline._play_analyzer.tendency_stats()
        # Determine last dominant play type from tendency
        actual_type = "pass" if tendency.get("pass", 0) > tendency.get("run", 0) else "run"

    if body.prediction in ("run", "pass"):
        correct = body.prediction == actual_type
        points = 10 if correct else 0
    else:
        # Formation guess
        current_formation = last_payload.get("formation_name", "")
        correct = body.prediction.lower() in current_formation.lower()
        points = 25 if correct else 0

    await save_prediction(
        play_id=body.play_id,
        session_id=body.session_id,
        prediction=body.prediction,
        correct=correct,
        points=points,
    )

    return JSONResponse({
        "ok": True,
        "correct": correct,
        "points": points,
        "actual": actual_type,
    })

# ---------------------------------------------------------------------------
# Highlight clips
# ---------------------------------------------------------------------------

@app.get("/clips")
async def list_clips():
    clips = sorted(
        [f.name for f in CLIPS_DIR.glob("*.mp4")],
        reverse=True
    )[:20]
    return JSONResponse({"clips": clips})

# ---------------------------------------------------------------------------
# Heatmap endpoint
# ---------------------------------------------------------------------------

@app.get("/heatmap/{game_id}")
async def get_heatmap(game_id: int):
    """Return the current session's player position heatmap as a PNG."""
    if not _pipeline.is_running or not _pipeline._heatmap:
        raise HTTPException(status_code=404, detail="No heatmap data available")

    import io
    from fastapi.responses import Response
    loop = __import__("asyncio").get_event_loop()
    png_bytes = await loop.run_in_executor(
        None,
        lambda: _pipeline._heatmap.render(f"Game #{game_id} — Player Heatmap")
    )
    return Response(content=png_bytes, media_type="image/png")

# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------

@app.websocket("/ws/coach")
async def ws_coach(websocket: WebSocket):
    await websocket.accept()
    _pipeline.add_coach_client(websocket)
    try:
        while True:
            # Keep connection alive; pipeline pushes data
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _pipeline.remove_client(websocket)

@app.websocket("/ws/fan")
async def ws_fan(websocket: WebSocket):
    await websocket.accept()
    _pipeline.add_fan_client(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _pipeline.remove_client(websocket)
