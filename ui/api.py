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
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import AnalysisPipeline, PipelineConfig
from db.session import init_db, save_game, save_play, save_prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="FootballVision", version="2.0.0")

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
CLIPS_DIR = ROOT / "clips"
CLIPS_DIR.mkdir(exist_ok=True)

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

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("FootballVision API ready.")

@app.on_event("shutdown")
async def shutdown():
    if _pipeline.is_running:
        await _pipeline.stop()

# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """
    Send the root straight to the tactical view.

    Coach/Fan were the original NFL-era modes; the football-analysis product
    is the tactical view, so there is one entry point rather than a menu.
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/tactical")

@app.get("/modes", response_class=HTMLResponse)
async def modes(request: Request):
    """The original mode selector, kept for the Coach/Fan views."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/coach", response_class=HTMLResponse)
async def coach_page(request: Request):
    return templates.TemplateResponse("coach.html", {"request": request})

@app.get("/fan", response_class=HTMLResponse)
async def fan_page(request: Request):
    return templates.TemplateResponse("fan.html", {"request": request})

@app.get("/tactical", response_class=HTMLResponse)
async def tactical_page(request: Request):
    return templates.TemplateResponse("tactical.html", {"request": request})

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

@app.post("/tactical/start")
async def tactical_start(body: TacticalStartRequest):
    global _tac
    if _tac.is_running:
        await _tac.stop()
    cfg = TacticalConfig(
        video_path=body.video_path,
        display_index=body.display_index,
        window_id=body.window_id,
        target_fps=max(1, min(body.target_fps, 15)),
    )
    _tac = TacticalPipeline(cfg)
    result = await _tac.start()
    return JSONResponse(result, status_code=200 if result.get("ok") else 400)

@app.post("/tactical/stop")
async def tactical_stop():
    return JSONResponse(await _tac.stop())

@app.get("/tactical/stats")
async def tactical_stats():
    return JSONResponse(_tac.stats())

@app.get("/tactical/similar")
async def tactical_similar(k: int = 4):
    return JSONResponse(_tac.find_similar(k=max(1, min(k, 12))))

@app.get("/tactical/videos")
async def tactical_videos():
    """Video files available to analyse, from data/ and uploads/."""
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    out = []
    for folder in (ROOT / "data", ROOT / "uploads"):
        if not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in exts:
                out.append({
                    "path": str(f),
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / 1e6, 1),
                })
    return JSONResponse({"videos": out})


class AskRequest(BaseModel):
    question: str
    backend: str = "auto"

@app.post("/tactical/ask")
async def tactical_ask(body: AskRequest):
    """
    Answer a question using only what the vision pipeline measured.

    The grounding source is detector output, not text, so the model is being
    arbitrated by a different modality. The response always carries the facts
    block so the caller can show exactly what the model was allowed to see.
    """
    from src.grounded_ask import answer
    pitch = (_tac._last_payload or {}).get("pitch") or {}
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
