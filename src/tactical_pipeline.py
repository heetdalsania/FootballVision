"""
Tactical Pipeline — the live analysis loop behind /tactical

This replaces the original hand-rolled detect/track/calibrate chain with the
proven `FootballEngine` (see src/fv_engine.py, adapted from roboflow/sports),
and layers this project's own contributions on top:

    FootballEngine  ->  detection, tracking, team assignment, pitch homography
    tactical_metrics->  team shape, compactness, pitch control, pressing
    state_embedding ->  "find similar situations" retrieval

Sources are pluggable: a video file (reliable, reproducible) or live screen
capture (only sees what macOS is actually rendering).

Why a separate module from `pipeline.py`: that file still serves the legacy
Coach and Fan views built for the NFL-era product, with a formation
classifier and play analyzer that do not apply here. Rather than bend it, the
football product gets a purpose-built loop.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TacticalConfig:
    """Tunable parameters for the live loop."""
    target_fps: int = 6            # frames pushed to clients per second
    jpeg_quality: int = 70
    analysis_width: int = 1280     # engine input width (detection resolution)
    stream_width: int = 960        # width of the JPEG sent to the browser
    device: str = "mps"

    # Source: exactly one of these
    video_path: Optional[str] = None
    display_index: Optional[int] = None
    window_id: Optional[int] = None

    loop_video: bool = True        # restart a video file when it ends
    snapshot_every: int = 15       # frames between retrieval snapshots


class TacticalPipeline:
    """Runs the engine over a source and broadcasts results to WebSocket clients."""

    def __init__(self, config: Optional[TacticalConfig] = None):
        self.config = config or TacticalConfig()

        self._clients: Set[Any] = set()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        # One worker serializes model loading and inference across stop/start.
        # Cancelling an asyncio future cannot stop Python code already running
        # in a thread; a shared pool prevents a quick restart from using the
        # same model concurrently with that finishing frame.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="footballvision-tactical"
        )

        self._engine = None
        self._source = None
        self._embedder = None
        self._situations = None
        self._intelligence = None
        self._capture_error: Optional[str] = None
        self._source_time_s: Optional[float] = None
        self._last_source_time_s: Optional[float] = None
        self._source_looped = False

        self.frame_id = 0
        self.started_at = 0.0
        self._last_payload: Dict = {}
        self._last_players: List[Dict] = []
        self._last_error: Optional[str] = None
        self._fps_ema: float = 0.0
        # Optional async hooks let the web layer persist sessions without
        # coupling this reusable vision loop to SQLite or FastAPI.
        self.snapshot_sink: Optional[Callable[[Dict], Awaitable[None]]] = None
        self.stopped_sink: Optional[
            Callable[[int, float, Optional[str]], Awaitable[None]]
        ] = None
        self.event_sink: Optional[Callable[[List[Dict]], Awaitable[None]]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> Dict:
        if self._running:
            return {"ok": False, "error": "Session already running"}

        self._last_error = None
        self._capture_error = None
        self._last_payload = {}
        self._last_players = []
        self._fps_ema = 0.0
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self._executor, self._load)
        except Exception as exc:
            logger.exception("Failed to start")
            return {"ok": False, "error": str(exc)}

        self._running = True
        self.started_at = time.time()
        self.frame_id = 0
        self._task = asyncio.create_task(self._run())
        return {"ok": True, "source": self._source_label()}

    async def stop(self) -> Dict:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._close_source()
        return {"ok": True, "frames": self.frame_id}

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Blocking setup: models and video source."""
        from src.fv_engine import FootballEngine
        from src.match_intelligence import MatchIntelligence
        from src.state_embedding import StateEmbedder, SituationStore

        # Validate/open the source before paying the model-load cost. This also
        # surfaces a missing file or screen permission error immediately.
        self._open_source()
        try:
            if self._engine is None:
                self._engine = FootballEngine(
                    device=self.config.device,
                    player_imgsz=self.config.analysis_width,
                )
                self._engine.load()
            else:
                # Keep heavyweight model weights, but never carry tracker ids,
                # homography, kit centroids, or ball position into a new match.
                self._engine.reset_session()

            self._embedder = StateEmbedder()
            self._situations = SituationStore(dim=self._embedder.dim)
            self._intelligence = MatchIntelligence()
            self._source_time_s = None
            self._last_source_time_s = None
            self._source_looped = False
        except Exception:
            self._close_source()
            raise

    def _close_source(self) -> None:
        if self._source is None:
            return
        try:
            self._source.release()
        except Exception:
            logger.debug("Source release failed", exc_info=True)
        finally:
            self._source = None

    def _open_source(self) -> None:
        cfg = self.config
        if cfg.video_path:
            import cv2
            cap = cv2.VideoCapture(cfg.video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video: {cfg.video_path}")
            self._source = cap
            logger.info("Source: video file %s", cfg.video_path)
        else:
            from src.capture import ScreenCapture, has_screen_permission
            if not has_screen_permission():
                raise PermissionError(
                    "macOS Screen Recording permission is not granted to this "
                    "process. Without it macOS returns a wallpaper-only image "
                    "with all windows removed. Grant it in System Settings > "
                    "Privacy & Security > Screen & System Audio Recording, then "
                    "restart the app running this server."
                )
            if cfg.window_id:
                # macOS renders nothing for a window on an inactive Space, and
                # an unrendered window cannot be captured by any API — the old
                # display-crop path silently returned the wallpaper instead.
                # Raise it first so the pixels exist.
                from src.capture import activate_window
                if activate_window(cfg.window_id):
                    logger.info("Activated window %s before capture", cfg.window_id)
                    time.sleep(1.0)  # let the Space switch finish compositing

            cap = ScreenCapture(
                display_index=cfg.display_index or 0,
                window_id=cfg.window_id,
                target_fps=max(cfg.target_fps * 2, 10),
            )
            cap.start()
            self._source = cap
            logger.info("Source: screen capture")

    def _source_label(self) -> str:
        cfg = self.config
        if cfg.video_path:
            import os
            return f"video: {os.path.basename(cfg.video_path)}"
        if cfg.window_id:
            return f"window {cfg.window_id}"
        return f"display {cfg.display_index or 0}"

    def _read_frame(self) -> Optional[np.ndarray]:
        import cv2
        src = self._source
        if src is None:
            return None
        if self.config.video_path:
            ok, frame = src.read()
            if not ok:
                if self.config.loop_video:
                    src.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = src.read()
                if not ok:
                    return None
            source_time = max(0.0, float(src.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            self._source_looped = (
                self._last_source_time_s is not None
                and source_time + .05 < self._last_source_time_s
            )
            self._last_source_time_s = source_time
            self._source_time_s = source_time
            return frame
        self._source_time_s = None
        frame = src.get_frame()
        # A screen-capture miss has a reason worth showing. Reporting "no
        # players" for a window macOS is not rendering is the failure mode that
        # cost the most debugging time on this project.
        self._capture_error = getattr(src, "last_error", None)
        return frame

    # ------------------------------------------------------------------
    # Client registration
    # ------------------------------------------------------------------
    def add_client(self, ws) -> None:
        self._clients.add(ws)

    def remove_client(self, ws) -> None:
        self._clients.discard(ws)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def _run(self) -> None:
        import cv2

        loop = asyncio.get_event_loop()
        interval = 1.0 / max(1, self.config.target_fps)
        last_send = 0.0

        try:
            while self._running:
                t0 = time.monotonic()

                frame = await loop.run_in_executor(self._executor, self._read_frame)
                if frame is None:
                    await asyncio.sleep(0.05)
                    continue

                # Resize once for analysis; the engine's imgsz handles the rest.
                if frame.shape[1] != self.config.analysis_width:
                    h = int(frame.shape[0] * self.config.analysis_width / frame.shape[1])
                    frame = cv2.resize(frame, (self.config.analysis_width, h))

                self.frame_id += 1
                result = await loop.run_in_executor(self._executor, self._analyze, frame)
                new_events = (result.intelligence or {}).get("new_events") or []
                if new_events and self.event_sink is not None:
                    try:
                        await self.event_sink(new_events)
                    except Exception:
                        logger.exception("Could not persist tactical events")

                now = time.monotonic()
                dt = now - t0
                self._fps_ema = (0.8 * self._fps_ema + 0.2 * (1.0 / dt)) if dt > 0 else self._fps_ema

                # Build the payload on the interval whether or not anyone is
                # listening: /tactical/ask and /tactical/stats read the latest
                # state, so gating this on connected clients meant the ask
                # endpoint saw nothing unless a browser happened to be open.
                send_due = now - last_send >= interval
                snapshot_due = (
                    self.snapshot_sink is not None
                    and self.frame_id % max(1, self.config.snapshot_every) == 0
                )
                if send_due or snapshot_due:
                    payload = self._build_payload(result)
                    self._last_payload = payload
                    if snapshot_due:
                        try:
                            await self.snapshot_sink(payload)
                        except Exception:
                            logger.exception("Could not persist tactical snapshot")
                    if send_due and self._clients:
                        await self._broadcast(payload)
                    if send_due:
                        last_send = now

                await asyncio.sleep(0)  # yield
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Tactical analysis stopped unexpectedly")
            self._last_error = f"Analysis stopped: {exc}"
            payload = {"type": "error", "error": self._last_error}
            self._last_payload = payload
            await self._broadcast(payload)
        finally:
            self._running = False
            self._close_source()
            if self.stopped_sink is not None:
                try:
                    elapsed = max(0.0, time.time() - self.started_at)
                    await self.stopped_sink(self.frame_id, elapsed, self._last_error)
                except Exception:
                    logger.exception("Could not finalise tactical session")

    def _analyze(self, frame: np.ndarray):
        """Blocking: run the engine and this project's analysis layers."""
        result = self._engine.process(frame, annotate=True)
        self._last_error = None if result.source_status == "ok" else result.source_status

        players = result.players or []
        self._last_players = players
        if self._intelligence is not None:
            if self._source_looped:
                self._intelligence.start_new_period()
                self._source_looped = False
            evidence_time_s = (
                self._source_time_s if self._source_time_s is not None
                else max(0.0, time.time() - self.started_at)
            )
            result.intelligence = self._intelligence.update(
                players,
                result.ball,
                result.frame_id,
                evidence_time_s,
                self._source_time_s,
            )

        # Retrieval snapshots: only meaningful with both teams present.
        if (self._situations is not None and players
                and self.frame_id % max(1, self.config.snapshot_every) == 0):
            teams = {p["team"] for p in players if p.get("role") == "player"}
            if 0 in teams and 1 in teams:
                try:
                    vec = self._embedder.embed(players)
                    self._situations.add(vec, {
                        "frame_id": self.frame_id,
                        "t": round(time.time() - self.started_at, 2),
                        "players": players,
                    })
                except Exception:
                    pass
        return result

    # ------------------------------------------------------------------
    def _build_payload(self, result) -> Dict:
        from src.homography import PITCH_LENGTH, PITCH_WIDTH
        from src.tactical_metrics import compute_metrics

        players = result.players or []
        outfield = [p for p in players if p.get("role") == "player"]

        concepts = compute_metrics(outfield, PITCH_LENGTH, PITCH_WIDTH) if outfield else None

        team_counts = {"0": 0, "1": 0, "-1": 0}
        for p in outfield:
            team_counts[str(p.get("team", -1))] = team_counts.get(str(p.get("team", -1)), 0) + 1

        return {
            "type": "tactical",
            "frame_id": result.frame_id,
            "fps": round(self._fps_ema, 1),
            "frame_b64": self._encode(result.annotated),
            "status": self._capture_error or result.source_status,
            "capture_error": self._capture_error,
            "green_fraction": result.green_fraction,
            "pitch": {
                "pitch_length": PITCH_LENGTH,
                "pitch_width": PITCH_WIDTH,
                "calibrated": result.calibrated,
                "n_keypoints": result.n_keypoints,
                "players": players,
                "ball": result.ball,
                "counts": result.counts,
                "team_counts": team_counts,
                "team_ready": result.team_ready,
                "team_status": result.team_status,
                "concepts": concepts,
                "intelligence": result.intelligence or {},
                "situations_stored": len(self._situations) if self._situations else 0,
            },
            "timings_ms": result.timings_ms,
            "elapsed_s": round(time.time() - self.started_at, 1),
            "source_time_s": self._source_time_s,
        }

    def _encode(self, frame: Optional[np.ndarray]) -> Optional[str]:
        if frame is None:
            return None
        import cv2
        w = self.config.stream_width
        if frame.shape[1] != w:
            h = int(frame.shape[0] * w / frame.shape[1])
            frame = cv2.resize(frame, (w, h))
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        return base64.b64encode(buf).decode("utf-8") if ok else None

    async def _broadcast(self, payload: Dict) -> None:
        if not self._clients:
            return
        msg = json.dumps(payload)
        dead = set()
        for ws in self._clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ------------------------------------------------------------------
    def find_similar(self, k: int = 4) -> Dict:
        """Retrieve past moments resembling the current one."""
        if not self._situations or not self._last_players:
            return {"ok": False, "error": "Nothing captured yet"}
        outfield = [p for p in self._last_players if p.get("role") == "player"]
        teams = {p["team"] for p in outfield}
        if 0 not in teams or 1 not in teams:
            return {"ok": False, "error": "Need both teams on the pitch to match"}
        vec = self._embedder.embed(outfield)
        now_t = round(time.time() - self.started_at, 2)
        hits = self._situations.query(vec, k=k, exclude_within_s=3.0, query_time_s=now_t)
        return {"ok": True, "store_size": len(self._situations), "results": hits}

    def stats(self) -> Dict:
        return {
            "running": self._running,
            "frame_id": self.frame_id,
            "fps": round(self._fps_ema, 1),
            "source": self._source_label() if self._running else None,
            "error": self._last_error,
            "elapsed_s": round(time.time() - self.started_at, 1) if self._running else 0,
            "events": len(self._intelligence.events) if self._intelligence else 0,
        }
