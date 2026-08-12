"""
Async Frame Processing Pipeline for FootballVision

Runs the full detection → tracking → formation → play analysis chain
in a background asyncio task, broadcasting results to WebSocket clients.

Architecture:
    VideoSource → asyncio.Queue → worker coroutine → broadcaster
                                                     ├── /ws/coach  (~10 fps)
                                                     └── /ws/fan    (~2 fps)
"""

import asyncio
import base64
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Tunable pipeline parameters."""
    coach_fps: int = 10      # frames/sec pushed to coach WebSocket
    fan_fps: int = 2         # frames/sec pushed to fan WebSocket
    frame_queue_size: int = 8
    jpeg_quality: int = 75   # JPEG compression for frame_b64
    # Detection resolution. A broadcast wide shot puts players at only a few
    # dozen pixels tall; downscaling a 4K capture to 720p before detection was
    # making distant players flicker in and out between frames.
    frame_width: int = 1920  # resize before encoding (0 = no resize)
    frame_height: int = 1080
    use_demo_source: bool = False  # use DemoSource instead of screen capture
    snapshot_every: int = 15  # frames between game-state snapshots for retrieval
    display_index: int = 0  # which display to screen-capture (real mode)
    window_id: Optional[int] = None  # capture just this window instead
    use_field_mask: bool = True  # reject detections off the playing surface
    detector_conf: float = 0.20  # low: broadcast players are small
    detector_imgsz: int = 1280   # high: preserves small distant players
    # Measured head-to-head on a real broadcast frame (2022 UCL final):
    # Roboflow gives 23 players + 2 referees + 1 goalkeeper + ball — close to
    # the true count and semantically labelled. Generic COCO YOLO reports
    # 30-45 "persons" because it cannot tell a player from a spectator, a
    # steward or a camera operator. So the football-specific model wins on
    # both counts; keep the flags separate so either can be disabled.
    use_roboflow_detector: bool = True
    use_roboflow_pitch: bool = True
    autocalibrate_every: int = 30  # frames between pitch-keypoint recalibrations


class AnalysisPipeline:
    """
    Coordinates the full real-time analysis pipeline.

    Usage:
        pipeline = AnalysisPipeline(config)
        await pipeline.start()
        ...
        await pipeline.stop()

    WebSocket clients register via:
        pipeline.add_coach_client(ws)
        pipeline.add_fan_client(ws)
        pipeline.remove_client(ws)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # WebSocket subscriber sets
        self._coach_clients: Set[Any] = set()
        self._fan_clients: Set[Any] = set()

        # Internal state
        self._frame_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.frame_queue_size
        )
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Game-level accumulators
        self.frame_id: int = 0
        self.game_start: float = 0.0
        self.play_count: int = 0
        self.formation_freq: Dict[str, int] = {}

        # Last analysis result (for fan polling)
        self._last_coach_payload: Dict = {}
        self._last_fan_payload: Dict = {}

        # Lazy-loaded ML components (loaded in start())
        self._detector = None
        self._tracker = None
        self._classifier = None
        self._play_analyzer = None
        self._source = None

        # Football-analysis components (pixel -> pitch metres + teams)
        self._homography = None
        self._team_assigner = None

        # Game-state retrieval ("find similar situations")
        self._embedder = None
        self._situations = None
        self._last_pitch_players: List[Dict] = []

        # Fan prediction window state
        self._predict_window_open = False
        self._predict_window_end: float = 0.0
        self._last_highlight: Optional[str] = None
        self._last_ball = None  # most confident ball detection this frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Initialize ML components and begin frame processing."""
        if self._running:
            return

        logger.info("Pipeline starting — loading ML components…")
        await asyncio.get_event_loop().run_in_executor(None, self._load_components)
        logger.info("ML components loaded.")

        # Start the video source — without this the source stays idle and
        # get_frame() returns None forever, so no frames are ever processed.
        if self._source is not None:
            started = self._source.start()
            if not started:
                logger.warning("Video source failed to start (%s)", type(self._source).__name__)

        self._running = True
        self.game_start = time.time()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        """Gracefully stop the pipeline."""
        self._running = False
        if self._source:
            try:
                self._source.stop()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._play_analyzer.reset() if self._play_analyzer else None
        logger.info("Pipeline stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # WebSocket client registration
    # ------------------------------------------------------------------

    def add_coach_client(self, ws):
        self._coach_clients.add(ws)
        logger.debug(f"Coach client added. Total: {len(self._coach_clients)}")

    def add_fan_client(self, ws):
        self._fan_clients.add(ws)
        logger.debug(f"Fan client added. Total: {len(self._fan_clients)}")

    def remove_client(self, ws):
        self._coach_clients.discard(ws)
        self._fan_clients.discard(ws)

    # ------------------------------------------------------------------
    # Internal: component loading (blocking, run in executor)
    # ------------------------------------------------------------------

    def _load_components(self):
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from src.detector import Detector
        from src.tracker import Tracker
        from src.play_analyzer import PlayAnalyzer
        from models.classifier import FormationClassifier

        from src.heatmap import HeatmapGenerator
        from src.highlight_clipper import HighlightClipper
        from src.homography import PitchHomography
        from src.team_assignment import TeamAssigner

        # Prefer the football-specific detector (player/goalkeeper/referee/ball)
        # over generic COCO, which lumps everyone into "person". Falls back
        # automatically when no Roboflow key or package is present.
        self._detector = None
        if self.config.use_roboflow_detector:
            try:
                from src.roboflow_models import RoboflowPlayerDetector
                rf = RoboflowPlayerDetector(confidence=self.config.detector_conf)
                if rf.available:
                    self._detector = rf
                    logger.info("Using Roboflow football detector")
            except Exception as exc:
                logger.warning("Roboflow detector unavailable: %s", exc)

        if self._detector is None:
            self._detector = Detector(
                confidence_threshold=self.config.detector_conf,
                imgsz=self.config.detector_imgsz,
            )

        # Longer buffer + faster confirmation: broadcast players are briefly
        # occluded by each other constantly, and a 3-hit confirmation made the
        # reported player count oscillate instead of settling near 22.
        self._tracker = Tracker(track_buffer=60)
        self._classifier = FormationClassifier(device="cpu")  # cpu for compat
        self._play_analyzer = PlayAnalyzer()

        # Pixel->pitch mapping + team colouring for the tactical view. The
        # homography starts uncalibrated (linear frame->pitch fallback) until
        # the user calibrates 4+ points via /session/calibrate.
        fw = self.config.frame_width or 1280
        fh = self.config.frame_height or 720
        if self._homography is None:
            self._homography = PitchHomography(frame_w=fw, frame_h=fh)
        else:
            self._homography.set_frame_fallback(fw, fh)
        self._team_assigner = TeamAssigner()

        # Automatic pitch calibration from detected landmarks — replaces
        # manual 4-point clicking and keeps up as the camera pans/zooms.
        self._pitch_detector = None
        self._autocal = None
        if self.config.use_roboflow_pitch:
            try:
                from src.roboflow_models import RoboflowPitchDetector
                from src.pitch_keypoints import AutoCalibrator
                pd = RoboflowPitchDetector()
                if pd.available:
                    self._pitch_detector = pd
                    self._autocal = AutoCalibrator(
                        every_n_frames=self.config.autocalibrate_every
                    )
                    logger.info("Automatic pitch calibration enabled")
            except Exception as exc:
                logger.warning("Auto-calibration unavailable: %s", exc)

        # Embedding-based situation retrieval.
        from src.state_embedding import StateEmbedder, SituationStore
        from src.homography import PITCH_LENGTH, PITCH_WIDTH
        self._embedder = StateEmbedder(pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH)
        self._situations = SituationStore(dim=self._embedder.dim)
        self._heatmap = HeatmapGenerator()
        self._clipper = HighlightClipper(
            fps=self.config.coach_fps,
            frame_width=self.config.frame_width or 1280,
            frame_height=self.config.frame_height or 720,
        )

        if self.config.use_demo_source:
            from src.video_source import DemoSource
            self._source = DemoSource()
        else:
            from src.video_source import ScreenCaptureSource
            self._source = ScreenCaptureSource(
                display_index=self.config.display_index,
                window_id=self.config.window_id,
            )

    # ------------------------------------------------------------------
    # Internal: main pipeline loop
    # ------------------------------------------------------------------

    async def _run(self):
        """Main pipeline coroutine — reads frames, runs ML, broadcasts."""
        coach_interval = 1.0 / self.config.coach_fps
        fan_interval = 1.0 / self.config.fan_fps

        last_coach_send = 0.0
        last_fan_send = 0.0

        loop = asyncio.get_event_loop()

        while self._running:
            t_start = time.monotonic()

            # Grab frame in executor (blocking I/O)
            frame = await loop.run_in_executor(None, self._source.get_frame)
            if frame is None:
                await asyncio.sleep(0.02)
                continue

            self.frame_id += 1

            # Run detection + tracking in executor
            analysis = await loop.run_in_executor(
                None, self._process_frame, frame
            )

            now = time.monotonic()

            # Broadcast to coach clients at coach_fps
            if now - last_coach_send >= coach_interval and self._coach_clients:
                payload = self._build_coach_payload(frame, analysis)
                self._last_coach_payload = payload
                await self._broadcast(self._coach_clients, payload)
                last_coach_send = now

            # Broadcast to fan clients at fan_fps
            if now - last_fan_send >= fan_interval and self._fan_clients:
                payload = self._build_fan_payload(analysis)
                self._last_fan_payload = payload
                await self._broadcast(self._fan_clients, payload)
                last_fan_send = now

            # Throttle to avoid pegging CPU when no clients connected
            elapsed = time.monotonic() - t_start
            sleep_for = max(0, (1.0 / 30) - elapsed)
            await asyncio.sleep(sleep_for)

    def _process_frame(self, frame: np.ndarray) -> Dict:
        """Blocking ML processing — detect, track, classify, analyze."""
        h, w = frame.shape[:2]

        # Resize if configured
        if self.config.frame_width and self.config.frame_height:
            frame_small = cv2.resize(
                frame, (self.config.frame_width, self.config.frame_height)
            )
        else:
            frame_small = frame

        # Segment the playing surface so the crowd in the stands can't be
        # mistaken for players — on a wide stadium shot they outnumber them.
        field = None
        if self.config.use_field_mask:
            try:
                from src.field_mask import build_field_mask
                field = build_field_mask(frame_small)
            except Exception:
                field = None

        # Detection (field-masked)
        detections = self._detector.detect(frame_small, field_mask=field)

        # Keep the ball out of the tracker: it is not a player, and letting it
        # into the track set corrupts team shape, compactness and pitch
        # control. Track it separately for the overlay.
        is_ball = getattr(self._detector, "is_ball", lambda _cid: False)
        ball_dets = [d for d in detections if is_ball(d.class_id)]
        player_dets = [d for d in detections if not is_ball(d.class_id)]
        self._last_ball = (
            max(ball_dets, key=lambda d: d.confidence) if ball_dets else None
        )

        # Tracking (players only)
        tracks = self._tracker.update(player_dets, frame_small)

        # Draw bounding boxes onto frame_small (coach gets annotated view)
        annotated = self._tracker.draw_tracks(frame_small.copy(), tracks)

        # Formation (demo heuristic)
        confirmed = [t for t in tracks if t.is_confirmed]

        # Football analysis: assign teams by jersey colour, then map each
        # player from pixel space to pitch metres via the homography. This is
        # what turns the feed into a top-down tactical view.
        # Automatic calibration: locate pitch landmarks and re-solve the
        # homography periodically, so positions stay correct as the camera
        # pans and zooms without anyone clicking anything.
        if self._autocal is not None and self._pitch_detector is not None:
            if self.frame_id % self.config.autocalibrate_every == 0:
                try:
                    kps = self._pitch_detector.keypoints(frame_small)
                    if not kps:
                        self._autocal.last_info = {
                            "error": "pitch model returned no keypoints"
                        }
                    else:
                        h, w = frame_small.shape[:2]
                        auto = self._autocal.maybe_update(self.frame_id, kps, w, h)
                        if auto is not None:
                            self._homography = auto
                except Exception as exc:
                    # Surface the reason instead of silently staying uncalibrated.
                    self._autocal.last_info = {
                        "error": f"{type(exc).__name__}: {exc}"[:200]
                    }

        if self._team_assigner is not None:
            self._team_assigner.assign(confirmed, frame_small)
        if self._homography is not None:
            for t in confirmed:
                # Map the player's FEET, not the bbox centre. The homography
                # maps the ground plane, and a player's contact point with it
                # is the bottom of their box — using the torso centre placed
                # everyone several metres further downfield than they stood.
                x1, _y1, x2, y2 = t.bbox
                feet_x = (x1 + x2) / 2.0
                t.pitch_xy = self._homography.to_pitch(feet_x, y2)
            self._capture_situation(confirmed)

        formation_result = self._classifier.demo_classify(
            confirmed,
            frame_width=self.config.frame_width or w,
            frame_height=self.config.frame_height or h,
        )

        # Update formation frequency counter
        fname = formation_result.formation.value
        self.formation_freq[fname] = self.formation_freq.get(fname, 0) + 1

        # Play analysis
        play_result = self._play_analyzer.update(
            confirmed,
            frame_id=self.frame_id,
            frame_w=self.config.frame_width or w,
            frame_h=self.config.frame_height or h,
        )

        # Trigger fan prediction window on snap + highlight clip
        if play_result.snap_detected:
            self.play_count += 1
            self._predict_window_open = True
            self._predict_window_end = time.time() + 5.0  # 5-second window
            self._clipper.trigger_clip()

        # Push frame to ring buffer and check for completed clip
        self._clipper.push_frame(frame)
        clip_name = self._clipper.tick()
        if clip_name:
            self.set_last_highlight(clip_name)

        # Accumulate heatmap
        self._heatmap.add_tracks(
            confirmed,
            frame_w=self.config.frame_width or w,
            frame_h=self.config.frame_height or h,
        )

        # Close prediction window if expired
        if self._predict_window_open and time.time() > self._predict_window_end:
            self._predict_window_open = False

        return {
            "annotated": annotated,
            "tracks": confirmed,
            "formation": formation_result,
            "play": play_result,
            "player_count": len(confirmed),
        }

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def _build_coach_payload(self, raw_frame: np.ndarray, analysis: Dict) -> Dict:
        annotated = analysis["annotated"]
        formation = analysis["formation"]
        play = analysis["play"]

        return {
            "type": "coach",
            "frame_id": self.frame_id,
            "frame_b64": self._encode_frame(annotated),
            "formation": {
                "name": _format_formation(formation.formation.value),
                "confidence": round(formation.confidence, 3),
                "top_k": [
                    {"name": _format_formation(f.value), "confidence": round(c, 3)}
                    for f, c in formation.top_k_predictions
                ],
            },
            "player_count": analysis["player_count"],
            "play_prediction": {
                "run": play.run_prob,
                "pass": play.pass_prob,
            },
            "alerts": play.alerts,
            "snap_detected": play.snap_detected,
            "play_count": self.play_count,
            "tendency": self._play_analyzer.tendency_stats(),
            "pitch": self._build_pitch_block(analysis["tracks"]),
        }

    def _build_pitch_block(self, tracks: List) -> Dict:
        """
        Top-down tactical data: every player in pitch metres + team colour.

        The frontend draws a 105x68 m pitch and plots each player at
        (x, y). `calibrated` tells the UI whether these are true homography
        coordinates or the linear fallback (so it can show a "calibrate" hint).
        """
        from src.homography import PITCH_LENGTH, PITCH_WIDTH

        players = []
        roles = {"player": 0, "goalkeeper": 0, "referee": 0}
        calibrated_now = bool(self._homography and self._homography.is_calibrated)
        for t in tracks:
            if t.pitch_xy is None:
                continue
            # With a real calibration we know where the pitch is, so anyone
            # whose feet land outside it (touchline staff, camera operators,
            # substitutes) can be rejected geometrically rather than by colour.
            if calibrated_now and not self._homography.on_pitch(*t.pitch_xy, margin=1.0):
                continue
            x, y = self._homography.clamp_to_pitch(*t.pitch_xy)
            role = str(getattr(t, "class_name", "player")).lower()
            if role not in roles:
                role = "player"
            roles[role] += 1
            players.append({
                "id": t.track_id,
                # Officials belong to neither side; tagging them with a team
                # inflated one team's count and skewed shape/pitch-control.
                "team": t.team if role == "player" else -1,
                "role": role,
                "x": round(x, 2),
                "y": round(y, 2),
            })

        calibrated = bool(self._homography and self._homography.is_calibrated)

        # Count teams over outfield players only — see the note above.
        team_counts = {0: 0, 1: 0, -1: 0}
        for p in players:
            if p["role"] == "player":
                team_counts[p["team"]] = team_counts.get(p["team"], 0) + 1

        # Structured football concepts (centroids, compactness, hull, pitch
        # control, pressing) derived from the pitch-metre positions.
        from src.tactical_metrics import compute_metrics
        concepts = compute_metrics(players, PITCH_LENGTH, PITCH_WIDTH)

        return {
            "calibrated": calibrated,
            "reproj_error_m": round(self._homography.reprojection_error, 2)
            if calibrated else None,
            "pitch_length": PITCH_LENGTH,
            "pitch_width": PITCH_WIDTH,
            "players": players,
            "team_counts": {str(k): v for k, v in team_counts.items()},
            "roles": roles,
            "concepts": concepts,
            "situations_stored": len(self._situations) if self._situations else 0,
            "ball": self._ball_pitch_xy(),
            "auto_calibrated": bool(self._autocal and self._autocal.calibrations),
            "calibration_info": dict(self._autocal.last_info) if self._autocal else {},
        }

    def _ball_pitch_xy(self) -> Optional[Dict]:
        """The tracked ball's position in pitch metres, if one was detected."""
        if self._last_ball is None or self._homography is None:
            return None
        cx, cy = self._last_ball.center
        x, y = self._homography.clamp_to_pitch(*self._homography.to_pitch(cx, cy))
        return {"x": round(x, 2), "y": round(y, 2),
                "confidence": round(self._last_ball.confidence, 3)}

    def calibrate_homography(self, correspondences: List[Dict]) -> Dict:
        """
        Calibrate the pixel->pitch homography from clicked correspondences.

        Args:
            correspondences: list of {"image": [px, py], "pitch": [x_m, y_m]}
                OR {"image": [px, py], "landmark": "center"} entries.

        Returns:
            {"ok": bool, "reproj_error_m": float, "n_points": int, ...}
        """
        if self._homography is None:
            from src.homography import PitchHomography
            fw = self.config.frame_width or 1280
            fh = self.config.frame_height or 720
            self._homography = PitchHomography(frame_w=fw, frame_h=fh)

        from src.homography import PITCH_LANDMARKS

        image_pts, pitch_pts = [], []
        for c in correspondences:
            image_pts.append(tuple(c["image"]))
            if "landmark" in c:
                pitch_pts.append(PITCH_LANDMARKS[c["landmark"]])
            else:
                pitch_pts.append(tuple(c["pitch"]))

        try:
            err = self._homography.calibrate(image_pts, pitch_pts)
        except (ValueError, KeyError) as exc:
            return {"ok": False, "error": str(exc)}

        return {
            "ok": True,
            "reproj_error_m": round(err, 3),
            "n_points": len(image_pts),
            "calibrated": True,
        }

    # ------------------------------------------------------------------
    # Situation retrieval ("find similar situations")
    # ------------------------------------------------------------------
    def _capture_situation(self, tracks: List) -> None:
        """Record the current game-state as a searchable snapshot."""
        players = [
            {"id": t.track_id, "team": t.team,
             "x": round(t.pitch_xy[0], 2), "y": round(t.pitch_xy[1], 2)}
            for t in tracks
            if t.pitch_xy is not None
        ]
        self._last_pitch_players = players

        if self._situations is None or self._embedder is None:
            return
        # Only snapshot periodically, and only when both teams are on the pitch
        # (a one-team frame isn't a meaningful "situation" to match against).
        if self.frame_id % max(1, self.config.snapshot_every) != 0:
            return
        teams = {p["team"] for p in players}
        if 0 not in teams or 1 not in teams:
            return

        vec = self._embedder.embed(players)
        self._situations.add(vec, {
            "frame_id": self.frame_id,
            "t": round(time.time() - self.game_start, 2),
            "players": players,
        })

    def find_similar(self, k: int = 4, exclude_recent_s: float = 3.0) -> Dict:
        """Retrieve the past game-states most similar to the current one."""
        if self._situations is None or self._embedder is None:
            return {"ok": False, "error": "Retrieval not initialised"}
        if not self._last_pitch_players:
            return {"ok": False, "error": "No players on the pitch yet"}
        teams = {p["team"] for p in self._last_pitch_players}
        if 0 not in teams or 1 not in teams:
            return {"ok": False, "error": "Need both teams on the pitch to match"}

        vec = self._embedder.embed(self._last_pitch_players)
        now_t = round(time.time() - self.game_start, 2)
        hits = self._situations.query(
            vec, k=k, exclude_within_s=exclude_recent_s, query_time_s=now_t
        )
        return {
            "ok": True,
            "store_size": len(self._situations),
            "query": {"t": now_t, "players": self._last_pitch_players},
            "results": hits,
        }

    def _build_fan_payload(self, analysis: Dict) -> Dict:
        formation = analysis["formation"]
        play = analysis["play"]
        tracks = analysis["tracks"]

        # Normalized player positions for formation diagram
        fw = self.config.frame_width or 1280
        fh = self.config.frame_height or 720
        positions = [
            {"x": round(t.center[0] / fw, 4), "y": round(t.center[1] / fh, 4)}
            for t in tracks
        ]

        # Formation frequency for fan stats
        total_frames = max(sum(self.formation_freq.values()), 1)
        freq = {
            _format_formation(k): round(v / total_frames, 3)
            for k, v in sorted(
                self.formation_freq.items(), key=lambda x: -x[1]
            )[:5]
        }

        predict_secs = 0
        if self._predict_window_open:
            predict_secs = max(0, int(self._predict_window_end - time.time()))

        return {
            "type": "fan",
            "frame_id": self.frame_id,
            "formation_name": _format_formation(formation.formation.value),
            "formation_positions": positions,
            "stats": {
                "plays_analyzed": self.play_count,
                "formation_freq": freq,
                "run_tendency": self._play_analyzer.tendency_stats().get("run", 0.0),
                "pass_tendency": self._play_analyzer.tendency_stats().get("pass", 0.0),
            },
            "predict_window": {
                "open": self._predict_window_open,
                "seconds_remaining": predict_secs,
            },
            "last_highlight": self._last_highlight,
            "alerts": play.alerts,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _encode_frame(self, frame: np.ndarray) -> str:
        """JPEG-encode frame and return as base64 string."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        _, buf = cv2.imencode(".jpg", frame, encode_params)
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    async def _broadcast(clients: Set[Any], payload: Dict):
        """Send JSON payload to all registered WebSocket clients."""
        if not clients:
            return
        msg = json.dumps(payload)
        dead = set()
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        clients -= dead

    def set_last_highlight(self, clip_path: str):
        """Called by highlight clipper to notify fans of a new clip."""
        self._last_highlight = clip_path


def _format_formation(raw: str) -> str:
    """Convert Formation enum value to human-readable label."""
    mapping = {
        "shotgun": "Shotgun",
        "shotgun_spread": "Shotgun Spread",
        "shotgun_trips": "Shotgun Trips",
        "shotgun_empty": "Shotgun Empty",
        "i_formation": "I-Formation",
        "pro_set": "Pro Set",
        "single_back": "Single Back",
        "pistol": "Pistol",
        "wildcat": "Wildcat",
        "goal_line": "Goal Line",
        "jumbo": "Jumbo",
        "defense_4_3": "4-3 Defense",
        "defense_3_4": "3-4 Defense",
        "nickel": "Nickel",
        "dime": "Dime",
        "prevent": "Prevent",
        "unknown": "—",
    }
    return mapping.get(raw, raw.replace("_", " ").title())
