"""
FootballVision Engine — real-time adaptation of roboflow/sports

This is deliberately an adaptation of the reference implementation at
https://github.com/roboflow/sports (`examples/soccer/main.py`) rather than a
reimplementation. It uses their weights, their `TeamClassifier`, their
`ViewTransformer` and `SoccerPitchConfiguration`, because that geometry is
proven. Team assignment defaults to this project's instant local kit-colour
classifier, with the reference SigLIP model available as an optional backend.

Reference pipeline (per frame):
    player model  -> ball / goalkeeper / player / referee detections
    pitch model   -> 32 pitch landmarks
    ByteTrack     -> persistent player IDs
    TeamClassifier-> SigLIP embedding + UMAP + KMeans over player crops
    ViewTransformer -> image pixels to pitch coordinates

The one structural difference: the reference makes a first pass over the whole
video to collect crops before fitting `TeamClassifier`. A live stream has no
"whole video", so crops are accumulated from the opening frames and the
classifier is fitted once enough have been gathered. Until then players are
reported with an unknown team rather than a guessed one.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = _ROOT / "weights"
_CACHE_DIR = Path(os.environ.get("FOOTBALLVISION_CACHE_DIR", _ROOT / ".cache"))
try:
    (_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
    (_CACHE_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(_CACHE_DIR / "ultralytics"))
except OSError:
    # A read-only deployment can still use each library's platform default.
    logger.debug("Could not create local library cache", exc_info=True)

PLAYER_MODEL = WEIGHTS_DIR / "football-player-detection.pt"
PITCH_MODEL = WEIGHTS_DIR / "football-pitch-detection.pt"
BALL_MODEL = WEIGHTS_DIR / "football-ball-detection.pt"

# Class ids of the reference player model.
BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID = 0, 1, 2, 3
CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

# Role stabilisation.
#
# The detector classifies every box independently on every frame, so a track
# whose identity is never in doubt to a human still flickers between roles.
# Measured over 247 frames of the reference match: 18 role changes across 4
# tracks, including the goalkeeper alternating player/goalkeeper for its whole
# life. The tracker already provides a stable id, so the role is decided from
# the track's accumulated evidence instead of the current frame.
#
# A plain majority vote is not enough on its own. The goalkeeper class is
# systematically under-detected — keepers are occluded by the goal frame, are
# often in poses no outfield player adopts, and wear a kit resembling neither
# team — so the measured split for the one keeper in frame was 56% player /
# 44% goalkeeper. A majority vote entrenches "player" and the keeper is lost.
# The football fact breaks the tie: only a goalkeeper holds a position that
# deep for the whole match.
GK_MIN_VOTE_SHARE = 0.20
# Penalty area is 16.5 m. The band is widened to 20 m to absorb the measured
# ~1 m reprojection error and the keeper's habit of holding the edge of the
# area. Position alone never promotes a track — it only breaks a tie for one
# that already carries real goalkeeper evidence, so a deep defender is safe.
GK_MAX_DEPTH_M = 20.0
PITCH_LENGTH_M = 105.0

# Keypoint confidence required before a landmark is trusted for homography —
# the model emits all 32 every frame, scoring unseen ones near zero.
KEYPOINT_CONFIDENCE = 0.5

# Crops to accumulate before fitting the team classifier.
TEAM_FIT_CROPS = 120
LOCAL_TEAM_FIT_CROPS = 48
POSITION_SMOOTHING_ALPHA = 0.38
BALL_SMOOTHING_ALPHA = 0.48
BALL_HOLD_FRAMES = 12


def smooth_point(
    previous: Optional[Tuple[float, float]],
    current: Tuple[float, float],
    alpha: float,
) -> Tuple[float, float]:
    """Exponential smoothing for projected pitch positions."""
    if previous is None:
        return float(current[0]), float(current[1])
    alpha = min(1.0, max(0.0, float(alpha)))
    return (
        alpha * float(current[0]) + (1.0 - alpha) * float(previous[0]),
        alpha * float(current[1]) + (1.0 - alpha) * float(previous[1]),
    )


@dataclass
class FrameResult:
    """Everything derived from one frame."""
    frame_id: int
    annotated: Optional[np.ndarray] = None
    players: List[Dict] = field(default_factory=list)   # id, team, role, x, y (metres)
    ball: Optional[Dict] = None
    calibrated: bool = False
    #: True only when a fresh homography was solved on THIS frame. `calibrated`
    #: alone cannot tell you that: a failed solve keeps the previous transformer
    #: (see `_pitch_transformer`), so `calibrated` stays True on frames that
    #: were carried rather than solved. Reporting a calibration rate without
    #: this split overstates what was measured.
    homography_solved: bool = False
    n_keypoints: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    team_ready: bool = False
    team_status: str = "learning"
    intelligence: Dict = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    #: Why nothing was found, when nothing was found. Without this the app
    #: silently reports an empty pitch whether the captured window shows a
    #: match, a spreadsheet, or a desktop — which is indistinguishable from
    #: a broken detector and wastes enormous debugging time.
    source_status: str = "ok"
    green_fraction: float = 0.0


def looks_like_football(frame: np.ndarray) -> Tuple[bool, float]:
    """
    Cheap check that the captured frame actually shows a pitch.

    Returns (is_football, green_fraction). A broadcast wide shot is heavily
    grass-dominated; a desktop, browser or paused title card is not.
    """
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = ((hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 95)
             & (hsv[:, :, 1] > 40)).mean()
    return bool(green >= 0.15), float(green)


class FootballEngine:
    """Real-time football analysis over arbitrary BGR frames."""

    def __init__(
        self,
        device: str = "mps",
        # Defaults chosen by scripts/optimize_sweep.py on 2026-08-09, on the
        # fastest configuration that holds player recall at 0.99 of the
        # 1280px baseline. See benchmarks/optimize_sweep.json.
        player_imgsz: int = 960,
        enable_team_classifier: bool = True,
        calibrate_every: int = 15,
        team_refresh_every: int = 90,
        ball_every: int = 5,
        ball_imgsz: int = 640,
        half: bool = False,
        team_backend: Optional[str] = None,
    ):
        """
        Args:
            calibrate_every: frames between pitch-keypoint solves. A broadcast
                camera moves smoothly, so re-solving every frame costs ~70-300 ms
                to produce a near-identical homography.
            team_refresh_every: frames after which a track's cached team is
                re-checked. A player's team never changes, so running SigLIP on
                every crop every frame was the single biggest cost (~600 ms/frame).
            ball_every: frames between dedicated ball passes. The ball model is
                a SECOND full inference over the frame, run whenever the primary
                detector misses the ball — which is most frames, because the ball
                is occluded, airborne or motion-blurred much of the time. Its cost
                was previously untimed and invisible in the breakdown.
            ball_imgsz: inference size for the ball pass. 0 reuses player_imgsz.
                The ball is one small object; it does not need the same
                resolution as twenty-two players.
            half: run the detectors in fp16.
        """
        self.device = device
        self.player_imgsz = player_imgsz
        self.enable_team_classifier = enable_team_classifier
        self.calibrate_every = max(1, calibrate_every)
        self.team_refresh_every = max(1, team_refresh_every)
        self.ball_every = max(1, ball_every)
        self.ball_imgsz = int(ball_imgsz or 0)
        self.half = bool(half)
        self.team_backend = (team_backend or os.environ.get(
            "FOOTBALLVISION_TEAM_BACKEND", "local"
        )).strip().lower()
        if self.team_backend not in {"local", "siglip"}:
            raise ValueError("team_backend must be 'local' or 'siglip'")

        self._player_model = None
        self._pitch_model = None
        self._ball_model = None
        self._tracker = None
        self._team_classifier = None
        self._pitch_config = None

        self._crops: List[np.ndarray] = []
        self._team_ready = False
        self._solved_this_frame = False
        self._frame_id = 0

        # track_id -> (team_id, frame_it_was_classified)
        self._team_cache: Dict[int, Tuple[int, int]] = {}
        # tracker_id -> {class_id: summed confidence}. Confidence-weighted so a
        # single low-confidence misread cannot outvote steady evidence.
        self._role_votes: Dict[int, Dict[int, float]] = {}
        # tracker_id -> recent pitch x in metres, for the goalkeeper tie-break.
        self._track_x: Dict[int, deque] = {}
        self._smoothed_positions: Dict[int, Tuple[float, float]] = {}
        self._last_ball: Optional[Tuple[float, float]] = None
        self._last_ball_frame: int = 0
        # Homography reused between calibration frames.
        self._transformer = None
        self._n_keypoints = 0

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load models. Blocking — call once, off the event loop."""
        import supervision as sv
        import torch
        from ultralytics import YOLO
        from sports.configs.soccer import SoccerPitchConfiguration

        missing = [p.name for p in (PLAYER_MODEL, PITCH_MODEL) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing weights {missing} in {WEIGHTS_DIR}. "
                "Run scripts/download_weights.sh"
            )

        if self.device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS is unavailable; falling back to CPU inference")
            self.device = "cpu"
        elif self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; falling back to CPU inference")
            self.device = "cpu"

        logger.info("Loading football models on %s", self.device)
        self._player_model = YOLO(str(PLAYER_MODEL)).to(device=self.device)
        self._pitch_model = YOLO(str(PITCH_MODEL)).to(device=self.device)
        # The player model's own ball class effectively never fires on wide
        # broadcast footage (measured: 0 hits across sampled frames), which is
        # why the reference implementation ships a dedicated ball detector.
        if BALL_MODEL.exists():
            self._ball_model = YOLO(str(BALL_MODEL)).to(device=self.device)
        # Same tracker settings as the reference implementation.
        self._tracker = sv.ByteTrack(minimum_consecutive_frames=3)
        self._pitch_config = SoccerPitchConfiguration()
        if self.enable_team_classifier and self.team_backend == "siglip":
            # Load the public embedding model during the visible startup phase,
            # never in the middle of live analysis. It can be cached ahead of
            # time without a login or token.
            try:
                from sports.common.team import TeamClassifier
                logger.info("Loading optional SigLIP team model")
                self._team_classifier = TeamClassifier(device=self.device)
            except Exception as exc:
                logger.warning("SigLIP unavailable; using local kit colours: %s", exc)
                self.team_backend = "local"
                self._team_classifier = None
        self.reset_session()
        logger.info("Football models ready")

    def reset_session(self) -> None:
        """Clear match-specific temporal state while retaining ML weights."""
        import supervision as sv

        self._tracker = sv.ByteTrack(minimum_consecutive_frames=3)
        self._crops = []
        self._team_ready = False
        self._solved_this_frame = False
        self._frame_id = 0
        self._team_cache = {}
        self._role_votes = {}
        self._track_x = {}
        self._smoothed_positions = {}
        self._last_ball = None
        self._last_ball_frame = 0
        self._transformer = None
        self._n_keypoints = 0
        if self.team_backend == "local":
            self._team_classifier = None

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, annotate: bool = True) -> FrameResult:
        """Run the full pipeline on one frame."""
        import time

        import supervision as sv

        self._frame_id += 1
        res = FrameResult(frame_id=self._frame_id)
        timings = {}

        # Bail out early — and say so — when the captured source plainly is
        # not football. Running the models on a browser window wastes ~450 ms
        # a frame and reports an empty pitch that looks like a broken system.
        is_football, green = looks_like_football(frame)
        res.green_fraction = round(green, 3)
        if not is_football:
            res.source_status = (
                f"no pitch in captured source (only {green:.0%} grass) — "
                "is the match visible on the selected window/screen?"
            )
            res.counts = {"player": 0, "goalkeeper": 0, "referee": 0, "ball": 0}
            res.annotated = frame if annotate else None
            return res

        # ---- detection -------------------------------------------------
        t0 = time.time()
        det_result = self._player_model(
            frame, imgsz=self.player_imgsz, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(det_result)
        timings["detect"] = (time.time() - t0) * 1000

        ball = detections[detections.class_id == BALL_ID]
        people = detections[detections.class_id != BALL_ID]

        # Dedicated ball pass. Detection is intermittent by nature (the ball is
        # occluded, airborne or motion-blurred much of the time), so the last
        # known position is held briefly rather than flickering to nothing.
        t_ball = time.time()
        ball_due = (self.ball_every <= 1
                    or self._frame_id % self.ball_every == 0)
        if self._ball_model is not None and len(ball) == 0 and ball_due:
            try:
                br = self._ball_model(
                    frame, imgsz=self.ball_imgsz or self.player_imgsz, verbose=False
                )[0]
                bd = sv.Detections.from_ultralytics(br)
                if len(bd):
                    keep = int(np.argmax(bd.confidence)) if bd.confidence is not None else 0
                    ball = bd[keep:keep + 1]
            except Exception as exc:
                logger.debug("ball model failed: %s", exc)
        timings["ball"] = (time.time() - t_ball) * 1000

        # ---- tracking (people only; the ball is not a player) -----------
        t0 = time.time()
        people = self._tracker.update_with_detections(people)
        timings["track"] = (time.time() - t0) * 1000

        # Decide each role from the track's history rather than this frame
        # alone, otherwise identities flicker frame to frame.
        people = self._stabilise_roles(people)

        players = people[people.class_id == PLAYER_ID]
        goalkeepers = people[people.class_id == GOALKEEPER_ID]
        referees = people[people.class_id == REFEREE_ID]

        res.counts = {
            "player": len(players),
            "goalkeeper": len(goalkeepers),
            "referee": len(referees),
            "ball": len(ball),
        }

        # ---- team classification ---------------------------------------
        t0 = time.time()
        team_ids = self._resolve_teams(frame, players)
        timings["team"] = (time.time() - t0) * 1000
        res.team_ready = self._team_ready
        res.team_status = (
            f"ready-{self.team_backend}" if self._team_ready
            else f"learning-{self.team_backend}" if self.enable_team_classifier
            else "unavailable"
        )

        # ---- pitch calibration -----------------------------------------
        t0 = time.time()
        transformer, n_kp = self._pitch_transformer(frame)
        timings["pitch"] = (time.time() - t0) * 1000
        res.n_keypoints = n_kp
        res.calibrated = transformer is not None
        res.homography_solved = getattr(self, "_solved_this_frame", False)

        # ---- project to pitch coordinates ------------------------------
        if transformer is not None:
            res.players = self._project(
                transformer, players, goalkeepers, referees, team_ids
            )
            if len(ball):
                bxy = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pt = transformer.transform_points(bxy[:1].astype(np.float32))
                if len(pt):
                    raw = (float(pt[0][0]) / 100.0, float(pt[0][1]) / 100.0)
                    self._last_ball = smooth_point(
                        self._last_ball, raw, BALL_SMOOTHING_ALPHA
                    )
                    self._last_ball_frame = self._frame_id
                    res.ball = {
                        "x": round(self._last_ball[0], 2),
                        "y": round(self._last_ball[1], 2),
                        "stale": False,
                    }
            if (res.ball is None and self._last_ball is not None
                    and self._frame_id - self._last_ball_frame <= BALL_HOLD_FRAMES):
                res.ball = {
                    "x": round(self._last_ball[0], 2),
                    "y": round(self._last_ball[1], 2),
                    "stale": True,
                    "age_frames": self._frame_id - self._last_ball_frame,
                }

        if annotate:
            res.annotated = self._annotate(
                frame, players, goalkeepers, referees, ball, team_ids
            )

        res.timings_ms = {k: round(v, 1) for k, v in timings.items()}
        return res

    # ------------------------------------------------------------------
    def _stabilise_roles(self, people):
        """Replace each detection's class with its track's accumulated verdict.

        The detector is re-run per frame and has no memory, so a box can be a
        player on one frame and a referee on the next. The tracker does have
        memory, so votes are accumulated per `tracker_id` and the winner is
        applied. Roles only get more certain as a track is seen for longer,
        which is the opposite of the current behaviour.
        """
        if len(people) == 0 or people.tracker_id is None:
            return people

        conf = (people.confidence if people.confidence is not None
                else np.ones(len(people), dtype=float))
        classes = np.asarray(people.class_id).copy()

        for i, tid in enumerate(people.tracker_id):
            tid = int(tid)
            votes = self._role_votes.setdefault(tid, {})
            observed = int(classes[i])
            votes[observed] = votes.get(observed, 0.0) + float(conf[i])
            classes[i] = self._decide_role(tid, votes)

        people.class_id = classes
        return people

    def _decide_role(self, tid: int, votes: Dict[int, float]) -> int:
        """Winning role for a track: weighted vote, with a goalkeeper prior."""
        total = sum(votes.values()) or 1.0
        if votes.get(GOALKEEPER_ID, 0.0) / total >= GK_MIN_VOTE_SHARE:
            # Real goalkeeper evidence exists. Confirm it against position
            # before overriding the vote, so a mis-detected outfielder is not
            # promoted just because the detector blinked once.
            xs = self._track_x.get(tid)
            if xs:
                med = float(np.median(xs))
                if med <= GK_MAX_DEPTH_M or med >= PITCH_LENGTH_M - GK_MAX_DEPTH_M:
                    return GOALKEEPER_ID
        return max(votes, key=votes.get)

    def _note_track_x(self, tid: int, x: float) -> None:
        """Record a track's pitch x so the goalkeeper rule has evidence."""
        buf = self._track_x.get(tid)
        if buf is None:
            buf = self._track_x[tid] = deque(maxlen=60)
        buf.append(float(x))

    def _resolve_teams(self, frame: np.ndarray, players) -> np.ndarray:
        """Team id per player; -1 until the classifier has been fitted."""
        if not self.enable_team_classifier or len(players) == 0:
            return np.full(len(players), -1, dtype=int)

        crops = self._get_crops(frame, players)

        if not self._team_ready:
            # Accumulate crops from the opening frames, then fit once.
            self._crops.extend(crops)
            fit_crops = (LOCAL_TEAM_FIT_CROPS if self.team_backend == "local"
                         else TEAM_FIT_CROPS)
            if len(self._crops) >= fit_crops:
                try:
                    logger.info("Fitting team classifier on %d crops",
                                len(self._crops))
                    if self.team_backend == "local":
                        from src.team_assignment import CropTeamClassifier
                        tc = CropTeamClassifier()
                    else:
                        from sports.common.team import TeamClassifier
                        tc = self._team_classifier or TeamClassifier(device=self.device)
                    tc.fit(self._crops)
                    self._team_classifier = tc
                    self._team_ready = True
                    self._crops = []
                    logger.info("Team classifier ready")
                except Exception as exc:
                    logger.warning("Team classifier failed: %s", exc)
                    # Don't retry every frame if it's broken.
                    self.enable_team_classifier = False
            return np.full(len(players), -1, dtype=int)

        # A player does not change team mid-match, so a track only needs
        # classifying once. Only crops without a fresh cached answer are sent
        # through the classifier — this makes cost near-zero on frames where
        # every player is already known.
        track_ids = (players.tracker_id if players.tracker_id is not None
                     else np.full(len(players), -1))
        teams = np.full(len(players), -1, dtype=int)
        todo_idx, todo_crops = [], []

        for i, tid in enumerate(track_ids):
            tid = int(tid)
            cached = self._team_cache.get(tid)
            if (cached is not None
                    and self._frame_id - cached[1] < self.team_refresh_every):
                teams[i] = cached[0]
            else:
                todo_idx.append(i)
                todo_crops.append(crops[i])

        if todo_crops:
            try:
                pred = np.asarray(
                    self._team_classifier.predict(todo_crops), dtype=int)
                for slot, i in enumerate(todo_idx):
                    if slot < len(pred):
                        teams[i] = int(pred[slot])
                        self._team_cache[int(track_ids[i])] = (
                            int(pred[slot]), self._frame_id)
            except Exception as exc:
                logger.warning("Team prediction failed: %s", exc)

        # Keep the cache from growing without bound over a long match.
        if len(self._team_cache) > 500:
            cutoff = self._frame_id - self.team_refresh_every * 4
            self._team_cache = {
                k: v for k, v in self._team_cache.items() if v[1] >= cutoff
            }

        return teams

    @staticmethod
    def _get_crops(frame: np.ndarray, detections) -> List[np.ndarray]:
        import supervision as sv
        return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

    # ------------------------------------------------------------------
    def _pitch_transformer(self, frame: np.ndarray):
        """
        Homography from image pixels to pitch centimetres, or None.

        Re-solved only every `calibrate_every` frames and reused in between:
        a broadcast camera moves smoothly, so consecutive solves are nearly
        identical while each costs 70-300 ms.
        """
        import supervision as sv
        from sports.common.view import ViewTransformer

        self._solved_this_frame = False
        due = (self._frame_id % self.calibrate_every == 0
               or self._transformer is None)
        if not due:
            return self._transformer, self._n_keypoints

        result = self._pitch_model(frame, verbose=False)[0]
        kp = sv.KeyPoints.from_ultralytics(result)

        # On a failed solve keep the previous homography rather than dropping
        # to uncalibrated — a single bad frame (replay cut, close-up) should
        # not discard a good calibration.
        def _keep():
            return self._transformer, self._n_keypoints

        if kp.xy is None or len(kp.xy) == 0:
            return _keep()

        conf = kp.confidence[0] if kp.confidence is not None else None
        if conf is None:
            return _keep()
        mask = conf > KEYPOINT_CONFIDENCE
        n = int(mask.sum())
        if n < 4:
            return _keep()

        source = kp.xy[0][mask].astype(np.float32)
        target = np.array(self._pitch_config.vertices, dtype=np.float32)[mask]
        try:
            self._transformer = ViewTransformer(source=source, target=target)
            self._n_keypoints = n
        except ValueError:
            return _keep()
        self._solved_this_frame = True
        return self._transformer, self._n_keypoints

    # ------------------------------------------------------------------
    def _project(self, transformer, players, goalkeepers, referees, team_ids
                 ) -> List[Dict]:
        """Map everyone to pitch metres via their ground-contact point."""
        import supervision as sv

        out: List[Dict] = []

        def add(dets, role: str, teams=None):
            if len(dets) == 0:
                return
            xy = dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_xy = transformer.transform_points(xy.astype(np.float32))
            for i in range(len(dets)):
                tid = (int(dets.tracker_id[i])
                       if dets.tracker_id is not None else -1)
                team = int(teams[i]) if teams is not None and i < len(teams) else -1
                x_m = float(pitch_xy[i][0]) / 100.0
                y_m = float(pitch_xy[i][1]) / 100.0
                if tid >= 0:
                    x_m, y_m = smooth_point(
                        self._smoothed_positions.get(tid),
                        (x_m, y_m),
                        POSITION_SMOOTHING_ALPHA,
                    )
                    self._smoothed_positions[tid] = (x_m, y_m)
                # Feed the goalkeeper tie-break. Only outfield-capable roles
                # matter here: a referee is never promoted to goalkeeper.
                if tid >= 0 and role in ("player", "goalkeeper"):
                    self._note_track_x(tid, x_m)
                out.append({
                    "id": tid,
                    "role": role,
                    "team": team,
                    # config vertices are centimetres; report metres
                    "x": round(x_m, 2),
                    "y": round(y_m, 2),
                })

        add(players, "player", team_ids)
        add(goalkeepers, "goalkeeper")
        add(referees, "referee")

        # A goalkeeper wears a third colour, so kit clustering cannot assign
        # it directly. Associate it with the nearest outfield-team centroid.
        team_points = {
            team: np.asarray([[p["x"], p["y"]] for p in out
                              if p["role"] == "player" and p["team"] == team])
            for team in (0, 1)
        }
        centroids = {team: pts.mean(axis=0) for team, pts in team_points.items() if len(pts)}
        if len(centroids) == 2:
            for person in out:
                if person["role"] == "goalkeeper":
                    xy = np.asarray([person["x"], person["y"]])
                    person["team"] = min(centroids, key=lambda t: np.linalg.norm(xy - centroids[t]))

        if len(self._smoothed_positions) > 500:
            live = {p["id"] for p in out if p["id"] >= 0}
            self._smoothed_positions = {
                tid: xy for tid, xy in self._smoothed_positions.items() if tid in live
            }
        return out

    # ------------------------------------------------------------------
    def _annotate(self, frame, players, goalkeepers, referees, ball, team_ids):
        """
        Ellipse annotation in the style of the reference implementation.

        Ellipses are drawn at each player's feet rather than boxes around them:
        it is the football-broadcast convention, it does not hide the player,
        and it reads clearly when 22 of them overlap. Track IDs are deliberately
        omitted here because the top-down pitch already carries them, and label
        boxes at this density obscure more than they explain.
        """
        import supervision as sv

        annotated = frame.copy()
        palette = sv.ColorPalette.from_hex(
            ["#e5484d", "#3b82f6", "#f5c542", "#9aa0a6"]
        )
        ellipse = sv.EllipseAnnotator(color=palette, thickness=2)

        merged = sv.Detections.merge([players, goalkeepers, referees])
        if len(merged) == 0:
            return annotated

        # 0/1 = teams, 2 = officials and keepers, 3 = unresolved team.
        lookup = np.array(
            [(int(t) if int(t) in (0, 1) else 3) for t in team_ids]
            + [2] * len(goalkeepers)
            + [2] * len(referees)
        )
        annotated = ellipse.annotate(annotated, merged, custom_color_lookup=lookup)

        if len(ball):
            tri = sv.TriangleAnnotator(
                color=sv.Color.from_hex("#ffffff"), base=20, height=17
            )
            annotated = tri.annotate(annotated, ball)
        return annotated


if __name__ == "__main__":
    import sys

    import cv2

    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: python -m src.fv_engine <image>")
        raise SystemExit(1)

    frame = cv2.imread(path)
    eng = FootballEngine()
    eng.load()
    r = eng.process(frame)
    print(f"counts       : {r.counts}")
    print(f"calibrated   : {r.calibrated} ({r.n_keypoints} keypoints)")
    print(f"projected    : {len(r.players)} people")
    print(f"ball         : {r.ball}")
    print(f"timings (ms) : {r.timings_ms}")
    for p in r.players[:6]:
        print(f"   {p['role']:<11} id={p['id']:<5} team={p['team']:<2} "
              f"({p['x']:6.1f}, {p['y']:5.1f}) m")
