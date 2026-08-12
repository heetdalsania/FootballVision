"""
Roboflow Football Models for FootballVision

Two specialist models replace what a generic COCO detector cannot do:

  * **football-players-detection-3zvbc** — trained on broadcast football, with
    the classes that actually matter: `player`, `goalkeeper`, `referee`, `ball`.
    A COCO model lumps all of these into "person" and detects the ball only as
    the generic "sports ball" class.
  * **football-field-detection-f07vi** — a pose model emitting 32 pitch
    landmarks, which drives automatic homography (see `src/pitch_keypoints.py`)
    and removes manual 4-point calibration.

Both run **locally** via the `inference` package — the weights are downloaded
and cached on first use, so there is no per-frame API call, no rate limit, and
no network dependency once warm. The Roboflow API key is only used to
authorise that initial download.

Everything here fails soft: if the key or package is missing, loaders return
None and the pipeline falls back to the YOLO detector.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PLAYER_MODEL_ID = "football-players-detection-3zvbc/9"
PITCH_MODEL_ID = "football-field-detection-f07vi/14"

_ROOT = Path(__file__).resolve().parent.parent
_KEY_FILE = _ROOT / ".roboflow_key"


def load_api_key() -> Optional[str]:
    """Read the Roboflow key from $ROBOFLOW_API_KEY or the .roboflow_key file."""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key
    if _KEY_FILE.exists():
        for line in _KEY_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return None


def _load(model_id: str):
    """Load a Roboflow model locally, or None if unavailable."""
    key = load_api_key()
    if not key:
        logger.info("No Roboflow API key — skipping %s", model_id)
        return None
    try:
        from inference import get_model
    except ImportError:
        logger.info("`inference` package not installed — skipping %s", model_id)
        return None
    try:
        model = get_model(model_id=model_id, api_key=key)
        logger.info("Loaded Roboflow model %s", model_id)
        return model
    except Exception as exc:
        logger.warning("Could not load %s: %s", model_id, exc)
        return None


class RoboflowPlayerDetector:
    """
    Player/goalkeeper/referee/ball detector.

    Emits the same `Detection` objects as `src/detector.py`, so it is a drop-in
    for the pipeline, and exposes `is_person` / `is_ball` with the same meaning.
    """

    #: Class names this model produces.
    PERSON_CLASSES = {"player", "goalkeeper", "referee"}
    BALL_CLASSES = {"ball"}

    #: The model's fixed class order. Seeded up-front because `is_ball()` and
    #: `is_person()` are consulted on the very first frame — populating this
    #: lazily during detection misclassified everything until a class had been
    #: seen at least once.
    DEFAULT_CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

    def __init__(self, confidence: float = 0.30):
        self.confidence = confidence
        self.model = _load(PLAYER_MODEL_ID)
        self.class_names: dict = dict(self.DEFAULT_CLASS_NAMES)

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, frame: np.ndarray, field_mask=None) -> List:
        """Detect players/keepers/referees/ball, optionally field-masked."""
        from src.detector import Detection

        if self.model is None:
            return []
        try:
            result = self.model.infer(frame, confidence=self.confidence)[0]
        except Exception as exc:
            logger.warning("Roboflow inference failed: %s", exc)
            return []

        use_mask = field_mask is not None and getattr(field_mask, "is_usable", False)
        out: List[Detection] = []

        for p in getattr(result, "predictions", []) or []:
            name = str(getattr(p, "class_name", "")).lower()
            # inference returns centre-x/centre-y plus width/height.
            cx, cy = float(p.x), float(p.y)
            w, h = float(p.width), float(p.height)
            bbox = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

            if use_mask and name not in self.BALL_CLASSES:
                if not field_mask.bbox_on_field(bbox):
                    continue

            cid = int(getattr(p, "class_id", 0) or 0)
            # Trust the name the model reported over the seeded default.
            if name:
                self.class_names[cid] = name
            out.append(Detection(
                bbox=bbox,
                confidence=float(getattr(p, "confidence", 0.0)),
                class_id=cid,
                class_name=name,
            ))
        return out

    def is_person(self, class_id: int) -> bool:
        return self.class_names.get(class_id, "") in self.PERSON_CLASSES

    def is_ball(self, class_id: int) -> bool:
        return self.class_names.get(class_id, "") in self.BALL_CLASSES

    def is_referee(self, class_id: int) -> bool:
        return self.class_names.get(class_id, "") == "referee"

    def is_goalkeeper(self, class_id: int) -> bool:
        return self.class_names.get(class_id, "") == "goalkeeper"


class RoboflowPitchDetector:
    """Pitch-landmark detector feeding automatic homography."""

    def __init__(self, confidence: float = 0.30):
        self.confidence = confidence
        self.model = _load(PITCH_MODEL_ID)

    @property
    def available(self) -> bool:
        return self.model is not None

    def keypoints(self, frame: np.ndarray) -> Optional[List]:
        """Return the frame's 32 pitch landmarks, or None."""
        if self.model is None:
            return None
        try:
            result = self.model.infer(frame, confidence=self.confidence)[0]
        except Exception as exc:
            logger.warning("Pitch keypoint inference failed: %s", exc)
            return None
        preds = getattr(result, "predictions", None)
        if not preds:
            return None
        return getattr(preds[0], "keypoints", None)


if __name__ == "__main__":
    import sys
    import cv2

    logging.basicConfig(level=logging.INFO)
    print("API key found:", bool(load_api_key()))

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: python -m src.roboflow_models <image>")
        raise SystemExit(0)

    frame = cv2.imread(path)
    pd = RoboflowPlayerDetector()
    print("player model available:", pd.available)
    if pd.available:
        dets = pd.detect(frame)
        from collections import Counter
        print("detections:", dict(Counter(d.class_name for d in dets)))

    kd = RoboflowPitchDetector()
    print("pitch model available:", kd.available)
    if kd.available:
        kps = kd.keypoints(frame)
        if kps:
            good = [k for k in kps if float(getattr(k, "confidence", 0)) >= 0.5]
            print(f"keypoints: {len(kps)} total, {len(good)} confident")
            from src.pitch_keypoints import calibrate_from_keypoints
            homo, info = calibrate_from_keypoints(kps)
            print("auto-calibration:", info)
