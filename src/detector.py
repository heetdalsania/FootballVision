"""
Player Detection Module for NFL Vision App
Uses YOLOv8 for real-time player detection.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


@dataclass
class Detection:
    """Represents a single detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        """Get area of bounding box."""
        return self.width * self.height
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center": self.center
        }


class Detector:
    """
    YOLOv8-based player detector for NFL Vision.
    
    Detects players (person class) in video frames with configurable
    confidence thresholds and filtering options.
    
    Example:
        detector = Detector()
        detections = detector.detect(frame)
        annotated = detector.draw_detections(frame, detections)
    """
    
    # COCO class indices
    PERSON_CLASS_ID = 0
    BALL_CLASS_ID = 32  # COCO "sports ball"

    # Football-specific classes (for future fine-tuned models)
    FOOTBALL_CLASSES = {
        0: "player",
        1: "referee",
        2: "ball",
        3: "coach"
    }

    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.20,
                 iou_threshold: float = 0.45,
                 device: str = "mps",  # Apple Silicon
                 classes: Optional[List[int]] = None,
                 imgsz: int = 1280,
                 detect_ball: bool = True):
        """
        Initialize the detector.

        Defaults are tuned for **broadcast football**, where players are small
        and numerous. Three settings matter far more than anything else:

          * **model size** — the nano model misses most distant players. The
            default is a large model; override with FOOTBALLVISION_MODEL.
          * **confidence** — broadcast players are small and low-confidence;
            0.5 discards most of them, so the default is 0.20.
          * **imgsz** — YOLO's default 640 destroys small players on a wide
            shot. 1280 keeps them resolvable.

        Args:
            model_path: YOLO weights. Defaults to $FOOTBALLVISION_MODEL, else
                "yolo11x.pt" (auto-downloaded by ultralytics on first use).
            confidence_threshold: Minimum confidence to keep a detection.
            iou_threshold: IoU threshold for NMS.
            device: "mps" (Apple Silicon), "cuda", or "cpu".
            classes: Class IDs to detect. None = person (+ ball if detect_ball).
            imgsz: Inference resolution. Higher finds smaller players.
            detect_ball: Also detect the ball (COCO "sports ball").
        """
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "Ultralytics not available. "
                "Install with: pip install ultralytics"
            )

        import os

        if model_path is None:
            model_path = os.environ.get("FOOTBALLVISION_MODEL", "yolo11x.pt")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz
        self.detect_ball = detect_ball

        if classes is None:
            classes = [self.PERSON_CLASS_ID]
            if detect_ball:
                classes.append(self.BALL_CLASS_ID)
        self.classes = classes

        # Load model
        print(f"[Detector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)

        # Get class names from model
        self.class_names = self.model.names

        # A football-specific model (classes like ball/goalkeeper/player/
        # referee) does NOT share COCO's indices, so the COCO person/ball
        # filter would select the wrong classes — or nothing at all. Detect
        # that case and keep every class the model knows.
        names_lower = {str(v).lower() for v in self.class_names.values()}
        self.is_coco = "person" in names_lower
        if not self.is_coco:
            self.classes = list(self.class_names.keys())
            print(f"[Detector] Non-COCO model detected — using its own classes")

        # Which class ids count as a player vs the ball, for this model.
        self.person_ids = {
            k for k, v in self.class_names.items()
            if str(v).lower() in {"person", "player", "goalkeeper", "referee"}
        }
        self.ball_ids = {
            k for k, v in self.class_names.items()
            if str(v).lower() in {"sports ball", "ball"}
        }

        print(f"[Detector] Ready on device: {device} (imgsz={imgsz}, "
              f"conf={confidence_threshold})")
        print(f"[Detector] Detecting classes: "
              f"{[self.class_names[c] for c in self.classes]}")

    def is_ball(self, class_id: int) -> bool:
        return class_id in self.ball_ids

    def is_person(self, class_id: int) -> bool:
        return class_id in self.person_ids

    def detect(self, frame: np.ndarray, field_mask=None) -> List[Detection]:
        """
        Run detection on a frame.

        Args:
            frame: BGR image as numpy array.
            field_mask: optional FieldMask (see src/field_mask.py). When given
                and usable, detections whose ground-contact point falls outside
                the playing surface are dropped — this is what removes the
                crowd, which otherwise outnumbers the players on a wide shot.

        Returns:
            List of Detection objects.
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False
        )

        use_mask = field_mask is not None and getattr(field_mask, "is_usable", False)

        detections = []

        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
            
            for box in boxes:
                # Extract detection info
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                bbox = tuple(xyxy)

                # Reject anything not standing on the playing surface (crowd,
                # dugout, people behind the hoardings). The ball is exempt —
                # it can legitimately be airborne above the stands' pixels.
                if use_mask and not self.is_ball(class_id):
                    if not field_mask.bbox_on_field(bbox):
                        continue

                detection = Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)

        return detections

    def detect_ball_sliced(
        self,
        frame: np.ndarray,
        slice_wh: Tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.2,
    ) -> List[Detection]:
        """
        Find the ball with SAHI-style sliced inference.

        The ball is only a handful of pixels on a wide broadcast shot, so a
        single whole-frame pass often misses it entirely. Slicing runs the
        detector over overlapping tiles at full resolution, which makes the
        ball comparatively large in each tile, then merges the results with
        NMS. Slower than one pass, so the pipeline runs it periodically
        rather than every frame.
        """
        try:
            import supervision as sv
        except ImportError:
            return []

        ball_ids = self.ball_ids
        if not ball_ids:
            return []

        def _callback(patch: np.ndarray) -> "sv.Detections":
            res = self.model(
                patch,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=sorted(ball_ids),
                imgsz=max(640, min(self.imgsz, 1280)),
                verbose=False,
            )[0]
            return sv.Detections.from_ultralytics(res)

        try:
            slicer = sv.InferenceSlicer(
                callback=_callback,
                slice_wh=slice_wh,
                overlap_ratio_wh=(overlap_ratio, overlap_ratio),
                iou_threshold=0.1,
            )
            merged = slicer(frame)
        except Exception:
            return []

        out: List[Detection] = []
        for i in range(len(merged)):
            x1, y1, x2, y2 = merged.xyxy[i].astype(int)
            cid = int(merged.class_id[i]) if merged.class_id is not None else -1
            out.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(merged.confidence[i]) if merged.confidence is not None else 0.0,
                class_id=cid,
                class_name=self.class_names.get(cid, "ball"),
            ))
        return out

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Run detection on multiple frames (batch inference).
        
        Args:
            frames: List of BGR images as numpy arrays.
        
        Returns:
            List of detection lists, one per frame.
        """
        batch_results = self.model(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False
        )
        
        all_detections = []
        
        for result in batch_results:
            frame_detections = []
            boxes = result.boxes
            
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    detection = Detection(
                        bbox=tuple(xyxy),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    )
                    frame_detections.append(detection)
            
            all_detections.append(frame_detections)
        
        return all_detections
    
    def draw_detections(self,
                        frame: np.ndarray,
                        detections: List[Detection],
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2,
                        show_labels: bool = True,
                        show_confidence: bool = True) -> np.ndarray:
        """
        Draw detection boxes on frame.
        
        Args:
            frame: BGR image as numpy array.
            detections: List of Detection objects.
            color: BGR color for boxes.
            thickness: Line thickness.
            show_labels: Whether to show class labels.
            show_confidence: Whether to show confidence scores.
        
        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(det.class_name)
                if show_confidence:
                    label_parts.append(f"{det.confidence:.2f}")
                label = " ".join(label_parts)
                
                # Background for label
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return annotated
    
    def filter_by_region(self,
                         detections: List[Detection],
                         region: Tuple[int, int, int, int]) -> List[Detection]:
        """
        Filter detections to only those within a region.
        
        Args:
            detections: List of Detection objects.
            region: (x1, y1, x2, y2) region bounds.
        
        Returns:
            Filtered list of detections.
        """
        rx1, ry1, rx2, ry2 = region
        
        filtered = []
        for det in detections:
            cx, cy = det.center
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                filtered.append(det)
        
        return filtered
    
    def filter_by_size(self,
                       detections: List[Detection],
                       min_area: int = 0,
                       max_area: int = float('inf')) -> List[Detection]:
        """
        Filter detections by bounding box area.
        
        Args:
            detections: List of Detection objects.
            min_area: Minimum box area.
            max_area: Maximum box area.
        
        Returns:
            Filtered list of detections.
        """
        return [d for d in detections if min_area <= d.area <= max_area]
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""
        self.confidence_threshold = threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "classes": self.classes
        }


if __name__ == "__main__":
    # Demo usage
    print("NFL Vision - Detector Demo")
    print("-" * 40)
    
    # Create detector
    detector = Detector(
        model_path="yolov8n.pt",
        confidence_threshold=0.5,
        device="mps"
    )
    
    # Test with synthetic image
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    print("\nRunning detection on test frame...")
    detections = detector.detect(test_frame)
    print(f"Found {len(detections)} detections")
    
    print("\nDetector stats:", detector.get_stats())
