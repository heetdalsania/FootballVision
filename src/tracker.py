"""
Player Tracking Module for NFL Vision App
Uses ByteTrack for multi-object tracking with persistent IDs.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

# Import our Detection class
from .detector import Detection


@dataclass
class Track:
    """Represents a tracked object across frames."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    age: int = 0  # Frames since first detection
    hits: int = 1  # Number of times detected
    time_since_update: int = 0  # Frames since last update
    
    # Position history for formation analysis
    position_history: List[Tuple[int, int]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)

    # Football-analysis fields (populated by the pipeline)
    team: int = -1                                  # -1 unknown, 0 team A, 1 team B
    pitch_xy: Optional[Tuple[float, float]] = None  # position in pitch metres

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def is_confirmed(self) -> bool:
        """
        Check if track is confirmed (enough hits).

        Two hits rather than three: broadcast players occlude each other
        constantly, and a stricter threshold made the reported player count
        oscillate (2-21) instead of settling near the ~22 actually on screen.
        """
        return self.hits >= 2
    
    def update_position_history(self, max_history: int = 30):
        """Update position history with current center."""
        self.position_history.append(self.center)
        if len(self.position_history) > max_history:
            self.position_history.pop(0)
        
        # Calculate velocity
        if len(self.position_history) >= 2:
            prev = self.position_history[-2]
            curr = self.position_history[-1]
            self.velocity = (curr[0] - prev[0], curr[1] - prev[1])
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center": self.center,
            "age": self.age,
            "hits": self.hits,
            "velocity": self.velocity,
            "team": self.team,
            "pitch_xy": self.pitch_xy,
        }


class Tracker:
    """
    ByteTrack-based multi-object tracker for NFL Vision.
    
    Maintains persistent player IDs across frames and tracks
    position history for formation analysis.
    
    Example:
        tracker = Tracker()
        tracks = tracker.update(detections)
        annotated = tracker.draw_tracks(frame, tracks)
    """
    
    def __init__(self,
                 track_thresh: float = 0.25,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30,
                 position_history_length: int = 30):
        """
        Initialize the tracker.
        
        Args:
            track_thresh: Detection confidence threshold for tracking.
            track_buffer: Number of frames to keep lost tracks.
            match_thresh: IoU threshold for matching detections to tracks.
            frame_rate: Video frame rate for velocity calculation.
            position_history_length: Number of positions to keep per track.
        """
        if not SUPERVISION_AVAILABLE:
            raise RuntimeError(
                "Supervision not available. "
                "Install with: pip install supervision"
            )
        
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.position_history_length = position_history_length
        
        # Initialize ByteTrack
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )
        
        # Track metadata storage
        self._track_metadata: Dict[int, Track] = {}
        self._frame_count = 0
        
        print(f"[Tracker] Initialized ByteTrack (buffer={track_buffer})")
    
    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Update tracks with new detections.

        Args:
            detections: List of Detection objects from detector.
            frame: Optional BGR frame the detections came from. Accepted for
                call-site compatibility (the pipeline passes it); team
                assignment and homography are applied downstream in the
                pipeline, so it is not required here.

        Returns:
            List of active Track objects with persistent IDs.
        """
        self._frame_count += 1
        
        if not detections:
            # Age existing tracks
            for track in self._track_metadata.values():
                track.time_since_update += 1
            return list(self._track_metadata.values())
        
        # Convert detections to supervision format
        xyxy = np.array([d.bbox for d in detections])
        confidence = np.array([d.confidence for d in detections])
        class_ids = np.array([d.class_id for d in detections])
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids
        )
        
        # Run ByteTrack
        tracked = self.byte_tracker.update_with_detections(sv_detections)
        
        # Convert to our Track format
        current_tracks = []
        active_ids = set()
        
        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                track_id = int(track_id)
                active_ids.add(track_id)
                
                bbox = tuple(tracked.xyxy[i].astype(int))
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
                class_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                
                # Find original detection for class name
                class_name = "player"
                for det in detections:
                    if np.allclose(det.bbox, bbox, atol=5):
                        class_name = det.class_name
                        break
                
                # Update or create track
                if track_id in self._track_metadata:
                    track = self._track_metadata[track_id]
                    track.bbox = bbox
                    track.confidence = conf
                    track.age += 1
                    track.hits += 1
                    track.time_since_update = 0
                    # Refresh the role too: a track created before the model
                    # resolved someone as a referee/goalkeeper would otherwise
                    # stay labelled "player" for its whole lifetime.
                    track.class_id = class_id
                    track.class_name = class_name
                    track.update_position_history(self.position_history_length)
                else:
                    track = Track(
                        track_id=track_id,
                        bbox=bbox,
                        confidence=conf,
                        class_id=class_id,
                        class_name=class_name
                    )
                    track.update_position_history(self.position_history_length)
                    self._track_metadata[track_id] = track
                
                current_tracks.append(track)
        
        # Update time_since_update for inactive tracks
        for track_id, track in self._track_metadata.items():
            if track_id not in active_ids:
                track.time_since_update += 1
        
        # Remove old tracks
        self._track_metadata = {
            tid: t for tid, t in self._track_metadata.items()
            if t.time_since_update < self.track_buffer
        }
        
        return current_tracks
    
    def get_tracks(self) -> List[Track]:
        """Get all active tracks."""
        return [t for t in self._track_metadata.values() 
                if t.time_since_update < self.track_buffer]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get only confirmed tracks (detected multiple times)."""
        return [t for t in self.get_tracks() if t.is_confirmed]
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get a specific track by ID."""
        return self._track_metadata.get(track_id)
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=self.frame_rate
        )
        self._track_metadata.clear()
        self._frame_count = 0
        print("[Tracker] Reset complete")
    
    def draw_tracks(self,
                    frame: np.ndarray,
                    tracks: Optional[List[Track]] = None,
                    show_ids: bool = True,
                    show_trails: bool = True,
                    trail_length: int = 15,
                    color_by_id: bool = True) -> np.ndarray:
        """
        Draw tracks on frame.
        
        Args:
            frame: BGR image as numpy array.
            tracks: List of Track objects. If None, uses current tracks.
            show_ids: Whether to show track IDs.
            show_trails: Whether to show position trails.
            trail_length: Number of trail points to show.
            color_by_id: Whether to color by track ID.
        
        Returns:
            Annotated frame.
        """
        import cv2
        
        annotated = frame.copy()
        
        if tracks is None:
            tracks = self.get_tracks()
        
        for track in tracks:
            # Generate color from track ID
            if color_by_id:
                np.random.seed(track.track_id)
                color = tuple(np.random.randint(100, 255, 3).tolist())
            else:
                color = (0, 255, 0)
            
            x1, y1, x2, y2 = track.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID label
            if show_ids:
                label = f"ID:{track.track_id}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Draw trail
            if show_trails and len(track.position_history) > 1:
                points = track.position_history[-trail_length:]
                for i in range(1, len(points)):
                    # Fade trail
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(
                        annotated,
                        points[i-1],
                        points[i],
                        color,
                        thickness
                    )
        
        return annotated
    
    def get_position_matrix(self) -> np.ndarray:
        """
        Get current positions of all confirmed tracks as matrix.
        
        Returns:
            numpy array of shape (N, 2) with [x, y] positions.
        """
        tracks = self.get_confirmed_tracks()
        if not tracks:
            return np.array([])
        return np.array([t.center for t in tracks])
    
    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Get position history for a specific track."""
        track = self.get_track(track_id)
        return track.position_history if track else []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        tracks = self.get_tracks()
        return {
            "frame_count": self._frame_count,
            "active_tracks": len(tracks),
            "confirmed_tracks": len([t for t in tracks if t.is_confirmed]),
            "total_tracks_seen": len(self._track_metadata),
            "settings": {
                "track_thresh": self.track_thresh,
                "track_buffer": self.track_buffer,
                "match_thresh": self.match_thresh
            }
        }


if __name__ == "__main__":
    # Demo usage
    print("NFL Vision - Tracker Demo")
    print("-" * 40)
    
    tracker = Tracker()
    
    # Simulate detections
    fake_detections = [
        Detection(bbox=(100, 100, 150, 200), confidence=0.9, class_id=0, class_name="player"),
        Detection(bbox=(300, 150, 350, 250), confidence=0.85, class_id=0, class_name="player"),
    ]
    
    print("\nFrame 1:")
    tracks = tracker.update(fake_detections)
    for t in tracks:
        print(f"  Track {t.track_id}: {t.center}")
    
    # Move detections slightly
    fake_detections[0] = Detection(bbox=(105, 105, 155, 205), confidence=0.9, class_id=0, class_name="player")
    fake_detections[1] = Detection(bbox=(310, 160, 360, 260), confidence=0.85, class_id=0, class_name="player")
    
    print("\nFrame 2:")
    tracks = tracker.update(fake_detections)
    for t in tracks:
        print(f"  Track {t.track_id}: {t.center}, velocity={t.velocity}")
    
    print("\nStats:", tracker.get_stats())
