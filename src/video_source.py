"""
Video Source Module for NFL Vision Dot Simulator
Unified abstraction for different video input sources.

Supports:
- Live screen capture
- Video file playback (MP4, MOV, etc.)
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from enum import Enum
import time
import os


class SourceType(Enum):
    """Video source types."""
    SCREEN = "screen"
    FILE = "file"


class VideoSource(ABC):
    """Abstract base class for video sources."""
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the next frame. Returns None if no frame available."""
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """Get the frame rate of the source."""
        pass
    
    @abstractmethod
    def is_live(self) -> bool:
        """Return True if this is a live source (can't seek/pause)."""
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """Start capturing/playing. Returns True on success."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop capturing/playing."""
        pass
    
    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if source is currently active."""
        pass
    
    @property
    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """Get (width, height) of frames."""
        pass


class ScreenCaptureSource(VideoSource):
    """
    Live screen capture source.
    
    Wraps the existing ScreenCapture class for the unified interface.
    """
    
    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        target_fps: int = 30,
        display_index: int = 0,
        window_id: Optional[int] = None,
    ):
        """
        Initialize screen capture source.

        Args:
            region: (left, top, width, height) to capture. None for whole display.
            target_fps: Target capture rate.
            display_index: which display to capture (see capture.list_displays()).
            window_id: capture just this window (see capture.list_windows()).
        """
        self._region = region
        self._target_fps = target_fps
        self._display_index = display_index
        self._window_id = window_id
        self._running = False
        self._capture = None
        self._frame_size = (1920, 1080)  # Default, updated on start
        
    def get_frame(self) -> Optional[np.ndarray]:
        if not self._running or not self._capture:
            return None
        
        frame = self._capture.get_frame()
        if frame is not None:
            self._frame_size = (frame.shape[1], frame.shape[0])
        return frame
    
    def get_fps(self) -> float:
        return float(self._target_fps)
    
    def is_live(self) -> bool:
        return True
    
    def start(self) -> bool:
        try:
            from src.capture import ScreenCapture
            self._capture = ScreenCapture(
                region=self._region,
                target_fps=self._target_fps,
                display_index=self._display_index,
                window_id=self._window_id,
            )
            self._capture.start()
            self._running = True
            return True
        except Exception as e:
            print(f"[ScreenCaptureSource] Failed to start: {e}")
            return False
    
    def stop(self) -> None:
        if self._capture:
            self._capture.stop()
        self._running = False
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._frame_size
    
    def set_region(self, left: int, top: int, width: int, height: int) -> None:
        """Update capture region."""
        self._region = (left, top, width, height)
        if self._capture:
            self._capture.set_region(left, top, width, height)


class VideoFileSource(VideoSource):
    """
    Video file playback source.
    
    Supports common video formats (MP4, MOV, AVI, etc.)
    """
    
    def __init__(self, file_path: str):
        """
        Initialize video file source.
        
        Args:
            file_path: Path to video file.
        """
        self._file_path = file_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._fps = 30.0
        self._frame_size = (1920, 1080)
        self._frame_count = 0
        self._current_frame = 0
        self._last_frame_time = 0
        self._paused = False
        
    def get_frame(self) -> Optional[np.ndarray]:
        if not self._running or not self._cap:
            return None
        
        if self._paused:
            # Return the same frame when paused
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)
        
        # Rate limiting to match original FPS
        current_time = time.time()
        elapsed = current_time - self._last_frame_time
        target_interval = 1.0 / self._fps
        
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        
        ret, frame = self._cap.read()
        if not ret:
            # Video ended - loop or stop
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._current_frame = 0
            ret, frame = self._cap.read()
            if not ret:
                return None
        
        self._current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._last_frame_time = time.time()
        
        return frame
    
    def get_fps(self) -> float:
        return self._fps
    
    def is_live(self) -> bool:
        return False
    
    def start(self) -> bool:
        if not os.path.exists(self._file_path):
            print(f"[VideoFileSource] File not found: {self._file_path}")
            return False
        
        try:
            self._cap = cv2.VideoCapture(self._file_path)
            if not self._cap.isOpened():
                print(f"[VideoFileSource] Failed to open: {self._file_path}")
                return False
            
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._frame_size = (width, height)
            self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self._running = True
            self._last_frame_time = time.time()
            
            print(f"[VideoFileSource] Opened: {self._file_path}")
            print(f"  Resolution: {width}x{height} @ {self._fps:.1f} FPS")
            print(f"  Duration: {self._frame_count / self._fps:.1f}s ({self._frame_count} frames)")
            
            return True
            
        except Exception as e:
            print(f"[VideoFileSource] Error: {e}")
            return False
    
    def stop(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._running = False
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._frame_size
    
    def pause(self) -> None:
        """Pause playback."""
        self._paused = True
    
    def resume(self) -> None:
        """Resume playback."""
        self._paused = False
    
    def seek(self, frame_number: int) -> None:
        """Seek to specific frame."""
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._current_frame = frame_number
    
    def seek_percent(self, percent: float) -> None:
        """Seek to percentage of video (0-100)."""
        frame = int((percent / 100) * self._frame_count)
        self.seek(frame)
    
    @property
    def current_frame(self) -> int:
        return self._current_frame
    
    @property
    def total_frames(self) -> int:
        return self._frame_count
    
    @property
    def progress_percent(self) -> float:
        if self._frame_count == 0:
            return 0.0
        return (self._current_frame / self._frame_count) * 100
    
    @property
    def current_time(self) -> float:
        """Current playback time in seconds."""
        return self._current_frame / self._fps if self._fps > 0 else 0
    
    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self._frame_count / self._fps if self._fps > 0 else 0


class DemoSource(VideoSource):
    """
    Demo source with synthetic football field.
    For testing without real video input.
    """
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self._width = width
        self._height = height
        self._fps = fps
        self._running = False
        self._frame_count = 0
        self._last_frame_time = 0
        
        # Player state for animation
        self._players = self._init_players()
    
    def _init_players(self) -> list:
        """Initialize animated players."""
        import random
        players = []
        
        # Offense (11 players)
        for i in range(11):
            players.append({
                'x': random.uniform(0.3, 0.7) * self._width,
                'y': random.uniform(0.4, 0.7) * self._height,
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(-2, 2),
                'team': 'offense'
            })
        
        # Defense (11 players)
        for i in range(11):
            players.append({
                'x': random.uniform(0.3, 0.7) * self._width,
                'y': random.uniform(0.2, 0.5) * self._height,
                'vx': random.uniform(-1.5, 1.5),
                'vy': random.uniform(-1.5, 1.5),
                'team': 'defense'
            })
        
        return players
    
    def get_frame(self) -> Optional[np.ndarray]:
        if not self._running:
            return None
        
        # Rate limiting
        current_time = time.time()
        elapsed = current_time - self._last_frame_time
        target_interval = 1.0 / self._fps
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        
        # Create field
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        
        # Green gradient
        for y in range(self._height):
            green = int(60 + 40 * np.sin(y / 50))
            frame[y, :] = (20, green, 20)
        
        # Yard lines
        for x in range(0, self._width, 80):
            cv2.line(frame, (x, 0), (x, self._height), (255, 255, 255), 1)
        
        # Update and draw players
        for p in self._players:
            # Update position
            p['x'] += p['vx']
            p['y'] += p['vy']
            
            # Bounce
            if p['x'] < 50 or p['x'] > self._width - 50:
                p['vx'] *= -1
            if p['y'] < 50 or p['y'] > self._height - 50:
                p['vy'] *= -1
            
            # Draw
            x, y = int(p['x']), int(p['y'])
            if p['team'] == 'offense':
                color = (50, 200, 255)  # Gold/orange
            else:
                color = (255, 150, 50)  # Blue
            
            cv2.circle(frame, (x, y), 12, color, -1)
            cv2.circle(frame, (x, y), 14, (255, 255, 255), 2)
        
        self._frame_count += 1
        self._last_frame_time = time.time()
        
        return frame
    
    def get_fps(self) -> float:
        return float(self._fps)
    
    def is_live(self) -> bool:
        return True  # Synthetic but continuous
    
    def start(self) -> bool:
        self._running = True
        self._last_frame_time = time.time()
        print("[DemoSource] Started synthetic field generator")
        return True
    
    def stop(self) -> None:
        self._running = False
        print("[DemoSource] Stopped")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._width, self._height)
    
    @property
    def players(self) -> list:
        """Get current player positions for direct dot simulator use."""
        return self._players


def create_source(
    source_type: SourceType,
    file_path: Optional[str] = None,
    region: Optional[Tuple[int, int, int, int]] = None,
    fps: int = 30
) -> VideoSource:
    """
    Factory function to create video sources.
    
    Args:
        source_type: Type of source to create
        file_path: Path for FILE source
        region: Screen region for SCREEN source
        fps: Target FPS
    
    Returns:
        VideoSource instance
    """
    if source_type == SourceType.FILE:
        if not file_path:
            raise ValueError("file_path required for FILE source")
        return VideoFileSource(file_path)
    
    elif source_type == SourceType.SCREEN:
        return ScreenCaptureSource(region=region, target_fps=fps)
    
    else:
        raise ValueError(f"Unknown source type: {source_type}")


if __name__ == "__main__":
    print("Video Source Test")
    print("-" * 40)
    
    # Test demo source
    source = DemoSource()
    source.start()
    
    for i in range(5):
        frame = source.get_frame()
        if frame is not None:
            print(f"Frame {i+1}: {frame.shape}")
    
    source.stop()
    print("Done")
