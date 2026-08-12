"""
NFL Vision App - Source Package
Screen capture, detection, tracking, and play tracking modules.

Note: Imports are lazy to avoid loading heavy dependencies on init.
"""

# Lazy imports - components loaded on demand
__all__ = [
    'ScreenCapture', 'Detector', 'Tracker', 'Detection', 'Track',
    'PlayStore', 'PlayTracker', 'PlayData'
]

def __getattr__(name):
    """Lazy load modules on first access."""
    if name == 'ScreenCapture':
        from .capture import ScreenCapture
        return ScreenCapture
    elif name == 'Detector':
        from .detector import Detector
        return Detector
    elif name == 'Detection':
        from .detector import Detection
        return Detection
    elif name == 'Tracker':
        from .tracker import Tracker
        return Tracker
    elif name == 'Track':
        from .tracker import Track
        return Track
    elif name == 'PlayStore':
        from .play_store import PlayStore
        return PlayStore
    elif name == 'PlayData':
        from .play_store import PlayData
        return PlayData
    elif name == 'PlayTracker':
        from .play_tracker import PlayTracker
        return PlayTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
