"""
macOS Screen Capture for FootballVision

Grabs frames from a chosen display (or a sub-region of it) using Quartz /
CoreGraphics — the native macOS screen API, no extra dependencies. Returns
BGR numpy frames ready for OpenCV / YOLO.

Which screen? `list_displays()` enumerates every attached display with its
index, resolution, and origin. `ScreenCapture(display_index=N)` then captures
that specific display; pass `region=(x, y, w, h)` (in that display's own point
coordinates) to capture just the video window instead of the whole desktop.

Requires macOS **Screen Recording** permission for the app running Python
(Terminal/iTerm). Without it, CoreGraphics hands back black or empty frames —
`get_frame()` returns None and `permission_ok()` reports False.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import Quartz
    import Quartz.CoreGraphics as CG
    QUARTZ_AVAILABLE = True
except Exception:  # pragma: no cover - non-macOS / missing pyobjc
    QUARTZ_AVAILABLE = False

# ScreenCaptureKit is Apple's replacement for the window-capture APIs they
# disabled in macOS 14/15. It buys two things here:
#
#   1. Enumeration of windows that are NOT currently on screen — minimised, or
#      sitting on a Space you have switched away from. `CGWindowListCopyWindowInfo`
#      with kCGWindowListOptionOnScreenOnly cannot see those at all, which is
#      why the picker used to show a single window.
#   2. Capture of a *window* rather than a rectangle of a display. The old path
#      cropped the display to the window's bounds, so anything overlapping the
#      window was captured instead of it. SCContentFilter captures the window
#      itself, independent of what is composited on top.
#
# ⚠️ Measured limit (2026-07-26, Darwin 25.5): SCK enumerates off-screen windows
# but CANNOT capture them — every attempt returns SCStreamErrorDomain -3811.
# macOS does not render windows on an inactive Space, so there are no pixels to
# capture by any API. `activate_window()` exists to fix that: raise the window
# first, then capture. This is what a screen-sharing app is doing when picking
# a background window appears to "just work".
try:
    if QUARTZ_AVAILABLE:
        # Touch CoreGraphics first: SCK calls assert on an uninitialised
        # CGS/WindowServer connection ("CGS_REQUIRE_INIT") if this is the very
        # first graphics call the process makes.
        Quartz.CGMainDisplayID()
    import ScreenCaptureKit as SCK
    SCK_AVAILABLE = True
except Exception:  # pragma: no cover - older macOS / missing pyobjc
    SCK_AVAILABLE = False

_SCK_TIMEOUT = 5.0

# System UI that is never a useful capture source but does have layer-0 windows.
_SYSTEM_UI_APPS = {
    "Dock", "Notification Center", "NotificationCenter", "loginwindow",
    "Spotlight", "UserNotificationCenter", "AccessibilityVisualsAgent",
    "Window Server", "Control Center", "SystemUIServer",
}


def _sck_shareable(on_screen_only: bool = False):
    """Fetch SCShareableContent synchronously. Returns None on failure.

    The API is async with a completion handler that fires on a background
    queue, so an Event is enough — no run loop needed. The handler must be a
    real function: a lambda that returns a value breaks the block signature
    and crashes inside ScreenCaptureKit.
    """
    if not SCK_AVAILABLE:
        return None
    box: dict = {}
    done = threading.Event()

    def handler(content, error):
        box["content"] = content
        box["error"] = error
        done.set()

    try:
        SCK.SCShareableContent.\
            getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                True, on_screen_only, handler)
    except Exception:
        return None
    if not done.wait(_SCK_TIMEOUT):
        return None
    if box.get("error") is not None:
        return None
    return box.get("content")


def _sck_window_objects(min_width: int, min_height: int) -> List[tuple]:
    """[(window_id, SCWindow)] for real application windows, largest first."""
    content = _sck_shareable(on_screen_only=False)
    if content is None:
        return []
    out = []
    for w in content.windows():
        try:
            if int(w.windowLayer()) != 0:
                continue
            fr = w.frame()
            if fr.size.width < min_width or fr.size.height < min_height:
                continue
            app = w.owningApplication()
            if app is None or not (app.applicationName() or "").strip():
                continue
            out.append((int(w.windowID()), w))
        except Exception:
            continue
    out.sort(key=lambda t: t[1].frame().size.width * t[1].frame().size.height,
             reverse=True)
    return out


def activate_window(window_id: int) -> bool:
    """Bring a window's application to the front so it can be captured.

    macOS will not render a window on an inactive Space, and an unrendered
    window cannot be captured by any API. Activating its application switches
    to that Space and makes the pixels exist.
    """
    if not SCK_AVAILABLE:
        return False
    try:
        from AppKit import NSRunningApplication
    except Exception:
        return False
    for wid, win in _sck_window_objects(1, 1):
        if wid != window_id:
            continue
        app = win.owningApplication()
        if app is None:
            return False
        running = NSRunningApplication.runningApplicationWithProcessIdentifier_(
            int(app.processID()))
        if running is None:
            return False
        # 1 << 1 == NSApplicationActivateIgnoringOtherApps
        running.activateWithOptions_(1 << 1)
        return True
    return False


def list_displays(max_displays: int = 16) -> List[dict]:
    """Return every active display: index, id, origin, size, primary flag."""
    if not QUARTZ_AVAILABLE:
        return []
    err, active, count = Quartz.CGGetActiveDisplayList(max_displays, None, None)
    if err:
        return []
    out = []
    for i in range(count):
        did = active[i]
        b = Quartz.CGDisplayBounds(did)
        out.append({
            "index": i,
            "id": int(did),
            "x": int(b.origin.x),
            "y": int(b.origin.y),
            "width": int(b.size.width),
            "height": int(b.size.height),
            "primary": bool(Quartz.CGDisplayIsMain(did)),
        })
    return out


def has_screen_permission() -> bool:
    """
    Whether this process holds macOS **Screen Recording** permission.

    This check is essential, not cosmetic. Without the permission macOS does
    **not** raise an error — `CGDisplayCreateImage` still returns a perfectly
    valid image, but with every application window stripped out, leaving only
    the desktop wallpaper. A naive "is the frame non-black?" test therefore
    passes while the capture is useless, and detection silently finds nothing.
    """
    if not QUARTZ_AVAILABLE:
        return False
    try:
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        return False


def request_screen_permission() -> bool:
    """
    Ask macOS to prompt for Screen Recording permission.

    Registers the host app in System Settings → Privacy & Security → Screen &
    System Audio Recording. The app must be **restarted** before a newly
    granted permission takes effect.
    """
    if not QUARTZ_AVAILABLE:
        return False
    try:
        return bool(Quartz.CGRequestScreenCaptureAccess())
    except Exception:
        return False


def list_windows(min_width: int = 240, min_height: int = 180) -> List[dict]:
    """
    Return on-screen application windows worth capturing.

    Capturing a single window (the YouTube tab, VLC, a stream player) is both
    better UX and better accuracy than grabbing a whole 4K display and then
    downscaling it — all the pixels go to the pitch instead of your desktop.

    Filters out desktop elements, menu-bar items, and tiny utility windows.

    Uses ScreenCaptureKit when available, which also returns windows that are
    minimised or on another Space — these carry ``on_screen: False``. They are
    listed deliberately: hiding them was the reason the picker used to show one
    window and no browser. They cannot be captured until raised, so the caller
    should activate them first (see ``activate_window``).
    """
    if SCK_AVAILABLE:
        out = []
        for wid, win in _sck_window_objects(min_width, min_height):
            app = win.owningApplication()
            owner = (app.applicationName() or "").strip()
            if owner in _SYSTEM_UI_APPS or owner.startswith("AutoFill"):
                continue
            fr = win.frame()
            out.append({
                "id": wid,
                "app": owner,
                "title": str(win.title() or ""),
                "x": int(fr.origin.x),
                "y": int(fr.origin.y),
                "width": int(fr.size.width),
                "height": int(fr.size.height),
                "on_screen": bool(win.isOnScreen()),
            })
        if out:
            return out
        # Fall through to Quartz if SCK returned nothing at all.

    if not QUARTZ_AVAILABLE:
        return []
    opts = (Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements)
    infos = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []

    out = []
    for info in infos:
        # Layer 0 is the normal application window layer; higher layers are
        # menus, docks, overlays and status items.
        if int(info.get("kCGWindowLayer", 1) or 0) != 0:
            continue
        b = info.get("kCGWindowBounds") or {}
        w, h = int(b.get("Width", 0)), int(b.get("Height", 0))
        if w < min_width or h < min_height:
            continue
        owner = str(info.get("kCGWindowOwnerName", "") or "")
        title = str(info.get("kCGWindowName", "") or "")
        if not owner:
            continue
        out.append({
            "id": int(info.get("kCGWindowNumber", 0)),
            "app": owner,
            "title": title,
            "x": int(b.get("X", 0)),
            "y": int(b.get("Y", 0)),
            "width": w,
            "height": h,
            # The Quartz path only ever enumerates on-screen windows.
            "on_screen": True,
        })
    # Biggest first — the media window is usually the largest.
    out.sort(key=lambda d: d["width"] * d["height"], reverse=True)
    return out


def _sck_capture(win, scale: int = 2):
    """Capture one SCWindow via SCScreenshotManager. Returns (CGImage, error).

    `error` is an int SCStreamErrorDomain code, or None. -3811 means the
    window is not being rendered (minimised, or on an inactive Space); no
    configuration fixes that, only raising the window does.
    """
    if not SCK_AVAILABLE:
        return None, None
    try:
        fr = win.frame()
        pixel_width = int(fr.size.width * scale)
        pixel_height = int(fr.size.height * scale)
        if not (16 <= pixel_width <= 16384 and 16 <= pixel_height <= 16384):
            return None, -2
        flt = SCK.SCContentFilter.alloc().initWithDesktopIndependentWindow_(win)
        cfg = SCK.SCStreamConfiguration.alloc().init()
        cfg.setWidth_(pixel_width)
        cfg.setHeight_(pixel_height)
        cfg.setShowsCursor_(False)
    except Exception:
        return None, None

    box: dict = {}
    done = threading.Event()

    def handler(image, error):
        box["image"] = image
        box["error"] = error
        done.set()

    try:
        SCK.SCScreenshotManager.\
            captureImageWithFilter_configuration_completionHandler_(flt, cfg, handler)
    except Exception:
        return None, None
    if not done.wait(_SCK_TIMEOUT):
        return None, None
    err = box.get("error")
    if err is not None:
        try:
            return None, int(err.code())
        except Exception:
            return None, -1
    return box.get("image"), None


def _cgimage_to_bgr(image) -> Optional[np.ndarray]:
    """Convert a CGImage (BGRA, possibly row-padded) to a BGR numpy array."""
    if image is None:
        return None
    width = Quartz.CGImageGetWidth(image)
    height = Quartz.CGImageGetHeight(image)
    if not (0 < width <= 16384 and 0 < height <= 16384):
        return None
    bytes_per_row = Quartz.CGImageGetBytesPerRow(image)
    provider = Quartz.CGImageGetDataProvider(image)
    data = Quartz.CGDataProviderCopyData(provider)
    if data is None:
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    # On Retina displays the row stride is padded (bytes_per_row > width*4) and
    # the returned buffer can carry a few trailing bytes; use the stride and
    # trim to exactly height rows before reshaping, then drop padding + alpha.
    needed = bytes_per_row * height
    if buf.size < needed:
        return None
    buf = buf[:needed].reshape((height, bytes_per_row // 4, 4))
    bgr = buf[:, :width, :3]                 # drop alpha; memory order is B,G,R
    return np.ascontiguousarray(bgr)


class ScreenCapture:
    """
    On-demand display capture with fps rate-limiting.

    Usage:
        cap = ScreenCapture(display_index=0)     # primary display
        cap.start()
        frame = cap.get_frame()                  # BGR numpy or None
        cap.stop()
    """

    def __init__(
        self,
        display_index: int = 0,
        region: Optional[Tuple[int, int, int, int]] = None,
        target_fps: int = 30,
        window_id: Optional[int] = None,
    ):
        """
        Args:
            display_index: which display to capture (ignored if window_id set).
            region: (left, top, width, height) sub-region of the display.
            target_fps: capture rate limit.
            window_id: capture just this window (see `list_windows()`), which
                keeps full resolution on the content that matters.
        """
        if not QUARTZ_AVAILABLE:
            raise RuntimeError(
                "Quartz not available — install pyobjc-framework-Quartz "
                "(macOS only)."
            )
        self._region = region
        self._target_fps = max(1, target_fps)
        self._running = False
        self._last_grab = 0.0
        self._window_id = window_id
        self._display_index = display_index
        self._display_id = self._resolve_display(display_index)
        # Retina: capture at 2x the window's point size so the pitch keeps
        # native pixels rather than being downscaled before detection.
        self._sck_scale = 2
        self._sck_cache = None
        # Last human-readable reason a frame could not be captured. The UI
        # surfaces this instead of reporting an empty pitch.
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    def _display_for_point(self, gx: float, gy: float) -> int:
        """Which display contains this global point (falls back to current)."""
        for d in list_displays():
            if (d["x"] <= gx < d["x"] + d["width"]
                    and d["y"] <= gy < d["y"] + d["height"]):
                return d["id"]
        return self._display_id

    def _resolve_display(self, index: int) -> int:
        displays = list_displays()
        if not displays:
            return Quartz.CGMainDisplayID()
        index = max(0, min(index, len(displays) - 1))
        self._display_index = index
        return displays[index]["id"]

    # ------------------------------------------------------------------
    def start(self) -> bool:
        self._running = True
        self._last_grab = 0.0
        return True

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def set_region(self, left: int, top: int, width: int, height: int) -> None:
        self._region = (left, top, width, height)

    def set_display(self, index: int) -> None:
        self._display_id = self._resolve_display(index)

    # ------------------------------------------------------------------
    def set_window(self, window_id: Optional[int]) -> None:
        """Capture a specific window (None reverts to whole-display capture)."""
        self._window_id = window_id
        self._sck_cache = None
        self.last_error = None

    def _sck_window(self):
        """The SCWindow for the current target, cached across frames.

        Re-fetching shareable content is an XPC round trip, far too expensive
        to repeat every frame, so the handle is cached and only refreshed when
        a capture fails.
        """
        if self._sck_cache is not None:
            return self._sck_cache
        for wid, win in _sck_window_objects(1, 1):
            if wid == self._window_id:
                self._sck_cache = win
                return win
        return None

    def _window_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """Current global bounds of the target window, or None if it's gone."""
        for w in list_windows(min_width=1, min_height=1):
            if w["id"] == self._window_id:
                return (w["x"], w["y"], w["width"], w["height"])
        return None

    def _grab_image(self):
        if self._window_id:
            # Preferred path: ScreenCaptureKit captures the *window*, so an
            # overlapping window no longer contaminates the frame the way a
            # display-rectangle crop does.
            if SCK_AVAILABLE:
                win = self._sck_window()
                if win is not None:
                    img, err = _sck_capture(win, scale=self._sck_scale)
                    if img is not None:
                        self.last_error = None
                        return img
                    if err == -3811:
                        # Not rendered by macOS: minimised or on another Space.
                        # Cropping the display here would silently return
                        # whatever is in front instead — usually the wallpaper —
                        # which is exactly the failure this guard prevents.
                        self.last_error = (
                            "window is not on screen (minimised, or on another "
                            "Space) — macOS renders no pixels for it, so it "
                            "cannot be captured until it is brought to the front"
                        )
                        return None
                    self._sck_cache = None  # stale handle; refetch next time

            # Fallback for macOS without ScreenCaptureKit. CGWindowListCreateImage
            # is disabled from macOS 14/15 onward and returns None, so capture
            # the display that hosts the window and crop to its bounds.
            bounds = self._window_bounds()
            if bounds is not None:
                gx, gy, gw, gh = bounds
                did = self._display_for_point(gx + gw / 2, gy + gh / 2)
                db = Quartz.CGDisplayBounds(did)
                # Convert global point coords to display-local point coords.
                rect = CG.CGRectMake(gx - db.origin.x, gy - db.origin.y, gw, gh)
                img = CG.CGDisplayCreateImageForRect(did, rect)
                if img is not None:
                    return img
            # Window vanished (closed/minimised) — fall through to the display.
        if self._region is not None:
            x, y, w, h = self._region
            rect = CG.CGRectMake(x, y, w, h)
            return CG.CGDisplayCreateImageForRect(self._display_id, rect)
        return CG.CGDisplayCreateImage(self._display_id)

    def get_frame(self) -> Optional[np.ndarray]:
        if not self._running:
            return None
        # Rate-limit to target_fps.
        now = time.time()
        interval = 1.0 / self._target_fps
        if now - self._last_grab < interval:
            time.sleep(max(0.0, interval - (now - self._last_grab)))
        self._last_grab = time.time()

        image = self._grab_image()
        return _cgimage_to_bgr(image)

    def permission_ok(self) -> bool:
        """True if this process holds macOS Screen Recording permission."""
        return has_screen_permission()


if __name__ == "__main__":
    print("Quartz available:", QUARTZ_AVAILABLE)
    for d in list_displays():
        tag = " (primary)" if d["primary"] else ""
        print(f"  display {d['index']}: {d['width']}x{d['height']} "
              f"@ ({d['x']},{d['y']}) id={d['id']}{tag}")
    if QUARTZ_AVAILABLE and list_displays():
        cap = ScreenCapture(display_index=0)
        cap.start()
        f = cap.get_frame()
        cap.stop()
        if f is None:
            print("get_frame() -> None (no Screen Recording permission?)")
        else:
            print(f"captured frame: {f.shape}  non-black={bool(f.any())}")
