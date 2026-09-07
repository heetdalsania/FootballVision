#!/usr/bin/env python3
"""Small always-on-top macOS monitor for a running FootballVision session."""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request

import AppKit
import Foundation
import objc


class PitchView(AppKit.NSView):
    def initWithFrame_(self, frame):
        self = objc.super(PitchView, self).initWithFrame_(frame)
        if self is not None:
            self.payload = {}
        return self

    def setPayload_(self, payload):
        self.payload = payload or {}
        self.setNeedsDisplay_(True)

    def drawRect_(self, _rect):
        bounds = self.bounds()
        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.04, 0.13, 0.075, 1
        ).setFill()
        AppKit.NSBezierPath.fillRect_(bounds)

        margin = 12.0
        usable_w = max(1.0, bounds.size.width - margin * 2)
        usable_h = max(1.0, bounds.size.height - margin * 2)
        scale = min(usable_w / 105.0, usable_h / 68.0)
        pitch_w, pitch_h = 105.0 * scale, 68.0 * scale
        left = (bounds.size.width - pitch_w) / 2
        bottom = (bounds.size.height - pitch_h) / 2

        AppKit.NSColor.colorWithWhite_alpha_(1, 0.48).setStroke()
        outline = AppKit.NSBezierPath.bezierPathWithRect_(
            AppKit.NSMakeRect(left, bottom, pitch_w, pitch_h)
        )
        outline.setLineWidth_(1.1)
        outline.stroke()
        line = AppKit.NSBezierPath.bezierPath()
        line.moveToPoint_(AppKit.NSMakePoint(left + pitch_w / 2, bottom))
        line.lineToPoint_(AppKit.NSMakePoint(left + pitch_w / 2, bottom + pitch_h))
        line.stroke()

        pitch = (self.payload or {}).get("pitch") or {}
        for player in pitch.get("players") or []:
            try:
                x = left + float(player["x"]) * scale
                y = bottom + float(player["y"]) * scale
            except (KeyError, TypeError, ValueError):
                continue
            team = player.get("team")
            if player.get("role") == "referee":
                color = AppKit.NSColor.systemYellowColor()
            elif team == 0:
                color = AppKit.NSColor.systemRedColor()
            elif team == 1:
                color = AppKit.NSColor.systemBlueColor()
            else:
                color = AppKit.NSColor.systemGrayColor()
            color.setFill()
            AppKit.NSBezierPath.fillRect_(AppKit.NSMakeRect(x - 3, y - 3, 6, 6))

        ball = pitch.get("ball")
        if ball:
            try:
                x = left + float(ball["x"]) * scale
                y = bottom + float(ball["y"]) * scale
                AppKit.NSColor.whiteColor().setFill()
                AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                    AppKit.NSMakeRect(x - 3, y - 3, 6, 6)
                ).fill()
            except (KeyError, TypeError, ValueError):
                pass


class MonitorController(Foundation.NSObject):
    def initWithURL_(self, url):
        self = objc.super(MonitorController, self).init()
        if self is not None:
            self.url = url.rstrip("/") + "/tactical/live"
            self.window = None
            self.image_view = None
            self.pitch_view = None
            self.status = None
            self.timer = None
        return self

    def applicationDidFinishLaunching_(self, _notification):
        screen = AppKit.NSScreen.mainScreen().visibleFrame()
        width = min(760.0, max(520.0, screen.size.width * 0.46))
        height = min(880.0, max(620.0, screen.size.height * 0.82))
        frame = AppKit.NSMakeRect(
            screen.origin.x + screen.size.width - width - 18,
            screen.origin.y + screen.size.height - height - 18,
            width,
            height,
        )
        style = (
            AppKit.NSWindowStyleMaskTitled
            | AppKit.NSWindowStyleMaskClosable
            | AppKit.NSWindowStyleMaskResizable
            | AppKit.NSWindowStyleMaskMiniaturizable
        )
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, AppKit.NSBackingStoreBuffered, False
        )
        self.window.setTitle_("FootballVision Live Monitor")
        self.window.setLevel_(AppKit.NSFloatingWindowLevel)
        self.window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        self.window.setDelegate_(self)
        self.window.setMinSize_(AppKit.NSMakeSize(460, 560))
        self.window.setBackgroundColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.04, 0.05, 0.075, 1
        ))

        content = self.window.contentView()
        self.image_view = AppKit.NSImageView.alloc().initWithFrame_(AppKit.NSZeroRect)
        self.image_view.setImageScaling_(AppKit.NSImageScaleProportionallyUpOrDown)
        self.image_view.setImageAlignment_(AppKit.NSImageAlignCenter)
        self.pitch_view = PitchView.alloc().initWithFrame_(AppKit.NSZeroRect)
        self.status = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSZeroRect)
        self.status.setEditable_(False)
        self.status.setSelectable_(False)
        self.status.setBezeled_(False)
        self.status.setDrawsBackground_(False)
        self.status.setTextColor_(AppKit.NSColor.whiteColor())
        self.status.setFont_(AppKit.NSFont.systemFontOfSize_weight_(12, AppKit.NSFontWeightMedium))
        self.status.setStringValue_("Waiting for live analysis…")
        content.addSubview_(self.image_view)
        content.addSubview_(self.pitch_view)
        content.addSubview_(self.status)
        self._layout()

        self.window.makeKeyAndOrderFront_(None)
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        self.timer = Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.5, self, "refresh:", None, True
        )
        self.refresh_(None)

    def _layout(self):
        if self.window is None:
            return
        bounds = self.window.contentView().bounds()
        width, height = bounds.size.width, bounds.size.height
        status_h = 38.0
        pitch_h = max(190.0, height * 0.36)
        self.status.setFrame_(AppKit.NSMakeRect(14, 4, width - 28, status_h - 8))
        self.pitch_view.setFrame_(AppKit.NSMakeRect(0, status_h, width, pitch_h))
        self.image_view.setFrame_(
            AppKit.NSMakeRect(0, status_h + pitch_h, width, height - status_h - pitch_h)
        )

    def windowDidResize_(self, _notification):
        self._layout()

    def windowWillClose_(self, _notification):
        AppKit.NSApp.terminate_(None)

    def refresh_(self, _timer):
        try:
            with urllib.request.urlopen(self.url, timeout=0.35) as response:
                data = json.loads(response.read())
        except Exception:
            self.status.setStringValue_("Waiting for the local FootballVision server…")
            return

        payload = data.get("payload") or {}
        if payload.get("frame_b64"):
            raw = base64.b64decode(payload["frame_b64"])
            ns_data = Foundation.NSData.dataWithBytes_length_(raw, len(raw))
            image = AppKit.NSImage.alloc().initWithData_(ns_data)
            if image is not None:
                self.image_view.setImage_(image)
        self.pitch_view.setPayload_(payload)

        pitch = payload.get("pitch") or {}
        intel = pitch.get("intelligence") or {}
        possession = intel.get("possession") or {}
        team = possession.get("team")
        possession_text = "unresolved" if team not in (0, 1) else f"Team {'A' if team == 0 else 'B'}"
        paused = " · PAUSED" if data.get("paused") else ""
        self.status.setStringValue_(
            f"{'LIVE' if data.get('running') else 'IDLE'}{paused}  ·  "
            f"Frame {payload.get('frame_id', 0)}  ·  Possession {possession_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    controller = MonitorController.alloc().initWithURL_(args.url)
    app.setDelegate_(controller)
    app.run()


if __name__ == "__main__":
    main()
