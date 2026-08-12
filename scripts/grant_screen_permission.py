#!/usr/bin/env python3
"""
Register this terminal for macOS Screen Recording permission.

macOS only lists an app under System Settings → Privacy & Security → Screen &
System Audio Recording *after* that app has attempted a screen capture. A
terminal that has never tried will simply not appear in the list, which is
why you cannot find it there yet.

Running this script performs that first attempt, which makes macOS show the
permission prompt and adds the terminal to the list.

Usage (must be run from Terminal.app / iTerm, NOT from an editor or agent):

    cd ~/Documents/Career/Projects/FootballVision
    .venv/bin/python scripts/grant_screen_permission.py
"""

import sys

try:
    import Quartz
except ImportError:
    sys.exit("pyobjc-framework-Quartz missing. Run: pip install pyobjc-framework-Quartz")


def status() -> bool:
    return bool(Quartz.CGPreflightScreenCaptureAccess())


def main() -> None:
    print("=" * 68)
    print("  FootballVision — macOS Screen Recording permission helper")
    print("=" * 68)

    before = status()
    print(f"\nCurrent permission: {'GRANTED' if before else 'NOT GRANTED'}")

    if before:
        print("\nNothing to do — this terminal can already capture the screen.")
        print("Start the server with:")
        print("    .venv/bin/python main.py --port 8077")
        return

    print("\nRequesting access — a macOS dialog should appear now…")
    Quartz.CGRequestScreenCaptureAccess()

    # Force an actual capture attempt too: on some macOS versions this is what
    # actually registers the app in the Privacy list.
    try:
        Quartz.CoreGraphics.CGDisplayCreateImage(Quartz.CGMainDisplayID())
    except Exception:
        pass

    print(f"Permission now reads: {'GRANTED' if status() else 'NOT GRANTED'}")
    print("\nNEXT STEPS")
    print("-" * 68)
    print("1. Open System Settings → Privacy & Security →")
    print("   'Screen & System Audio Recording'.")
    print("2. Your terminal (Terminal or iTerm) should now be listed —")
    print("   turn its switch ON.")
    print("3. QUIT the terminal completely (Cmd+Q — not just close the window).")
    print("   macOS only applies this permission to a freshly started app.")
    print("4. Reopen it and start the server:")
    print("     cd ~/Documents/Career/Projects/FootballVision")
    print("     .venv/bin/python main.py --port 8077")
    print("5. Re-run this script any time to confirm it reads GRANTED.")
    print("-" * 68)


if __name__ == "__main__":
    main()
