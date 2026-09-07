#!/usr/bin/env python3
"""
FootballVision — Real-Time Game Analyzer
Entry point. Starts the FastAPI/uvicorn server.

Usage:
    python main.py                        # default port 8000
    python main.py --port 8080
    python main.py --demo                 # use synthetic DemoSource (no screen capture)
    python main.py --host 127.0.0.1
"""

import argparse
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║   FootballVision — Real-Time Football Analysis                ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   OPEN  →  http://localhost:{port}/                             ║
║                                                               ║
║   Top-down pitch · automatic calibration · pitch control      ║
║   pressing · team shape · similar-situation retrieval         ║
╚═══════════════════════════════════════════════════════════════╝

  Press "Start Analysis", pick the window or screen showing the
  match, and analysis begins.            Ctrl+C to stop.
"""


def main():
    parser = argparse.ArgumentParser(description="FootballVision Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic DemoSource (no real screen capture)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    args = parser.parse_args()

    # Pass demo flag to pipeline via env var (picked up by api.py startup)
    if args.demo:
        os.environ["FOOTBALLVISION_DEMO"] = "1"

    print(BANNER.format(port=args.port))

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run:\n  pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        "ui.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
