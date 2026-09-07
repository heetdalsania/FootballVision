#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="$ROOT_DIR/dist/FootballVision.app"

mkdir -p "$BUNDLE/Contents/MacOS"
cp "$ROOT_DIR/packaging/macos/Info.plist" "$BUNDLE/Contents/Info.plist"
cp "$ROOT_DIR/packaging/macos/FootballVision" "$BUNDLE/Contents/MacOS/FootballVision"
chmod +x "$BUNDLE/Contents/MacOS/FootballVision"

echo "Built $BUNDLE"
echo "Keep the app inside this project's dist folder so it can use the local environment."
