Plan: NFL Vision App (High-level roadmap)

TL;DR: Build a hybrid macOS-first prototype: local screen capture → real-time player detection + tracking → temporal classifier for plays/formations → overlay UI; iterate toward higher accuracy and lower latency. Start with YOLOv8 + ByteTrack + a lightweight temporal model, expand with cloud-assisted refinement, improved datasets, and tooling for annotations.

Steps
Prototype capture and detection: implement macOS screen capture, run YOLOv8 detection, persist frames ([example repo: ultralytics/ultralytics]).
Add tracking: integrate ByteTrack/DeepSORT to assign persistent player IDs and outputs.
Build temporal classifier: aggregate player positions → train small transformer/TCN for formation/scheme classification using NFL Big Data Bowl + annotated clips.
UI overlay & tallies: create Electron/SwiftUI overlay showing detections, scheme labels, and occurrence counters; add mode presets (casual/advanced/athlete).
Data & ops: set up CVAT annotation pipeline, evaluation harness (mAP, MOTA/IDF1, formation accuracy), and a system for periodic model retraining.
Further Considerations
Licensing: confirm broadcast/video rights before any distribution; prefer local-only processing or licensed feeds.
Mode trade-offs: Option A — casual first (fast, lower accuracy); Option B — athlete-first (requires hardware + strict latency).
Hardware: develop on Apple Silicon; scale training/inference on NVIDIA GPUs (TensorRT/ONNX for production).




Plan: 2–3 Week Prototype Sprint

TL;DR: Deliver a macOS-first hybrid prototype that captures screen video, runs real-time player detection + tracking, applies a lightweight temporal classifier for plays/formations, and displays an overlay with tallies and three mode presets. Focus: fast iteration—YOLOv8 + ByteTrack locally, dataset pipeline + CVAT, evaluation harness.

Steps
1. Initialize repo and screen capture: implement `src/capture.py` with `ScreenCapture` to stream/save frames.
2. Add detection and tracking: integrate `src/detector.py` (`Detector` using YOLOv8) and `src/tracker.py` (`Tracker` using ByteTrack).
3. Build dataset & classifier: create `data/README.md` and `models/classifier.py` implementing `FormationClassifier` (temporal transformer/TCN).
4. UI overlay and tallies: implement `ui/overlay.py` and `ui/app.py` with `OverlayApp` supporting casual/advanced/athlete modes.
5. Annotation & evaluation: setup `tools/annotation.md` for CVAT and `eval/metrics.py` exposing `Evaluator` (mAP, MOTA, IDF1, formation accuracy).

Further Considerations
- Licensing: confirm broadcast rights before saving or distributing video—Option A local-only, Option B licensed feeds.
- Hardware: develop on Apple Silicon; plan cloud GPUs (NVIDIA A10/3090) for training.
- Prioritization: start with casual-mode accuracy and latency; iterate toward athlete-grade performance.

Next actions
- Confirm scope or timeline changes.
- If confirmed, scaffold repository files and implement `src/capture.py` next.

