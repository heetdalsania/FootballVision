# FootballVision

Real-time football (soccer) video analysis for local match footage and macOS
screen capture. The primary tactical dashboard detects and tracks players,
projects them onto a top-down pitch, separates teams, calculates team shape and
pitch-control metrics, and retrieves similar earlier situations.

Everything in the default workflow runs locally with free, open-source
components. It does not require an account, API key, or cloud upload.

## Requirements

- macOS for live screen/window capture (video-file analysis works elsewhere)
- Python 3.11 or 3.12 recommended
- About 450 MB for the three football model files

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
./scripts/download_weights.sh
```

The model script skips files already present in `weights/`. Recordings can be
uploaded directly from the source picker, or placed in `data/` or `uploads/`.
Uploads stay on this computer.

For live capture, grant the terminal or app running FootballVision access in
**System Settings → Privacy & Security → Screen & System Audio Recording**, then
restart that app.

## Run

```bash
source .venv/bin/activate
python main.py
```

Open <http://localhost:8000>. Choose a video, window, or display from **Start
Analysis**. A video file is the most reproducible way to verify the pipeline.
Sessions and sampled pitch states are saved in local SQLite history. From the
dashboard, **Build exports** creates tracking CSV, metrics CSV, and a visual
match report whenever the session contains resolved team data.

Useful options:

```bash
python main.py --host 127.0.0.1 --port 8080
python main.py --reload  # development only
python main.py --demo    # legacy synthetic Coach demo
```

Team separation uses a fast local jersey-colour model by default. The larger
public SigLIP model is optional and still needs no login or token; cache it
before kickoff so it never interrupts analysis:

```bash
python scripts/download_team_model.py
FOOTBALLVISION_TEAM_BACKEND=siglip python main.py
```

## Verify

```bash
python -m pip install -r requirements-dev.txt
./scripts/quality_gate.sh
```

The test suite covers geometry, team assignment, tactical metrics, situation
retrieval, report rendering, API source validation, and pipeline lifecycle
failures. Benchmarks and their measured outputs live in `benchmarks/`.

Run the real golden-video regression locally (about 15–30 seconds on a modern
machine):

```bash
python scripts/golden_video_benchmark.py
# or include it in the complete gate
RUN_GOLDEN=1 ./scripts/quality_gate.sh
```

The benchmark fails if football recognition, calibration, people recall,
projection, team resolution/stability, or latency crosses the checked-in
thresholds. GitHub Actions runs the deterministic test gate on every push and
pull request; the media/model benchmark stays local because large videos and
weights are intentionally not committed.

## Architecture

| Area | Responsibility |
| --- | --- |
| `ui/api.py` | FastAPI pages, REST endpoints, and WebSockets |
| `src/tactical_pipeline.py` | Source lifecycle and real-time analysis loop |
| `src/fv_engine.py` | Detection, tracking, team classification, calibration |
| `src/tactical_metrics.py` | Shape, compactness, pressing, and pitch control |
| `src/state_embedding.py` | Similar-situation indexing and retrieval |
| `src/capture.py` | macOS display/window capture |
| `src/match_report.py` | Static post-match analyst report |
| `src/session_report.py` | Report/CSV generation from saved live sessions |
| `db/session.py` | SQLite sessions, snapshots, and artifact metadata |

`src/pipeline.py` and the `/coach` and `/fan` pages are the earlier NFL-oriented
prototype. The root route intentionally opens the football tactical product.

## Notes

- Processing stays local; videos are not uploaded to an external service.
- `weights/`, videos, clips, captures, the SQLite database, and API keys are
  intentionally ignored by Git.
- Detection quality still depends on camera angle, resolution, occlusion, and
  the supplied model weights. Passing software tests does not guarantee perfect
  perception on every broadcast.

## License and attribution

See [LICENSE](LICENSE) and [NOTICE](NOTICE). The tactical engine adapts the
open-source [Roboflow Sports](https://github.com/roboflow/sports) soccer example.
