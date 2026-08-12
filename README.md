# NFL Vision

AI-powered video analysis for NFL game film. Like PlayVision, but for football.

## What It Does

```
Upload Video → Detect Players → Track Movement → Tag Plays → Generate Stats
```

1. **Upload** game/practice footage (MP4, MOV)
2. **Process** - AI detects and tracks all 22 players
3. **Analyze** - Auto-segments plays, classifies formations
4. **Review** - Search plays, view stats, export reports

## Install

```bash
pip install -r requirements.txt
```

## Usage

### Process a Video
```bash
python main.py process game_film.mp4
```

### View Analysis
```bash
python main.py analyze game_film_analysis.json
```

### Start Web Server (coming soon)
```bash
python main.py serve
```

## How It Works

| Component | Purpose |
|-----------|---------|
| `src/detector.py` | YOLO player detection |
| `src/tracker.py` | ByteTrack multi-player tracking |
| `src/video_processor.py` | Batch video analysis engine |
| `src/analytics.py` | Statistics and report generation |
| `models/classifier.py` | Formation classification |
| `models/coverage_classifier.py` | Defensive coverage detection |

## Output

Analysis is saved as JSON:
```json
{
  "video_name": "game.mp4",
  "total_plays": 45,
  "plays": [
    {
      "play_id": "play_001",
      "offense_formation": "shotgun",
      "coverage": "cover_2",
      "duration_ms": 4200
    }
  ]
}
```

## Tech Stack

- **Detection**: YOLOv8 (ultralytics)
- **Tracking**: ByteTrack (supervision)
- **ML**: PyTorch
- **Video**: OpenCV

## Inspired By

- [Roboflow Sports](https://github.com/roboflow/sports)
- [PlayVision](https://playvision.io)
