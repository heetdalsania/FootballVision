# Local football video data

FootballVision reads videos from this folder and from `uploads/`. Video files
remain local and are ignored by Git; confirm that you have the right to analyse
the footage and do not redistribute broadcast content.

## Suggested layout

```text
data/
├── sample.mp4
└── reliability/
    ├── daylight-wide.mp4
    ├── night-match.mp4
    ├── low-resolution.mp4
    ├── camera-pan.mp4
    └── partial-occlusion.mp4
```

The filenames are only examples. Add footage you are allowed to use, then run:

```bash
.venv/bin/python scripts/video_reliability_suite.py
```

The suite uses the same checked-in quality thresholds as the golden sample and
writes detailed local results to `benchmarks/video_suite_latest.json`. It needs
no account, API key, or cloud upload.

Corrections made in the dashboard can be downloaded as dataset JSON plus
tracking, metrics, and event CSV files. Those exports are suitable for local
review or conversion into a future training dataset.
