# NFL Vision - Dataset Documentation

This directory contains datasets and annotations for training the NFL Vision models.

## Directory Structure

```
data/
├── README.md           # This file
├── raw/                # Raw video clips and screenshots
│   ├── broadcasts/     # Recorded broadcast clips
│   └── screenshots/    # Screen capture frames
├── annotations/        # CVAT annotation exports
│   ├── players/        # Player bounding box annotations
│   └── formations/     # Formation labels per clip
├── processed/          # Preprocessed training data
│   ├── detection/      # YOLO format annotations
│   └── classification/ # Formation classification datasets
└── splits/             # Train/val/test splits
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Data Sources

### Primary Sources

1. **NFL Big Data Bowl** (https://www.kaggle.com/c/nfl-big-data-bowl-2024)
   - Player tracking data with positions
   - Play-by-play annotations
   - Formation labels

2. **Screen Captures**
   - Local screen recordings of NFL games
   - Used for detection model fine-tuning

### Licensing Note

> ⚠️ **Important**: Broadcast video rights must be confirmed before distribution.
> This project uses local-only processing. Do not redistribute captured content.

## Annotation Guidelines

### Player Detection

| Class | ID | Description |
|-------|-----|-------------|
| player | 0 | Any player on field |
| referee | 1 | Officials |
| ball | 2 | Football |
| coach | 3 | Sideline coaches |

**Bounding Box Format**: YOLO format (class_id, x_center, y_center, width, height)

### Formation Labels

| Formation | Code | Description |
|-----------|------|-------------|
| Shotgun | SHOT | QB 5+ yards behind center |
| Shotgun Spread | SHOT_SPR | Shotgun with 4+ wide receivers |
| I-Formation | I_FORM | FB directly behind QB, RB behind FB |
| Pro Set | PRO | Two backs beside each other |
| Single Back | SNGL | One running back |
| Pistol | PIST | QB 3-4 yards back, RB directly behind |
| 4-3 Defense | D43 | 4 linemen, 3 linebackers |
| 3-4 Defense | D34 | 3 linemen, 4 linebackers |
| Nickel | NICK | 5 defensive backs |
| Dime | DIME | 6 defensive backs |

## CVAT Annotation Workflow

See `../tools/annotation.md` for detailed CVAT setup and labeling instructions.

### Quick Start

1. Export frames: `python -m src.capture --save-frames`
2. Import to CVAT project
3. Annotate player bounding boxes
4. Export in YOLO format

## Preprocessing Scripts

```bash
# Convert CVAT exports to YOLO format
python tools/preprocess.py --input annotations/players --output processed/detection

# Generate formation classification dataset from tracking data
python tools/generate_formation_dataset.py --input data/nfl_big_data --output processed/classification

# Create train/val/test splits
python tools/split_dataset.py --input processed --output splits --ratios 0.7,0.15,0.15
```

## Dataset Statistics

*To be populated after data collection*

| Dataset | Samples | Classes | Notes |
|---------|---------|---------|-------|
| Detection (train) | - | 4 | Player, referee, ball, coach |
| Detection (val) | - | 4 | |
| Formation (train) | - | 18 | All formation types |
| Formation (val) | - | 18 | |

## Model Training

### Detection (YOLOv8)

```bash
yolo detect train data=data/processed/detection/data.yaml model=yolov8n.pt epochs=100
```

### Formation Classification

```python
from models.classifier import FormationClassifier

classifier = FormationClassifier()
history = classifier.train(
    dataset=training_data,
    epochs=100,
    save_path="models/weights/formation_classifier.pt"
)
```

## Evaluation

See `../eval/metrics.py` for evaluation code.

```bash
# Run full evaluation suite
python -m eval.metrics --detection-model models/yolov8_players.pt \
                       --classifier-model models/formation_classifier.pt \
                       --test-data data/splits/test.txt
```
