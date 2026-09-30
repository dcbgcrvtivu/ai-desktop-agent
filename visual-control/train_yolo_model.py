from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "calculator_yolo" / "data.yaml"
RUNS_DIR = ROOT / "runs"
ULTRALYTICS_CONFIG = ROOT / ".ultralytics"
ULTRALYTICS_CONFIG.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_CONFIG)

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATA_YAML),
        epochs=12,
        imgsz=640,
        batch=8,
        patience=4,
        project=str(RUNS_DIR),
        name="calculator_yolo_train",
        exist_ok=True,
        workers=0,
        device="cpu",
        seed=7,
        verbose=True,
    )
    best = RUNS_DIR / "calculator_yolo_train" / "weights" / "best.pt"
    last = RUNS_DIR / "calculator_yolo_train" / "weights" / "last.pt"
    print(f"best={best}")
    print(f"last={last}")
    print(results)


if __name__ == "__main__":
    main()
