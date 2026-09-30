from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
(ROOT / ".matplotlib").mkdir(exist_ok=True)

from ultralytics import YOLO


WEIGHTS = ROOT / "runs" / "calculator_yolo_train" / "weights" / "best.pt"
SOURCE = ROOT / "dataset" / "calculator_yolo" / "images" / "val" / "calculator_val_0000.png"
OUT_DIR = ROOT / "output" / "trained_yolo"
IMAGE_OUT = ROOT / "output" / "images" / "07_trained_yolo_prediction.png"
JSON_OUT = OUT_DIR / "prediction_val_0000.json"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    results = model.predict(
        source=str(SOURCE),
        imgsz=640,
        conf=0.25,
        project=str(OUT_DIR),
        name="predict_val_0000",
        exist_ok=True,
        save=True,
        verbose=False,
    )
    result = results[0]
    plotted = result.plot()
    from PIL import Image

    Image.fromarray(plotted[..., ::-1]).save(IMAGE_OUT)
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        detections.append(
            {
                "class_id": cls_id,
                "label": model.names[cls_id],
                "confidence": round(float(box.conf.item()), 4),
                "xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
            }
        )
    JSON_OUT.write_text(json.dumps({"weights": str(WEIGHTS), "source": str(SOURCE), "detections": detections}, indent=2), encoding="utf-8")
    print(f"image={IMAGE_OUT}")
    print(f"json={JSON_OUT}")
    print(f"detections={len(detections)}")


if __name__ == "__main__":
    main()
