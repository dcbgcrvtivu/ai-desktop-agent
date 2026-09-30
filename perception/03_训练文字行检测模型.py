# -*- coding: utf-8 -*-
"""训练 YOLOv8 文字行级检测模型。"""
from argparse import ArgumentParser
from pathlib import Path
import shutil

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="训练 YOLOv8 文字行检测模型。")
    parser.add_argument(
        "--work-dir",
        default="perception/work",
        help="文字行级数据集工作目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    dataset_yaml = work_dir / "dataset" / "text_line_dataset.yaml"
    runs_dir = work_dir / "runs"
    run_name = "text_line_detector_960"
    run_dir = runs_dir / run_name

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"未找到数据集配置文件：{dataset_yaml}")
    if run_dir.exists():
        shutil.rmtree(run_dir)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=960,
        batch=1,
        device="cpu",
        workers=0,
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        patience=15,
        seed=42,
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        save=True,
        plots=True,
        verbose=True,
    )

    best_model_path = run_dir / "weights" / "best.pt"
    print("\n文字行级 YOLO 模型训练结束。")
    print(f"最佳模型：{best_model_path}")
    print(f"最佳模型是否存在：{best_model_path.exists()}")


if __name__ == "__main__":
    main()
