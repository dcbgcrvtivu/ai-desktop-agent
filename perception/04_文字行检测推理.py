# -*- coding: utf-8 -*-
"""加载训练好的 best.pt，对屏幕截图进行文字行检测。"""
from argparse import ArgumentParser
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="执行 YOLO 文字行检测。")
    parser.add_argument(
        "--project-dir",
        default="perception/work",
        help="实验工程目录，默认：perception/work",
    )
    parser.add_argument(
        "--work-dir",
        default="perception/work",
        help="文字行级训练输出目录",
    )
    return parser.parse_args()


def read_image_safe(image_path):
    """兼容 Windows 中文路径。"""
    file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"图片读取失败：{image_path}")
    return image_bgr


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    input_dir = project_dir / "input_screenshots"
    weights_dir = project_dir / "weights"
    result_dir = project_dir / "output_results_line"
    weights_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = work_dir / "runs" / "text_line_detector_960" / "weights" / "best.pt"
    target_model_path = weights_dir / "text_line_yolo.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"未找到最佳模型：{best_model_path}")
    shutil.copy2(best_model_path, target_model_path)

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not image_paths:
        raise FileNotFoundError("input_screenshots 文件夹中没有图片。")
    for old_file in result_dir.glob("*_line_yolo.png"):
        old_file.unlink()

    model = YOLO(str(target_model_path))
    for index, image_path in enumerate(image_paths, start=1):
        results = model.predict(
            source=read_image_safe(image_path),
            imgsz=960,
            conf=0.12,
            iou=0.60,
            device="cpu",
            verbose=False,
        )
        result = results[0]
        annotated_bgr = result.plot(labels=False, conf=False, line_width=2)
        save_path = result_dir / f"{image_path.stem}_line_yolo.png"
        Image.fromarray(annotated_bgr[:, :, ::-1]).save(save_path)
        box_count = len(result.boxes) if result.boxes is not None else 0
        print(f"[{index}/{len(image_paths)}] {image_path.name}：检测框数量={box_count}")

    print(f"\n检测完成，结果保存目录：{result_dir}")


if __name__ == "__main__":
    main()
