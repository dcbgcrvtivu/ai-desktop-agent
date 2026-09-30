# -*- coding: utf-8 -*-
"""可选进阶：加载训练好的模型，对视频文件或摄像头进行实时文字行检测。"""
from argparse import ArgumentParser
from pathlib import Path
import shutil

import cv2
from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="YOLO 文字行视频检测（可选进阶功能）。")
    parser.add_argument(
        "--project-dir",
        default="perception/work",
        help="实验工程目录，默认：perception/work",
    )
    parser.add_argument(
        "--work-dir",
        default="perception/work",
        help="训练输出目录",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="视频文件路径，或摄像头编号（默认 0）。",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="设置后将检测视频保存到 output_results_line。",
    )
    return parser.parse_args()


def parse_source(source_text):
    return int(source_text) if source_text.isdigit() else source_text


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    weights_dir = project_dir / "weights"
    result_dir = project_dir / "output_results_line"
    weights_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_model = work_dir / "runs" / "text_line_detector_960" / "weights" / "best.pt"
    local_model = weights_dir / "text_line_yolo.pt"
    if not best_model.exists() and not local_model.exists():
        raise FileNotFoundError("未找到 best.pt。请先运行 03_训练文字行检测模型.py。")
    if best_model.exists():
        shutil.copy2(best_model, local_model)

    model = YOLO(str(local_model))
    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源：{args.source}")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = result_dir / "video_text_line_detection.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        print(f"将保存视频到：{output_path}")

    print("开始视频检测。按 q 键退出。")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            source=frame,
            imgsz=960,
            conf=0.12,
            iou=0.60,
            device="cpu",
            verbose=False,
        )[0]
        display = result.plot(labels=False, conf=False, line_width=2)
        cv2.imshow("YOLO Text Line Detection", display)
        if writer is not None:
            writer.write(display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
