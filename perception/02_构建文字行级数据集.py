# -*- coding: utf-8 -*-
"""使用 EasyOCR 生成文字行级 YOLO 数据集。"""
from argparse import ArgumentParser
from pathlib import Path
import csv
import shutil

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageEnhance


def parse_args():
    parser = ArgumentParser(description="构建文字行级 YOLO 数据集。")
    parser.add_argument(
        "--project-dir",
        default="perception/work",
        help="实验工程目录，默认：perception/work",
    )
    parser.add_argument(
        "--work-dir",
        default="perception/work",
        help="英文路径的数据集工作目录",
    )
    return parser.parse_args()


def read_image_safe(image_path):
    """兼容 Windows 中文路径的图像读取。"""
    file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"图片读取失败：{image_path}")
    return image_bgr


def bbox_to_rect(bbox, image_width, image_height):
    points = np.array(bbox, dtype=np.float32)
    x1 = max(0, int(np.floor(points[:, 0].min())))
    y1 = max(0, int(np.floor(points[:, 1].min())))
    x2 = min(image_width - 1, int(np.ceil(points[:, 0].max())))
    y2 = min(image_height - 1, int(np.ceil(points[:, 1].max())))
    return [x1, y1, x2, y2]


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0


def remove_near_duplicate_boxes(boxes_with_confidence):
    sorted_boxes = sorted(boxes_with_confidence, key=lambda item: item[4], reverse=True)
    kept = []
    for candidate in sorted_boxes:
        if not any(box_iou(candidate[:4], item[:4]) >= 0.85 for item in kept):
            kept.append(candidate)
    return kept


def merge_boxes_into_text_lines(boxes_with_confidence):
    """先按纵向中心聚类，再合并同一行中相邻的 OCR 框。"""
    if not boxes_with_confidence:
        return []

    def center_y(box):
        return (box[1] + box[3]) / 2

    def box_height(box):
        return box[3] - box[1]

    line_groups = []
    for box in sorted(boxes_with_confidence, key=center_y):
        current_center_y, current_height = center_y(box), box_height(box)
        best_group_index, best_distance = None, float("inf")
        for group_index, group in enumerate(line_groups):
            group_center_y = float(np.median([center_y(item) for item in group]))
            group_height = float(np.median([box_height(item) for item in group]))
            allowed_distance = max(8, 0.45 * (current_height + group_height))
            distance = abs(current_center_y - group_center_y)
            if distance <= allowed_distance and distance < best_distance:
                best_group_index, best_distance = group_index, distance
        if best_group_index is None:
            line_groups.append([box])
        else:
            line_groups[best_group_index].append(box)

    merged_lines = []
    for group in line_groups:
        group = sorted(group, key=lambda item: item[0])
        median_height = float(np.median([box_height(item) for item in group]))
        gap_threshold = max(25, int(1.8 * median_height))
        current_x1, current_y1, current_x2, current_y2 = group[0][:4]
        for next_box in group[1:]:
            x1, y1, x2, y2 = next_box[:4]
            if x1 - current_x2 <= gap_threshold:
                current_x1, current_y1 = min(current_x1, x1), min(current_y1, y1)
                current_x2, current_y2 = max(current_x2, x2), max(current_y2, y2)
            else:
                merged_lines.append([current_x1, current_y1, current_x2, current_y2])
                current_x1, current_y1, current_x2, current_y2 = x1, y1, x2, y2
        merged_lines.append([current_x1, current_y1, current_x2, current_y2])

    return [box for box in merged_lines if box[2] - box[0] >= 20 and box[3] - box[1] >= 8]


def rect_to_yolo_line(x1, y1, x2, y2, image_width, image_height):
    center_x = (x1 + x2) / 2 / image_width
    center_y = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def save_yolo_label(label_path, yolo_lines):
    label_path.write_text("\n".join(yolo_lines), encoding="utf-8")


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    input_dir = project_dir / "input_screenshots"
    ocr_model_dir = project_dir / "easyocr_models"
    preview_dir = project_dir / "reports" / "line_label_preview"
    work_dir = Path(args.work_dir).resolve()
    dataset_dir = work_dir / "dataset"
    train_image_dir = dataset_dir / "images" / "train"
    val_image_dir = dataset_dir / "images" / "val"
    train_label_dir = dataset_dir / "labels" / "train"
    val_label_dir = dataset_dir / "labels" / "val"

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if len(image_paths) < 4:
        raise ValueError("截图数量不足，至少需要 4 张图片。")

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for folder in [train_image_dir, val_image_dir, train_label_dir, val_label_dir, preview_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    val_names = {image_paths[1].name, image_paths[-1].name}
    reader = easyocr.Reader(["ch_sim", "en"], gpu=False, model_storage_directory=str(ocr_model_dir), verbose=False)
    summary_rows = []

    for index, image_path in enumerate(image_paths, start=1):
        image_bgr = read_image_safe(image_path)
        image_height, image_width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ocr_results = reader.readtext(image_bgr, detail=1, paragraph=False, min_size=10)

        raw_boxes = []
        for bbox, _, confidence in ocr_results:
            x1, y1, x2, y2 = bbox_to_rect(bbox, image_width, image_height)
            if confidence >= 0.20 and x2 - x1 >= 15 and y2 - y1 >= 8:
                raw_boxes.append([x1, y1, x2, y2, float(confidence)])

        line_boxes = merge_boxes_into_text_lines(remove_near_duplicate_boxes(raw_boxes))
        yolo_lines = [rect_to_yolo_line(x1, y1, x2, y2, image_width, image_height) for x1, y1, x2, y2 in line_boxes]

        preview_rgb = image_rgb.copy()
        for x1, y1, x2, y2 in line_boxes:
            cv2.rectangle(preview_rgb, (x1, y1), (x2, y2), (0, 120, 255), 2)
        Image.fromarray(preview_rgb).save(preview_dir / f"{image_path.stem}_line_boxes.png")

        if image_path.name in val_names:
            Image.fromarray(image_rgb).save(val_image_dir / f"{image_path.stem}.jpg", quality=95)
            save_yolo_label(val_label_dir / f"{image_path.stem}.txt", yolo_lines)
            split_name, saved_samples = "val", 1
        else:
            pil_image = Image.fromarray(image_rgb)
            augmented_images = {
                "original": pil_image,
                "bright": ImageEnhance.Brightness(pil_image).enhance(1.22),
                "dark": ImageEnhance.Brightness(pil_image).enhance(0.75),
                "contrast": ImageEnhance.Contrast(pil_image).enhance(1.35),
                "sharp": ImageEnhance.Sharpness(pil_image).enhance(1.60),
            }
            for aug_name, aug_image in augmented_images.items():
                aug_image.save(train_image_dir / f"{image_path.stem}_{aug_name}.jpg", quality=95)
                save_yolo_label(train_label_dir / f"{image_path.stem}_{aug_name}.txt", yolo_lines)
            split_name, saved_samples = "train", len(augmented_images)

        summary_rows.append([image_path.name, split_name, len(ocr_results), len(raw_boxes), len(line_boxes), saved_samples])
        print(f"[{index}/{len(image_paths)}] {image_path.name}: 文字行框={len(line_boxes)}，划分={split_name}")

    yaml_text = f"""# 文字行级检测数据集\npath: {dataset_dir.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: text_line\n"""
    yaml_path = dataset_dir / "text_line_dataset.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    summary_path = work_dir / "line_dataset_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["原始截图", "数据集划分", "OCR初始框数量", "过滤后框数量", "文字行框数量", "生成样本数量"])
        writer.writerows(summary_rows)

    print("\n文字行级 YOLO 数据集生成完成。")
    print(f"训练图片数量：{len(list(train_image_dir.glob('*.jpg')))}")
    print(f"验证图片数量：{len(list(val_image_dir.glob('*.jpg')))}")
    print(f"数据集配置文件：{yaml_path}")


if __name__ == "__main__":
    main()
