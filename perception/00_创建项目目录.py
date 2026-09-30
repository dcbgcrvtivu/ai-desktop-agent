# -*- coding: utf-8 -*-
"""创建 YOLO 文字区域检测实验所需目录。"""
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(description="创建 YOLO 文字区域检测实验工程目录。")
    parser.add_argument(
        "--project-dir",
        default="perception/work",
        help="工程目录名称或路径，默认：perception/work",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    folders = [
        project_dir / "input_screenshots",
        project_dir / "output_results_line",
        project_dir / "weights",
        project_dir / "reports" / "line_label_preview",
        project_dir / "easyocr_models",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print("实验工程创建完成：")
    print(project_dir)
    print("\\n目录结构：")
    for folder in folders:
        print("├──", folder.relative_to(project_dir))


if __name__ == "__main__":
    main()
