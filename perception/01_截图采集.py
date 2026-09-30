# -*- coding: utf-8 -*-
"""F8 截图工具：按 F8 保存整个屏幕，按 F9 退出。"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from PIL import ImageGrab
from pynput import keyboard


def parse_args():
    parser = ArgumentParser(description="将屏幕截图保存到 YOLO 实验工程。")
    parser.add_argument(
        "--project-dir",
        default="perception/work",
        help="实验工程目录，默认：perception/work",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    screenshot_dir = project_dir / "input_screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    print("YOLO 文字区域检测截图工具已启动")
    print("按 F8：截图并保存")
    print("按 F9：退出")
    print(f"保存目录：{screenshot_dir}")

    def save_screenshot():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = screenshot_dir / f"screenshot_{timestamp}.png"
        image = ImageGrab.grab()
        image.save(save_path)
        print(f"已保存：{save_path.name}，尺寸：{image.size}")

    def on_press(key):
        if key == keyboard.Key.f8:
            save_screenshot()
        elif key == keyboard.Key.f9:
            print("截图工具已退出。")
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()
