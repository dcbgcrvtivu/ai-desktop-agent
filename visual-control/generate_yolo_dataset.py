from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset" / "calculator_yolo"
IMAGE_SIZE = 640
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "plus", "minus", "multiply", "divide", "equals", "clear", "display"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_BIG = font(32, True)
FONT_KEY = font(30, True)
FONT_SMALL = font(15)


def class_id(label: str) -> int:
    return CLASSES.index(label)


def yolo_box(box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / IMAGE_SIZE
    cy = (y1 + y2) / 2 / IMAGE_SIZE
    w = (x2 - x1) / IMAGE_SIZE
    h = (y2 - y1) / IMAGE_SIZE
    return cx, cy, w, h


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str, radius: int = 10, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def render_sample(index: int, split: str) -> None:
    rng = random.Random(index * 1009 + (0 if split == "train" else 97))
    bg = rng.choice(["#eef2f7", "#f8fafc", "#edf6f9", "#f4f4f5"])
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), bg)
    draw = ImageDraw.Draw(img)

    margin_x = rng.randint(35, 95)
    margin_y = rng.randint(25, 80)
    panel_w = rng.randint(390, 500)
    panel_h = rng.randint(510, 585)
    panel = (margin_x, margin_y, margin_x + panel_w, margin_y + panel_h)
    rounded(draw, panel, rng.choice(["#ffffff", "#f1f5f9", "#e5e7eb"]), "#aab7c4", radius=14, width=2)
    draw.text((panel[0] + 28, panel[1] + 22), "Calculator", font=FONT_SMALL, fill="#1f2937")

    display_h = rng.randint(62, 78)
    display = (panel[0] + 28, panel[1] + 58, panel[2] - 28, panel[1] + 58 + display_h)
    rounded(draw, display, "#0f172a", "#111827", radius=6, width=2)
    value = rng.choice(["0", "12", "19", "18", "7", "35", "42"])
    draw.text((display[2] - 35 - len(value) * 16, display[1] + 16), value, font=FONT_BIG, fill="#dcfce7")

    labels_grid = [
        ["7", "8", "9", "divide"],
        ["4", "5", "6", "multiply"],
        ["1", "2", "3", "minus"],
        ["0", "clear", "equals", "plus"],
    ]
    printable = {"divide": "/", "multiply": "*", "minus": "-", "plus": "+", "equals": "=", "clear": "C"}

    key_w = int((panel_w - 72 - 3 * 14) / 4)
    key_h = rng.randint(54, 66)
    gap_x = 14
    gap_y = rng.randint(14, 20)
    grid_x = panel[0] + 36
    grid_y = display[3] + rng.randint(35, 48)

    labels: list[str] = [f"{class_id('display')} {' '.join(f'{v:.6f}' for v in yolo_box(display))}"]
    for row, row_labels in enumerate(labels_grid):
        for col, label in enumerate(row_labels):
            x1 = grid_x + col * (key_w + gap_x)
            y1 = grid_y + row * (key_h + gap_y)
            box = (x1, y1, x1 + key_w, y1 + key_h)
            fill = rng.choice(["#ffffff", "#f8fafc"])
            if label in {"divide", "multiply", "minus", "plus", "equals"}:
                fill = rng.choice(["#e0f2fe", "#dbeafe"])
            rounded(draw, box, fill, "#94a3b8", radius=7, width=2)
            text = printable.get(label, label)
            tw = draw.textlength(text, font=FONT_KEY)
            draw.text((x1 + key_w / 2 - tw / 2, y1 + 12), text, font=FONT_KEY, fill="#111827")
            labels.append(f"{class_id(label)} {' '.join(f'{v:.6f}' for v in yolo_box(box))}")

    if rng.random() < 0.35:
        for _ in range(rng.randint(3, 8)):
            x = rng.randint(0, IMAGE_SIZE - 1)
            y = rng.randint(0, IMAGE_SIZE - 1)
            color = rng.choice(["#e2e8f0", "#cbd5e1", "#f1f5f9"])
            draw.rectangle((x, y, min(x + rng.randint(5, 18), IMAGE_SIZE), min(y + rng.randint(3, 12), IMAGE_SIZE)), fill=color)

    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"calculator_{split}_{index:04d}.png"
    label_path = label_dir / f"calculator_{split}_{index:04d}.txt"
    img.save(image_path)
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def write_yaml() -> None:
    yaml_text = "\n".join(
        [
            f"path: {DATASET_ROOT.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            *[f"  {idx}: {name}" for idx, name in enumerate(CLASSES)],
            "",
        ]
    )
    (DATASET_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    for i in range(220):
        render_sample(i, "train")
    for i in range(45):
        render_sample(i, "val")
    write_yaml()
    print(DATASET_ROOT / "data.yaml")


if __name__ == "__main__":
    main()
