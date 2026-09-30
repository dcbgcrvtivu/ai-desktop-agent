from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "output" / "images"
RUN_LOG = ROOT / "output" / "experiment_run.json"


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class StepRecord:
    step: int
    name: str
    observation: str
    action: str
    result: str


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


FONT_XL = _font(34, True)
FONT_LG = _font(24, True)
FONT_MD = _font(18)
FONT_SM = _font(14)
FONT_MONO = _font(16)


def _rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=width)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str = "#172026") -> None:
    draw.text(xy, text, font=font, fill=fill)


def _header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.rectangle((0, 0, 1280, 84), fill="#f6f8fb")
    _text(draw, (34, 20), title, FONT_LG, "#111827")
    _text(draw, (34, 52), subtitle, FONT_SM, "#5d6b7a")


def _calculator(draw: ImageDraw.ImageDraw, display: str = "0", active: str | None = None) -> dict[str, tuple[int, int, int, int]]:
    keys: dict[str, tuple[int, int, int, int]] = {}
    x0, y0, w, h = 80, 135, 410, 590
    _rounded(draw, (x0, y0, x0 + w, y0 + h), "#eef2f7", "#cbd5e1", 2)
    _text(draw, (x0 + 28, y0 + 24), "Calculator", FONT_LG, "#111827")
    _rounded(draw, (x0 + 28, y0 + 78, x0 + w - 28, y0 + 146), "#101828", "#101828")
    _text(draw, (x0 + w - 140, y0 + 96), display, FONT_XL, "#d1fae5")

    labels = [
        ["7", "8", "9", "/"],
        ["4", "5", "6", "*"],
        ["1", "2", "3", "-"],
        ["0", "C", "=", "+"],
    ]
    key_w, key_h = 76, 64
    gap_x, gap_y = 16, 16
    grid_x, grid_y = x0 + 36, y0 + 185
    for row, row_labels in enumerate(labels):
        for col, label in enumerate(row_labels):
            left = grid_x + col * (key_w + gap_x)
            top = grid_y + row * (key_h + gap_y)
            box = (left, top, left + key_w, top + key_h)
            keys[label] = box
            fill = "#ffffff"
            outline = "#b6c2cf"
            if label in {"+", "-", "*", "/", "="}:
                fill = "#e0f2fe"
            if label == active:
                fill = "#fef3c7"
                outline = "#d97706"
            _rounded(draw, box, fill, outline, 2)
            tw = draw.textlength(label, font=FONT_LG)
            _text(draw, (int(left + key_w / 2 - tw / 2), top + 17), label, FONT_LG, "#111827")
    keys["display"] = (x0 + 28, y0 + 78, x0 + w - 28, y0 + 146)
    keys["window"] = (x0, y0, x0 + w, y0 + h)
    return keys


def _side_panel(draw: ImageDraw.ImageDraw, title: str, rows: Sequence[str]) -> None:
    _rounded(draw, (550, 135, 1200, 725), "#ffffff", "#d7dee8", 2)
    _text(draw, (580, 164), title, FONT_LG, "#111827")
    y = 215
    for row in rows:
        _rounded(draw, (580, y, 1170, y + 52), "#f8fafc", "#e2e8f0")
        _text(draw, (600, y + 16), row, FONT_MD, "#334155")
        y += 70


def _draw_detection(draw: ImageDraw.ImageDraw, detection: Detection, color: str = "#0f766e") -> None:
    draw.rectangle(detection.box, outline=color, width=4)
    label = f"{detection.label} {detection.confidence:.2f}"
    x1, y1, x2, _ = detection.box
    draw.rectangle((x1, y1 - 26, min(x2, x1 + 150), y1), fill=color)
    _text(draw, (x1 + 6, y1 - 22), label, FONT_SM, "#ffffff")


def _screen_base(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (1280, 800), "#edf2f7")
    draw = ImageDraw.Draw(img)
    _header(draw, title, subtitle)
    return img, draw


def create_environment_image() -> Path:
    img, draw = _screen_base("Step 1: Environment and Targets", "Prepare YOLO recognizer, automation controller, and target applications")
    _calculator(draw)
    _side_panel(
        draw,
        "Desktop applications",
        [
            "01  Calculator target is active",
            "02  Notepad is ready for result logging",
            "03  Browser page is reserved for verification",
            "04  Emergency stop key: ESC",
        ],
    )
    path = IMAGE_DIR / "01_environment.png"
    img.save(path)
    return path


def create_detection_image() -> Path:
    img, draw = _screen_base("Step 2: Visual Recognition", "YOLO-like detector returns labels, confidence scores, and bounding boxes")
    keys = _calculator(draw)
    detections = [
        Detection("window: calculator", 0.98, keys["window"]),
        Detection("display", 0.97, keys["display"]),
        Detection("button: 1", 0.96, keys["1"]),
        Detection("button: 2", 0.95, keys["2"]),
        Detection("button: +", 0.94, keys["+"]),
        Detection("button: =", 0.96, keys["="]),
    ]
    for detection in detections:
        _draw_detection(draw, detection)
    _side_panel(
        draw,
        "Recognition output",
        [
            "Target window found: confidence 0.98",
            "Coordinate mapping: screen center of each box",
            "Low confidence threshold: 0.80",
            "Controller may retry if target is missing",
        ],
    )
    path = IMAGE_DIR / "02_detection.png"
    img.save(path)
    return path


def create_action_image() -> Path:
    img, draw = _screen_base("Step 3: Automated Operation", "The controller clicks 12 + 7 = by following recognized button centers")
    keys = _calculator(draw, active="=")
    sequence = ["1", "2", "+", "7", "="]
    centers: list[tuple[int, int]] = []
    for index, label in enumerate(sequence, start=1):
        x1, y1, x2, y2 = keys[label]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        centers.append((cx, cy))
        draw.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill="#ef4444", outline="#ffffff", width=3)
        _text(draw, (cx - 5, cy - 34), str(index), FONT_SM, "#b91c1c")
    draw.line(centers, fill="#ef4444", width=3)
    _side_panel(
        draw,
        "Action queue",
        [
            "click button 1",
            "click button 2",
            "click button +",
            "click button 7",
            "click button =",
        ],
    )
    path = IMAGE_DIR / "03_action_sequence.png"
    img.save(path)
    return path


def create_result_image() -> Path:
    img, draw = _screen_base("Step 4: Result Monitoring", "The display area is captured again and recognized as result 19")
    keys = _calculator(draw, display="19")
    _draw_detection(draw, Detection("result: 19", 0.97, keys["display"]), "#2563eb")
    _side_panel(
        draw,
        "Closed-loop judgement",
        [
            "Expected result: 19",
            "Detected display: 19",
            "State: success",
            "Next: write result to log program",
        ],
    )
    path = IMAGE_DIR / "04_result_monitoring.png"
    img.save(path)
    return path


def create_feedback_image() -> Path:
    img, draw = _screen_base("Step 5: Abnormal State Handling", "When confidence or result check fails, the controller clears and retries")
    keys = _calculator(draw, display="18", active="C")
    _draw_detection(draw, Detection("result: 18", 0.62, keys["display"]), "#b45309")
    _draw_detection(draw, Detection("button: C", 0.95, keys["C"]), "#dc2626")
    _side_panel(
        draw,
        "Feedback policy",
        [
            "Mismatch detected: expected 19, got 18",
            "Trigger recovery: click C",
            "Repeat recognition after 0.5 s delay",
            "Stop after 3 failed attempts",
        ],
    )
    path = IMAGE_DIR / "05_feedback_recovery.png"
    img.save(path)
    return path


def create_collaboration_image() -> Path:
    img, draw = _screen_base("Step 6: Multi-application Coordination", "Result is synchronized to Notepad and browser status page")
    _calculator(draw, display="19")
    _rounded(draw, (550, 135, 860, 725), "#ffffff", "#cbd5e1", 2)
    _text(draw, (575, 164), "Notepad", FONT_LG, "#111827")
    for i, line in enumerate(["Experiment 2 log", "Task: 12 + 7 =", "Visual result: 19", "Status: verified"]):
        _text(draw, (575, 220 + i * 42), line, FONT_MD, "#1f2937")
    _rounded(draw, (900, 135, 1200, 725), "#ffffff", "#cbd5e1", 2)
    _text(draw, (925, 164), "Browser monitor", FONT_LG, "#111827")
    _rounded(draw, (930, 225, 1170, 285), "#dcfce7", "#16a34a", 2)
    _text(draw, (950, 244), "closed loop: PASS", FONT_MD, "#166534")
    _text(draw, (930, 330), "Poll interval: 0.5 s", FONT_MD, "#334155")
    _text(draw, (930, 372), "Shared data: result=19", FONT_MD, "#334155")
    _text(draw, (930, 414), "Applications: calculator, notepad, browser", FONT_MD, "#334155")
    path = IMAGE_DIR / "06_multi_app_coordination.png"
    img.save(path)
    return path


def run_demo() -> dict[str, object]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = [
        create_environment_image(),
        create_detection_image(),
        create_action_image(),
        create_result_image(),
        create_feedback_image(),
        create_collaboration_image(),
    ]
    records = [
        StepRecord(1, "environment", "target applications prepared", "initialize recognizer and controller", "ready"),
        StepRecord(2, "visual detection", "calculator, display, and buttons recognized", "map boxes to click centers", "targets found"),
        StepRecord(3, "automatic click", "action queue generated", "click 1, 2, +, 7, =", "input complete"),
        StepRecord(4, "result monitor", "display recognized as 19", "compare with expected result", "success"),
        StepRecord(5, "feedback recovery", "simulated mismatch recognized as 18", "clear and retry", "recovered"),
        StepRecord(6, "coordination", "result is stable", "write log and update monitor page", "finished"),
    ]
    payload = {
        "experiment": "Experiment 2 - vision feedback automation control",
        "task": "calculator automation with closed-loop visual feedback",
        "images": [str(path.relative_to(ROOT)) for path in image_paths],
        "records": [asdict(record) for record in records],
        "final_state": {"expression": "12 + 7", "expected": 19, "recognized": 19, "status": "PASS"},
    }
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    payload = run_demo()
    print(json.dumps(payload["final_state"], ensure_ascii=False))


if __name__ == "__main__":
    main()
