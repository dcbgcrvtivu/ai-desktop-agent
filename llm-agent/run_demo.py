from __future__ import annotations

import argparse
import html
import json
import textwrap
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from llm_auto_agent import OUTPUT_DIR, RUN_LOG, ROOT, ScenarioResult, run_many, scenario_contexts


DEMO_HTML = OUTPUT_DIR / "demo.html"
DEMO_GIF = OUTPUT_DIR / "automation_demo.gif"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = font(32, True)
FONT_H2 = font(22, True)
FONT_BODY = font(17)
FONT_SMALL = font(14)
FONT_MONO = font(15)


def rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str = "#cbd5e1", width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fnt: ImageFont.ImageFont, fill: str = "#111827") -> None:
    draw.text(xy, value, font=fnt, fill=fill)


def wrap_lines(value: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in value.splitlines() or [""]:
        if raw_line.isascii():
            lines.extend(textwrap.wrap(raw_line, width=width, break_long_words=False) or [""])
            continue
        current = ""
        current_units = 0
        for char in raw_line:
            units = 2 if ord(char) > 127 else 1
            if current and current_units + units > width:
                lines.append(current)
                current = char
                current_units = units
            else:
                current += char
                current_units += units
        lines.append(current)
    return lines


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, lines: list[str], fill: str = "#ffffff") -> None:
    rect(draw, box, fill)
    text(draw, (box[0] + 18, box[1] + 16), title, FONT_H2)
    y = box[1] + 58
    for line in lines:
        for wrapped in wrap_lines(line, 54):
            text(draw, (box[0] + 18, y), wrapped, FONT_BODY, "#334155")
            y += 26
        y += 6


def scenario_image(result: ScenarioResult) -> Path:
    contexts = scenario_contexts()
    context = contexts[result.scenario]
    img = Image.new("RGB", (1280, 760), "#eef2f7")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1280, 88), fill="#101828")
    title = {
        "calculator": "场景1：打开计算器并完成简单计算",
        "text_extract": "场景2：整理文本信息并提取要点",
        "form_fill": "场景3：将 OCR 结果自动填入表格",
        "batch_tasks": "场景4：批量执行简单任务序列",
    }[result.scenario]
    text(draw, (34, 20), title, FONT_TITLE, "#ffffff")
    text(draw, (34, 58), f"Planner: {result.planner}   Mode: {result.mode}   Status: {result.status}", FONT_SMALL, "#cbd5e1")

    draw_panel(draw, (34, 120, 594, 382), "屏幕文本 / OCR 输入", wrap_lines(context.screen_text, 58), "#ffffff")
    visible_actions = result.actions[:6]
    action_lines = [
        f"{idx:02d}. {action.action} -> {action.target}"
        + (f" = {action.value}" if action.value and len(action.value) < 40 else "")
        for idx, action in enumerate(visible_actions, start=1)
    ]
    if len(result.actions) > len(visible_actions):
        action_lines.append(f"... 共 {len(result.actions)} 个动作")
    draw_panel(draw, (630, 120, 1246, 382), "模型输出的动作指令", action_lines, "#ffffff")

    if result.scenario == "calculator":
        draw_calculator(draw, active_result="384")
    elif result.scenario == "text_extract":
        draw_document(draw)
    elif result.scenario == "form_fill":
        draw_form(draw)
    else:
        draw_batch(draw)

    log_lines = [f"{item['step']:02d}. {item['result']}" for item in result.execution_log[:6]]
    if len(result.execution_log) > 6:
        log_lines.append(f"... 共 {len(result.execution_log)} 条执行记录")
    draw_panel(draw, (630, 405, 1246, 720), "执行日志与反馈", log_lines + [result.observation_after], "#f8fafc")
    path = OUTPUT_DIR / f"{result.scenario}_demo.png"
    img.save(path)
    return path


def draw_calculator(draw: ImageDraw.ImageDraw, active_result: str) -> None:
    rect(draw, (74, 405, 550, 720), "#f8fafc")
    rect(draw, (112, 435, 512, 499), "#111827", "#111827")
    text(draw, (438, 449), active_result, FONT_TITLE, "#bbf7d0")
    labels = [["7", "8", "9", "+"], ["4", "5", "6", "-"], ["1", "2", "3", "="]]
    for r, row in enumerate(labels):
        for c, label in enumerate(row):
            x = 112 + c * 100
            y = 525 + r * 54
            fill = "#e0f2fe" if label in {"+", "-", "="} else "#ffffff"
            rect(draw, (x, y, x + 74, y + 40), fill)
            text(draw, (x + 28, y + 8), label, FONT_BODY)


def draw_document(draw: ImageDraw.ImageDraw) -> None:
    rect(draw, (74, 405, 550, 720), "#ffffff")
    lines = ["项目名称: 智能实验助手", "截止日期: 2026-07-12", "负责人: Demo User", "关键任务: 部署、演示、报告", "风险: API key 缺失时离线降级"]
    text(draw, (100, 435), "文档摘要区", FONT_H2)
    for idx, line in enumerate(lines):
        text(draw, (100, 481 + idx * 36), line, FONT_BODY, "#334155")


def draw_form(draw: ImageDraw.ImageDraw) -> None:
    rect(draw, (74, 405, 550, 720), "#ffffff")
    rows = [("学号", "0000000000"), ("姓名", "Demo User"), ("实验", "实验三"), ("模型", "offline fallback"), ("状态", "PASS")]
    text(draw, (100, 435), "结构化表格", FONT_H2)
    for idx, (key, value) in enumerate(rows):
        y = 481 + idx * 40
        rect(draw, (100, y, 220, y + 30), "#f1f5f9")
        rect(draw, (230, y, 510, y + 30), "#ffffff")
        text(draw, (112, y + 6), key, FONT_SMALL)
        text(draw, (244, y + 6), value, FONT_SMALL, "#334155")


def draw_batch(draw: ImageDraw.ImageDraw) -> None:
    rect(draw, (74, 405, 550, 720), "#ffffff")
    text(draw, (100, 435), "任务记录", FONT_H2)
    lines = ["计算任务: 128 + 256 = 384", "文档要点: 截止日期 2026-07-12", "表格状态: 已填入并提交 PASS"]
    for idx, line in enumerate(lines):
        rect(draw, (100, 485 + idx * 56, 516, 525 + idx * 56), "#f8fafc")
        text(draw, (116, 495 + idx * 56), line, FONT_SMALL, "#334155")


def build_html(image_paths: list[Path]) -> None:
    cards = []
    for path in image_paths:
        rel = path.relative_to(ROOT).as_posix()
        cards.append(f'<section><h2>{html.escape(path.stem)}</h2><img src="../{html.escape(rel)}" alt="{html.escape(path.stem)}"></section>')
    DEMO_HTML.write_text(
        """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>Experiment 3 LLM Automation Demo</title>
<style>
body{margin:0;background:#eef2f7;color:#111827;font-family:"Microsoft YaHei",Arial,sans-serif}
main{max-width:1120px;margin:0 auto;padding:28px}
h1{font-size:28px;margin:0 0 18px}
section{background:white;border:1px solid #d7dee8;border-radius:8px;margin:18px 0;padding:16px}
h2{font-size:18px;margin:0 0 12px}
img{width:100%;height:auto;border:1px solid #cbd5e1;border-radius:6px}
</style>
<main>
<h1>实验三：大模型自动化闭环演示</h1>
"""
        + "\n".join(cards)
        + "\n</main>\n</html>\n",
        encoding="utf-8",
    )


def build_gif(image_paths: list[Path]) -> None:
    frames = [Image.open(path).resize((960, 570)) for path in image_paths]
    frames[0].save(DEMO_GIF, save_all=True, append_images=frames[1:], duration=1400, loop=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 3 LLM automation demo.")
    parser.add_argument("--scenario", choices=["all", "calculator", "text_extract", "form_fill", "batch_tasks"], default="all")
    parser.add_argument("--mode", choices=["simulate", "real"], default="simulate")
    parser.add_argument("--confirm", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_names = list(scenario_contexts()) if args.scenario == "all" else [args.scenario]
    results = run_many(scenario_names, mode=args.mode, confirm=args.confirm)
    image_paths = [scenario_image(result) for result in results]
    build_html(image_paths)
    build_gif(image_paths)
    print(
        json.dumps(
            {
                "log": str(RUN_LOG),
                "html": str(DEMO_HTML),
                "gif": str(DEMO_GIF),
                "images": [str(path) for path in image_paths],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
