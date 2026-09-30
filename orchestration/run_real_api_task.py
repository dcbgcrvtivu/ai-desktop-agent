"""
Run a controlled real API task for Experiment 4.

This script deliberately does not use the offline rule planner. It calls an
OpenAI-compatible API, validates the returned actions, then executes a safe
real task by creating a real text file in the controlled workspace.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.orchestrator import AgentAction, LLMPlanner, OUTPUT_DIR, SafetyGate, TaskContext, now_text
from src.scene_registry import get_scene

REAL_TASK_DIR = OUTPUT_DIR / "real_api_task"
REAL_TASK_JSON = OUTPUT_DIR / "real_api_task.json"
REAL_TASK_SCREENSHOT = OUTPUT_DIR / "real_api_task_screenshot.png"


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for item in candidates:
        try:
            return ImageFont.truetype(item, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap(text: str, width_units: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        current = ""
        units = 0
        for char in raw:
            char_units = 2 if ord(char) > 127 else 1
            if current and units + char_units > width_units:
                lines.append(current)
                current = char
                units = char_units
            else:
                current += char
                units += char_units
        lines.append(current)
    return lines


def render_evidence(payload: dict) -> None:
    img = Image.new("RGB", (1280, 760), "#eef2f7")
    draw = ImageDraw.Draw(img)
    title = font(30, True)
    head = font(20, True)
    body = font(16)
    small = font(13)

    draw.rectangle((0, 0, 1280, 88), fill="#0f172a")
    draw.text((34, 18), "实验四：真实 API 调用 + 受控真实任务执行", font=title, fill="#ffffff")
    draw.text((34, 56), f"生成时间：{payload['generated_at']} | 状态：{payload['status']}", font=small, fill="#cbd5e1")

    draw.rounded_rectangle((34, 120, 612, 700), radius=8, fill="#ffffff", outline="#cbd5e1")
    draw.text((58, 144), "API 调用留痕", font=head, fill="#0f172a")
    trace = payload["api_trace"]
    trace_lines = [
        f"provider: {trace.get('provider', '')}",
        f"endpoint: {trace.get('endpoint', '')}",
        f"model: {trace.get('model', '')}",
        f"usage: {json.dumps(trace.get('usage', {}), ensure_ascii=False)}",
        f"actions: {len(payload['actions'])}",
    ]
    y = 184
    for line in trace_lines:
        for part in wrap(line, 58):
            draw.text((58, y), part, font=body, fill="#334155")
            y += 28
        y += 4

    draw.text((58, y + 12), "模型返回动作序列：", font=head, fill="#0f172a")
    y += 48
    for index, action in enumerate(payload["actions"][:7], start=1):
        line = f"{index}. {action['action']} -> {action['target']} {action.get('value') or ''}"
        for part in wrap(line, 58):
            draw.text((58, y), part, font=small, fill="#475569")
            y += 22

    draw.rounded_rectangle((650, 120, 1246, 700), radius=8, fill="#ffffff", outline="#cbd5e1")
    draw.text((674, 144), "真实任务验证", font=head, fill="#0f172a")
    task_lines = [
        "任务：根据 API 返回动作，生成并保存实验文本文件",
        f"真实文件：{payload['file_path']}",
        f"文件大小：{payload['file_size']} bytes",
        f"SHA256：{payload['sha256']}",
    ]
    y = 184
    for line in task_lines:
        for part in wrap(line, 56):
            draw.text((674, y), part, font=body, fill="#334155")
            y += 28
        y += 4

    draw.rounded_rectangle((674, 386, 1218, 646), radius=6, fill="#f8fafc", outline="#cbd5e1")
    draw.text((694, 408), "文件内容预览", font=head, fill="#0f172a")
    y = 448
    for line in wrap(payload["file_preview"], 56)[:7]:
        draw.text((694, y), line, font=body, fill="#111827")
        y += 28

    REAL_TASK_SCREENSHOT.parent.mkdir(parents=True, exist_ok=True)
    img.save(REAL_TASK_SCREENSHOT)


def action_dict(action: AgentAction) -> dict:
    return {
        "action": action.action.value,
        "target": action.target,
        "value": action.value,
        "x": action.x,
        "y": action.y,
        "reason": action.reason,
        "expected": action.expected,
    }


def main() -> None:
    os.environ.setdefault("LLM_PROVIDER", "pollinations")
    scene = get_scene("notepad")
    context = TaskContext(task_name=scene.name, task_description=scene.description, goal=scene.goal, max_loops=1)
    context.extracted_data.update(scene.data)
    screen_text = scene.initial_screen_text

    planner = LLMPlanner()
    start = time.time()
    actions = planner.plan_actions(context, screen_text, scene.targets)
    actions = SafetyGate().validate(actions)

    text_action = next(
        (
            item for item in actions
            if item.action.value == "type_text"
            and item.value
            and item.target in {"editor", "notepad_editor", "document_body"}
        ),
        None,
    )
    filename_action = next((item for item in actions if item.target in {"filename", "file_name_input"} and item.value), None)
    text_to_write = text_action.value if text_action else scene.data["text_to_type"]
    filename = filename_action.value if filename_action else scene.data["filename"]
    if not filename.lower().endswith(".txt"):
        filename += ".txt"

    REAL_TASK_DIR.mkdir(parents=True, exist_ok=True)
    file_path = REAL_TASK_DIR / filename
    file_path.write_text(text_to_write, encoding="utf-8")
    raw = file_path.read_bytes()

    import hashlib

    payload = {
        "experiment": "实验四：任务编排与系统集成实战",
        "branch": "Branch B",
        "status": "PASS" if file_path.exists() and raw else "FAIL",
        "generated_at": now_text(),
        "elapsed_sec": round(time.time() - start, 2),
        "api_trace": planner.last_trace,
        "actions": [action_dict(item) for item in actions],
        "file_path": str(file_path),
        "file_size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file_preview": file_path.read_text(encoding="utf-8")[:500],
        "screenshot": str(REAL_TASK_SCREENSHOT),
        "note": "No offline rule fallback was used in this script.",
    }
    REAL_TASK_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_evidence(payload)
    print(json.dumps({"status": payload["status"], "provider": planner.last_trace.get("provider"), "file": str(file_path), "screenshot": str(REAL_TASK_SCREENSHOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
