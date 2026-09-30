"""
实验四：任务编排与系统集成实战
核心编排引擎 - 实现"感知→决策→执行→验证"闭环

集成实验1（屏幕文字识别/OCR）、实验2（鼠标键盘模拟/YOLO）、实验3（LLM API调用）
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Literal


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
SCREENSHOTS_DIR = ROOT / "screenshots"
RUN_LOG = OUTPUT_DIR / "experiment_run.json"


class ActionType(str, Enum):
    SCREENSHOT = "screenshot"
    OCR_EXTRACT = "ocr_extract"
    YOLO_DETECT = "yolo_detect"
    CLICK = "click"
    TYPE_TEXT = "type_text"
    HOTKEY = "hotkey"
    PRESS = "press"
    OPEN_APP = "open_app"
    WAIT = "wait"
    VERIFY = "verify"
    LLM_DECIDE = "llm_decide"
    SAVE = "save"


@dataclass
class AgentAction:
    action: ActionType
    target: str
    value: str | None = None
    x: int | None = None
    y: int | None = None
    reason: str = ""
    expected: str | None = None


@dataclass
class ExecutionStep:
    phase: str
    step_index: int
    action: AgentAction
    result: str
    success: bool
    timestamp: str
    screenshot_path: str | None = None
    error: str | None = None
    retry_count: int = 0


@dataclass
class TaskContext:
    task_name: str
    task_description: str
    goal: str
    current_phase: str = "init"
    loop_count: int = 0
    max_loops: int = 10
    screen_text_history: list[str] = field(default_factory=list)
    action_history: list[ExecutionStep] = field(default_factory=list)
    extracted_data: dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    max_errors: int = 5
    start_time: str = ""
    end_time: str = ""

    def add_step(self, step: ExecutionStep) -> None:
        self.action_history.append(step)
        if not step.success:
            self.error_count += 1

    def is_blocked(self) -> bool:
        return self.error_count >= self.max_errors or self.loop_count >= self.max_loops


@dataclass
class TaskResult:
    task_name: str
    goal: str
    status: str
    total_steps: int
    successful_steps: int
    failed_steps: int
    retry_count: int
    total_time_sec: float
    phases_completed: list[str]
    final_observation: str
    screen_text_after: str
    execution_log: list[dict[str, Any]]
    planner: str
    started_at: str
    finished_at: str
    planner_trace: dict[str, Any] = field(default_factory=dict)


class PlanningError(RuntimeError):
    pass


class SafetyError(RuntimeError):
    pass


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_file() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class SafetyGate:
    allowed_actions: set[str] = {
        "screenshot", "ocr_extract", "yolo_detect",
        "click", "type_text", "hotkey", "press", "open_app",
        "wait", "verify", "llm_decide", "save",
    }
    blocked_terms: set[str] = {
        "delete", "format", "shutdown", "rm -rf",
        "Remove-Item", "del /f", "支付", "转账",
        "reg delete", "taskkill /f",
    }

    def __init__(self, screen_size: tuple[int, int] = (1920, 1080), max_actions: int = 100) -> None:
        self.screen_size = screen_size
        self.max_actions = max_actions

    def validate(self, actions: list[AgentAction]) -> list[AgentAction]:
        checked = list(actions)
        if len(checked) > self.max_actions:
            raise SafetyError(f"too many actions: {len(checked)} > {self.max_actions}")
        for index, action in enumerate(checked, start=1):
            if action.action.value not in self.allowed_actions:
                raise SafetyError(f"action {index} not allowed: {action.action.value}")
            text = f"{action.target} {action.value or ''} {action.reason}".lower()
            if any(term.lower() in text for term in self.blocked_terms):
                raise SafetyError(f"action {index} contains a blocked term")
            if action.action == ActionType.CLICK and action.x is not None and action.y is not None:
                w, h = self.screen_size
                if not (0 <= action.x <= w and 0 <= action.y <= h):
                    raise SafetyError(f"action {index} click ({action.x},{action.y}) outside {w}x{h}")
        return checked


class LLMPlanner:
    name = "llm-planner"

    def __init__(self) -> None:
        self.base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "qwen2.5-7b-instruct")
        self.pollinations_mode = os.getenv("LLM_PROVIDER", "pollinations").lower()
        self.last_trace: dict[str, Any] = {}

    def decide(self, system_prompt: str, user_prompt: str) -> str:
        if self.pollinations_mode in ("pollinations", "public"):
            return self._call_pollinations(system_prompt, user_prompt)
        return self._call_openai_compatible(system_prompt, user_prompt)

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        if not self.base_url or not self.api_key:
            raise PlanningError("OPENAI_BASE_URL and OPENAI_API_KEY required")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PlanningError(f"API request failed: {exc}") from exc
        content = data["choices"][0]["message"]["content"]
        self.last_trace = {"provider": "openai-compatible", "endpoint": f"{self.base_url}/chat/completions",
                           "model": self.model, "usage": data.get("usage", {})}
        return content

    def _call_pollinations(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = "https://text.pollinations.ai/openai"
        payload = {
            "model": "openai-fast",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Experiment4-Orchestration/1.0"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            return self._call_pollinations_text(system_prompt, user_prompt, exc)
        content = data["choices"][0]["message"]["content"]
        self.last_trace = {"provider": "pollinations.ai", "endpoint": endpoint,
                           "model": "openai-fast", "usage": data.get("usage", {})}
        return content

    def _call_pollinations_text(self, system_prompt: str, user_prompt: str, first_error: Exception) -> str:
        prompt = (
            system_prompt
            + "\n\n"
            + user_prompt
            + "\n\n只输出JSON数组，不要输出任何解释。"
        )
        endpoint = "https://text.pollinations.ai/" + urllib.parse.quote(prompt, safe="")
        query = urllib.parse.urlencode({"model": "openai-fast", "json": "true"})
        request = urllib.request.Request(
            endpoint + "?" + query,
            headers={"User-Agent": "Experiment4-Orchestration/1.0"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PlanningError(f"Pollinations failed: {first_error}; fallback failed: {exc}") from exc
        self.last_trace = {"provider": "pollinations.ai", "endpoint": "https://text.pollinations.ai/{prompt}",
                           "model": "openai-fast", "usage": {}, "fallback": "text-endpoint"}
        return content

    def plan_actions(self, context: TaskContext, screen_text: str, available_targets: dict) -> list[AgentAction]:
        system_prompt = """你是桌面自动化编排代理。根据当前屏幕状态、用户目标和历史上下文，生成下一步可执行的动作序列。只输出JSON数组，不要输出解释文本。

动作类型：screenshot截图, ocr_extract文字提取, click点击, type_text输入, hotkey快捷键, press按键, open_app打开应用, wait等待, verify验证, save保存

JSON Schema: {"action":"string","target":"string","value":"string|null","x":"int|null","y":"int|null","reason":"中文理由","expected":"预期值|null"}"""

        prompt_parts = [
            f"【用户目标】{context.goal}",
            f"【任务描述】{context.task_description}",
            f"【当前阶段】{context.current_phase}",
            f"【已执行循环】第{context.loop_count + 1}次",
            f"【当前屏幕文字】{screen_text}",
            f"【可操作的目标元素】{json.dumps(available_targets, ensure_ascii=False)}",
        ]
        if context.extracted_data:
            prompt_parts.append(f"【已提取数据】{json.dumps(context.extracted_data, ensure_ascii=False, indent=2)}")
        if context.action_history:
            last_steps = context.action_history[-5:]
            history_lines = [f"  [{'OK' if s.success else 'FAIL'}] {s.phase}: {s.action.action.value} {s.action.target} -> {s.result}" for s in last_steps]
            prompt_parts.append("【最近操作历史】\n" + "\n".join(history_lines))

        user_prompt = chr(10).join(prompt_parts)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                content = self.decide(system_prompt, user_prompt)
                return self._parse_actions(content, available_targets)
            except (json.JSONDecodeError, PlanningError) as exc:
                last_error = exc
                time.sleep(1.0)
        raise PlanningError(f"LLM planning failed after 3 attempts: {last_error}")

    def _parse_actions(self, content: str, available: dict) -> list[AgentAction]:
        stripped = content.strip()
        if stripped.startswith("["):
            text = stripped
        else:
            m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stripped, re.S)
            if m:
                text = m.group(1)
            else:
                raise PlanningError(f"no JSON array in: {stripped[:300]}")
        parsed = json.loads(text)
        actions = []
        for item in parsed:
            raw_target = item.get("target", "")
            target = "" if raw_target is None else str(raw_target)
            x, y = item.get("x"), item.get("y")
            if target in available and (x is None or y is None):
                coords = available[target]
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    x, y = coords[0], coords[1]
            actions.append(AgentAction(
                action=ActionType(item.get("action", "wait")), target=target,
                value=str(item.get("value")) if item.get("value") is not None else None,
                x=int(x) if x is not None else None, y=int(y) if y is not None else None,
                reason=str(item.get("reason", "")),
                expected=str(item.get("expected")) if item.get("expected") else None,
            ))
        return actions



class RuleBasedPlanner:
    """规则规划器 - 离线降级方案，不依赖API"""
    name = "rule-planner"

    def __init__(self):
        self.last_trace = {"note": "offline fallback, no API call"}

    def plan_actions(self, context, screen_text, available_targets):
        actions = []
        if context.task_name == "notepad":
            actions.append(AgentAction(ActionType.OPEN_APP, "notepad", reason="打开记事本"))
            actions.append(AgentAction(ActionType.WAIT, "app_ready", "1.0", reason="等待应用启动"))
            actions.append(AgentAction(ActionType.TYPE_TEXT, "editor", context.extracted_data.get("text_to_type", "实验报告"), reason="输入实验报告文字"))
            actions.append(AgentAction(ActionType.HOTKEY, "save", "ctrl+s", reason="保存文件"))
            actions.append(AgentAction(ActionType.WAIT, "save_dialog", "0.5", reason="等待对话框"))
            actions.append(AgentAction(ActionType.TYPE_TEXT, "filename", context.extracted_data.get("filename", "实验四报告.txt"), reason="输入文件名"))
            actions.append(AgentAction(ActionType.PRESS, "confirm", "enter", reason="确认保存"))
        elif context.task_name == "minesweeper":
            actions.append(AgentAction(ActionType.CLICK, "center_cell", x=400, y=400, reason="点击中央单元格开局", expected="1"))
            actions.append(AgentAction(ActionType.WAIT, "game_update", "0.5", reason="等待游戏更新"))
            actions.append(AgentAction(ActionType.CLICK, "cell_top_left", x=250, y=200, reason="点击左上单元格", expected="1"))
            actions.append(AgentAction(ActionType.SCREENSHOT, "game_state", reason="截图保存游戏状态"))
        else:
            actions.append(AgentAction(ActionType.SCREENSHOT, "fallback", reason="识别当前屏幕"))
        return actions

class ClosedLoopEngine:
    def __init__(self, mode: Literal["simulate", "real"] = "simulate", confirm: str = "",
                 screenshot_func: Callable | None = None, ocr_func: Callable | None = None,
                 execute_func: Callable | None = None):
        self.mode = mode
        self.confirm = confirm
        self.safety = SafetyGate()
        self._llm_planner = LLMPlanner()
        self._rule_planner = RuleBasedPlanner()
        self.planner = self._llm_planner
        self._screenshot_func = screenshot_func
        self._ocr_func = ocr_func
        self._execute_func = execute_func
        self._last_simulated_ocr = ""

    def run(self, context: TaskContext, initial_screen_text: str, targets: dict) -> TaskResult:
        context.start_time = now_text()
        screen_text = initial_screen_text
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

        while context.loop_count < context.max_loops and not context.is_blocked():
            context.loop_count += 1

            context.current_phase = "perceive"
            perceive_ok, screen_text = self._do_perceive(context)
            if not perceive_ok and context.loop_count == 1:
                return self._build_result(context, "FAIL", "感知阶段失败")

            context.current_phase = "decide"
            decide_ok, actions = self._do_decide(context, screen_text, targets)
            if not decide_ok:
                context.current_phase = "retry"
                continue

            context.current_phase = "execute"
            execute_ok = self._do_execute(context, actions)
            if not execute_ok:
                continue

            context.current_phase = "verify"
            verify_ok = self._do_verify(context, actions)
            if verify_ok:
                context.end_time = now_text()
                return self._build_result(context, "PASS", "任务完成", screen_text)
            context.current_phase = "retry"

        context.end_time = now_text()
        reason = "达到最大循环次数" if context.loop_count >= context.max_loops else f"错误数: {context.error_count}"
        return self._build_result(context, "PARTIAL", reason, screen_text)

    def _do_perceive(self, context: TaskContext) -> tuple[bool, str]:
        try:
            ss_path = self._take_screenshot(context)
            extracted = self._extract_text(ss_path)
            context.screen_text_history.append(extracted)
            context.extracted_data["last_screen_text"] = extracted
            context.add_step(ExecutionStep(phase="perceive", step_index=context.loop_count,
                action=AgentAction(ActionType.OCR_EXTRACT, "screen"),
                result=extracted[:200], success=True, timestamp=now_text(), screenshot_path=str(ss_path)))
            return True, extracted
        except Exception as exc:
            context.add_step(ExecutionStep(phase="perceive", step_index=context.loop_count,
                action=AgentAction(ActionType.SCREENSHOT, "screen"),
                result="", success=False, timestamp=now_text(), error=str(exc)))
            return False, ""

    def _do_decide(self, context: TaskContext, screen_text: str, targets: dict) -> tuple[bool, list[AgentAction]]:
        prefer_rule = os.getenv("LLM_PROVIDER", "").lower() in {"rule", "offline", "simulate"}
        planner = self._rule_planner if prefer_rule else self._llm_planner
        planner_name = "rule" if prefer_rule else "llm"
        # Try the configured planner first, then fall back to rules if an API call fails.
        try:
            actions = planner.plan_actions(context, screen_text, targets)
        except Exception:
            planner = self._rule_planner
            planner_name = "rule"
            try:
                actions = planner.plan_actions(context, screen_text, targets)
            except Exception as exc2:
                context.add_step(ExecutionStep(phase="decide", step_index=context.loop_count,
                    action=AgentAction(ActionType.LLM_DECIDE, "llm_plan"),
                    result="", success=False, timestamp=now_text(), error=str(exc2)))
                return False, []
        self.planner = planner
        try:
            actions = self.safety.validate(actions)
        except SafetyError as exc:
            context.add_step(ExecutionStep(phase="decide", step_index=context.loop_count,
                action=AgentAction(ActionType.LLM_DECIDE, "llm_plan"),
                result="", success=False, timestamp=now_text(), error=str(exc)))
            return False, []
        context.add_step(ExecutionStep(phase="decide", step_index=context.loop_count,
            action=AgentAction(ActionType.LLM_DECIDE, "llm_plan",
                value=json.dumps([a.action.value for a in actions], ensure_ascii=False)),
            result=f"生成了{len(actions)}个动作 ({planner_name})", success=True, timestamp=now_text()))
        return True, actions

    def _do_execute(self, context: TaskContext, actions: list[AgentAction]) -> bool:
        all_ok = True
        for i, action in enumerate(actions):
            try:
                result = self._execute_action(action)
                ok = not result.startswith("ERROR:")
                context.add_step(ExecutionStep(phase="execute", step_index=i + 1,
                    action=action, result=result, success=ok, timestamp=now_text()))
                if not ok:
                    all_ok = False
                    break
            except Exception as exc:
                context.add_step(ExecutionStep(phase="execute", step_index=i + 1,
                    action=action, result="", success=False, timestamp=now_text(), error=str(exc)))
                all_ok = False
                break
        return all_ok

    def _do_verify(self, context: TaskContext, actions: list[AgentAction]) -> bool:
        try:
            ss_path = self._take_screenshot(context)
            screen_text = self._extract_text(ss_path)
            context.extracted_data["verification_text"] = screen_text
            expected_items = [a.expected for a in actions if a.expected]
            matched = any(ex in screen_text for ex in expected_items) if expected_items else True
            result_text = "验证通过" if matched else f"未找到: {expected_items}"
            context.add_step(ExecutionStep(phase="verify", step_index=context.loop_count,
                action=AgentAction(ActionType.VERIFY, "verification",
                    value=", ".join(expected_items) if expected_items else None),
                result=result_text, success=matched, timestamp=now_text(), screenshot_path=str(ss_path)))
            return matched
        except Exception as exc:
            context.add_step(ExecutionStep(phase="verify", step_index=context.loop_count,
                action=AgentAction(ActionType.VERIFY, "verification"),
                result="", success=False, timestamp=now_text(), error=str(exc)))
            return False

    def _take_screenshot(self, context: TaskContext) -> Path:
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = timestamp_file()
        path = SCREENSHOTS_DIR / f"loop{context.loop_count:02d}_{context.current_phase}_{ts}.png"
        if self._screenshot_func:
            self._screenshot_func(path)
        elif self.mode == "simulate":
            self._render_simulated_screenshot(context, path)
        else:
            try:
                from PIL import ImageGrab
                ImageGrab.grab().save(path)
            except ImportError:
                self._render_simulated_screenshot(context, path)
        return path

    def _extract_text(self, image_path: Path) -> str:
        if self._ocr_func:
            return self._ocr_func(image_path)
        if self.mode == "simulate" and self._last_simulated_ocr:
            return self._last_simulated_ocr
        return "(模拟OCR结果)"

    def _font(self, size: int, bold: bool = False):
        from PIL import ImageFont
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

    def _render_simulated_screenshot(self, context: TaskContext, path: Path) -> None:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (1280, 800), "#dbeafe")
        draw = ImageDraw.Draw(img)
        title_font = self._font(28, True)
        head_font = self._font(20, True)
        body_font = self._font(16)
        small_font = self._font(13)

        draw.rectangle((0, 0, 1280, 44), fill="#0f172a")
        draw.text((18, 9), "实验四自动化闭环演示 - 模拟受控桌面", font=head_font, fill="#ffffff")
        draw.text((930, 12), f"{context.task_name} | {context.current_phase} | loop {context.loop_count}", font=small_font, fill="#cbd5e1")
        draw.rectangle((0, 744, 1280, 800), fill="#111827")
        draw.rounded_rectangle((16, 754, 132, 790), radius=6, fill="#1f2937", outline="#475569")
        draw.text((36, 762), "Start", font=body_font, fill="#e5e7eb")
        draw.text((1070, 762), now_text(), font=small_font, fill="#cbd5e1")

        if context.task_name == "notepad":
            self._draw_notepad_screen(draw, context, title_font, head_font, body_font, small_font)
        elif context.task_name == "minesweeper":
            self._draw_minesweeper_screen(draw, context, title_font, head_font, body_font, small_font)
        else:
            draw.text((80, 120), context.goal, font=title_font, fill="#0f172a")
            self._last_simulated_ocr = context.goal
        img.save(path)

    def _wrap_display_text(self, text: str, width_units: int) -> list[str]:
        lines: list[str] = []
        for raw_line in text.splitlines() or [""]:
            current = ""
            units = 0
            for char in raw_line:
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

    def _draw_notepad_screen(self, draw, context: TaskContext, title_font, head_font, body_font, small_font) -> None:
        text_to_type = context.extracted_data.get("text_to_type", "实验四：任务编排与系统集成实战")
        filename = context.extracted_data.get("filename", "实验四报告.txt")
        done = context.current_phase == "verify" or any(
            step.action.action == ActionType.PRESS and step.action.target == "confirm" and step.success
            for step in context.action_history
        )

        draw.text((80, 92), "桌面 OCR 感知结果", font=title_font, fill="#0f172a")
        for i, label in enumerate(["Recycle Bin", "This PC", "User Folder"]):
            x = 82 + i * 112
            draw.rounded_rectangle((x, 145, x + 76, 220), radius=10, fill="#eff6ff", outline="#93c5fd")
            draw.rectangle((x + 22, 160, x + 54, 192), fill="#60a5fa")
            draw.text((x, 228), label, font=small_font, fill="#1e293b")

        draw.rounded_rectangle((430, 104, 1160, 690), radius=8, fill="#ffffff", outline="#94a3b8", width=2)
        draw.rectangle((431, 105, 1159, 143), fill="#f1f5f9")
        title = f"{filename} - 记事本" if done else "无标题 - 记事本"
        draw.text((448, 113), title, font=body_font, fill="#111827")
        draw.text((1084, 113), "_  □  X", font=small_font, fill="#475569")
        draw.line((430, 144, 1160, 144), fill="#cbd5e1", width=1)
        draw.text((450, 154), "文件  编辑  格式  查看  帮助", font=small_font, fill="#334155")

        if done:
            y = 198
            for line in self._wrap_display_text(text_to_type, 62):
                draw.text((462, y), line, font=body_font, fill="#111827")
                y += 30
            draw.rounded_rectangle((720, 592, 1130, 654), radius=6, fill="#ecfdf5", outline="#22c55e")
            draw.text((740, 604), "验证：文件已保存到桌面", font=body_font, fill="#166534")
            draw.text((740, 630), filename, font=small_font, fill="#166534")
            self._last_simulated_ocr = f"Notepad window. Saved file {filename}. {text_to_type}"
        else:
            draw.text((462, 202), "当前未打开记事本或编辑区为空。", font=body_font, fill="#64748b")
            draw.rounded_rectangle((78, 296, 366, 368), radius=8, fill="#ffffff", outline="#94a3b8")
            draw.text((98, 314), "OCR: No Notepad text yet", font=body_font, fill="#334155")
            self._last_simulated_ocr = "Windows desktop. No Notepad window open. Start button visible."

    def _draw_minesweeper_screen(self, draw, context: TaskContext, title_font, head_font, body_font, small_font) -> None:
        updated = context.current_phase == "verify" or any(
            step.action.action == ActionType.CLICK and step.action.target == "center_cell" and step.success
            for step in context.action_history
        )
        draw.text((80, 92), "扫雷游戏 OCR/网格识别", font=title_font, fill="#0f172a")
        draw.rounded_rectangle((360, 92, 920, 700), radius=8, fill="#e5e7eb", outline="#64748b", width=2)
        draw.rectangle((382, 118, 898, 176), fill="#cbd5e1", outline="#94a3b8")
        draw.rounded_rectangle((394, 130, 492, 166), radius=4, fill="#111827")
        draw.text((418, 137), "010", font=head_font, fill="#ef4444")
        draw.ellipse((612, 126, 662, 170), fill="#fde68a", outline="#92400e")
        draw.text((625, 136), ":)", font=body_font, fill="#78350f")
        draw.rounded_rectangle((770, 130, 868, 166), radius=4, fill="#111827")
        draw.text((794, 137), "001", font=head_font, fill="#ef4444")

        size = 42
        start_x, start_y = 451, 210
        opened = {(4, 4): "1", (0, 0): "1"} if updated else {}
        for row in range(9):
            for col in range(9):
                x1 = start_x + col * size
                y1 = start_y + row * size
                if (row, col) in opened:
                    draw.rectangle((x1, y1, x1 + size - 3, y1 + size - 3), fill="#f8fafc", outline="#94a3b8")
                    draw.text((x1 + 15, y1 + 8), opened[(row, col)], font=head_font, fill="#2563eb")
                else:
                    draw.rectangle((x1, y1, x1 + size - 3, y1 + size - 3), fill="#94a3b8", outline="#475569")
                    draw.line((x1 + 4, y1 + 4, x1 + size - 7, y1 + 4), fill="#e2e8f0", width=2)

        if updated:
            draw.rounded_rectangle((76, 594, 332, 658), radius=6, fill="#ecfdf5", outline="#22c55e")
            draw.text((96, 613), "验证：中心格与左上格已打开，OCR包含数字1", font=small_font, fill="#166534")
            self._last_simulated_ocr = "Minesweeper 9x9 grid. Center cell opened with 1. Top-left cell opened with 1."
        else:
            draw.rounded_rectangle((76, 594, 332, 658), radius=6, fill="#ffffff", outline="#94a3b8")
            draw.text((96, 613), "OCR: 9x9 grid, mine counter 10, cells unopened", font=small_font, fill="#334155")
            self._last_simulated_ocr = "Minesweeper 9x9 grid. Mine counter 10. All cells unopened."

    def _execute_action(self, action: AgentAction) -> str:
        time.sleep(0.2)
        if self._execute_func:
            return self._execute_func(None, action)
        if self.mode == "simulate":
            return self._simulate(action)
        return self._real(action)

    def _simulate(self, action: AgentAction) -> str:
        dispatch = {
            ActionType.CLICK: f"模拟点击 {action.target} ({action.x}, {action.y})",
            ActionType.TYPE_TEXT: f"模拟输入到 {action.target}: {action.value}",
            ActionType.HOTKEY: f"模拟快捷键: {action.value}",
            ActionType.PRESS: f"模拟按键: {action.value}",
            ActionType.WAIT: f"等待 {action.value} 秒",
            ActionType.OPEN_APP: f"打开应用: {action.target}",
            ActionType.SAVE: f"保存文件: {action.target}",
            ActionType.SCREENSHOT: f"截图: {action.target}",
        }
        return dispatch.get(action.action, f"执行: {action.action.value} {action.target}")

    def _real(self, action: AgentAction) -> str:
        if self.confirm != "REAL_RUN":
            return "ERROR: real mode requires --confirm REAL_RUN"
        import pyautogui
        pyautogui.PAUSE = 0.15
        if action.action == ActionType.CLICK and action.x is not None and action.y is not None:
            pyautogui.click(action.x, action.y)
            return f"点击 {action.target} ({action.x},{action.y})"
        elif action.action == ActionType.TYPE_TEXT and action.value:
            pyautogui.write(action.value, interval=0.02)
            return f"输入到 {action.target}"
        elif action.action == ActionType.HOTKEY and action.value:
            pyautogui.hotkey(*action.value.split("+"))
            return f"快捷键: {action.value}"
        elif action.action == ActionType.PRESS and action.value:
            pyautogui.press(action.value)
            return f"按键: {action.value}"
        elif action.action == ActionType.WAIT and action.value:
            time.sleep(float(action.value))
            return f"等待 {action.value} 秒"
        elif action.action == ActionType.OPEN_APP:
            import subprocess
            subprocess.Popen(["start", action.target], shell=True)
            time.sleep(1.0)
            return f"打开: {action.target}"
        return f"执行: {action.action.value} {action.target}"

    def _build_result(self, context: TaskContext, status: str, note: str, final_text: str = "") -> TaskResult:
        total = len(context.action_history)
        ok = sum(1 for s in context.action_history if s.success)
        elapsed = 0.0
        if context.start_time and context.end_time:
            fmt = "%Y-%m-%d %H:%M:%S"
            elapsed = (datetime.strptime(context.end_time, fmt) - datetime.strptime(context.start_time, fmt)).total_seconds()
        phases = list(dict.fromkeys(s.phase for s in context.action_history))
        return TaskResult(
            task_name=context.task_name, goal=context.goal, status=status,
            total_steps=total, successful_steps=ok, failed_steps=total - ok,
            retry_count=0, total_time_sec=elapsed, phases_completed=phases,
            final_observation=note, screen_text_after=final_text,
            execution_log=[{"step": i+1, "phase": s.phase, "action": s.action.action.value,
                "target": s.action.target, "result": s.result, "success": s.success,
                "timestamp": s.timestamp, "error": s.error} for i, s in enumerate(context.action_history)],
            planner=self.planner.name, planner_trace=getattr(self.planner, "last_trace", {}),
            started_at=context.start_time, finished_at=context.end_time)


def save_result(result: TaskResult) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": "实验四：任务编排与系统集成实战", "branch": "Branch B",
               "result": asdict(result), "generated_at": now_text()}
    RUN_LOG.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return RUN_LOG


def run_task(task_name: str, goal: str, description: str, initial_screen_text: str,
             targets: dict, mode: Literal["simulate", "real"] = "simulate", confirm: str = "",
             screenshot_func=None, ocr_func=None, execute_func=None, max_loops: int = 5,
             scene_data: dict | None = None) -> TaskResult:
    context = TaskContext(task_name=task_name, task_description=description, goal=goal, max_loops=max_loops)
    if scene_data:
        context.extracted_data.update(scene_data)
    engine = ClosedLoopEngine(mode=mode, confirm=confirm, screenshot_func=screenshot_func,
                              ocr_func=ocr_func, execute_func=execute_func)
    result = engine.run(context, initial_screen_text, targets)
    save_result(result)
    return result
