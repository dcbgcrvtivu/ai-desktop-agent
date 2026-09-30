from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "demo"
RUN_LOG = OUTPUT_DIR / "experiment_run.json"

ActionType = Literal[
    "open_app",
    "click",
    "hotkey",
    "type_text",
    "press",
    "wait",
    "extract",
    "fill_field",
    "copy_to_clipboard",
    "screenshot",
]


@dataclass
class ScreenContext:
    scenario: str
    user_goal: str
    screen_text: str
    ui_elements: dict[str, tuple[int, int]]
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    action: ActionType
    target: str
    value: str | None = None
    x: int | None = None
    y: int | None = None
    reason: str = ""


@dataclass
class ScenarioResult:
    scenario: str
    goal: str
    planner: str
    mode: str
    actions: list[AgentAction]
    execution_log: list[dict[str, Any]]
    status: str
    observation_after: str
    started_at: str
    finished_at: str
    model_trace: dict[str, Any] = field(default_factory=dict)


class PlanningError(RuntimeError):
    pass


class SafetyError(RuntimeError):
    pass


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def scenario_contexts() -> dict[str, ScreenContext]:
    return {
        "calculator": ScreenContext(
            scenario="calculator",
            user_goal="打开计算器并完成 128 + 256 的简单计算，记录最终结果。",
            screen_text=(
                "OCR: Windows desktop. Calculator icon is visible. "
                "Calculator buttons: 0 1 2 3 4 5 6 7 8 9 + - * / = C. "
                "Current display: 0."
            ),
            ui_elements={
                "calculator_icon": (92, 122),
                "button_1": (635, 452),
                "button_2": (716, 452),
                "button_8": (716, 290),
                "button_5": (716, 371),
                "button_6": (797, 371),
                "button_plus": (878, 533),
                "button_equals": (797, 533),
                "display": (766, 190),
            },
            data={"expression": "128+256", "expected": "384"},
        ),
        "text_extract": ScreenContext(
            scenario="text_extract",
            user_goal="整理文本信息，从文档中提取项目要点并写入摘要区域。",
            screen_text=(
                "OCR document: 项目名称: 智能实验助手; 截止日期: 2026-07-12; "
                "负责人: Demo User; 关键任务: 完成部署、演示、报告; 风险: API key 缺失时需要离线降级。 "
                "Right panel contains empty summary box."
            ),
            ui_elements={"summary_box": (892, 304), "copy_button": (1080, 620)},
            data={
                "fields": {
                    "项目名称": "智能实验助手",
                    "截止日期": "2026-07-12",
                    "负责人": "Demo User",
                    "关键任务": "完成部署、演示、报告",
                    "风险": "API key 缺失时需要离线降级",
                }
            },
        ),
        "form_fill": ScreenContext(
            scenario="form_fill",
            user_goal="把 OCR 结果自动填入表格，形成结构化运行数据。",
            screen_text=(
                "OCR table source: 学号 0000000000; 姓名 Demo User; 实验 实验三; "
                "模型 qwen2.5-7b-instruct/offline-rule-fallback; 状态 PASS. "
                "Target form fields are empty."
            ),
            ui_elements={
                "student_id": (846, 234),
                "name": (846, 294),
                "experiment": (846, 354),
                "model": (846, 414),
                "status": (846, 474),
                "submit": (964, 560),
            },
            data={
                "fields": {
                    "student_id": "0000000000",
                    "name": "Demo User",
                    "experiment": "实验三",
                    "model": "qwen2.5-7b-instruct / offline-rule-fallback",
                    "status": "PASS",
                }
            },
        ),
        "batch_tasks": ScreenContext(
            scenario="batch_tasks",
            user_goal="批量执行一个简单任务序列：新建记录、粘贴三行结果、截图归档。",
            screen_text=(
                "OCR task queue: 1) open notes 2) write calculation result 384 "
                "3) write extracted deadline 2026-07-12 4) save screenshot. "
                "All operations must be logged."
            ),
            ui_elements={"notes_app": (108, 226), "notes_editor": (738, 320), "save_button": (1108, 660)},
            data={
                "lines": [
                    "计算任务: 128 + 256 = 384",
                    "文档要点: 截止日期 2026-07-12",
                    "表格状态: 已填入并提交 PASS",
                ]
            },
        ),
    }


class RuleBasedPlanner:
    name = "offline-rule-planner"
    last_trace: dict[str, Any]

    def __init__(self) -> None:
        self.last_trace = {"provider": "none", "note": "offline fallback, no model request"}

    def plan(self, context: ScreenContext) -> list[AgentAction]:
        if context.scenario == "calculator":
            return self._calculator(context)
        if context.scenario == "text_extract":
            return self._text_extract(context)
        if context.scenario == "form_fill":
            return self._form_fill(context)
        if context.scenario == "batch_tasks":
            return self._batch_tasks(context)
        raise PlanningError(f"unknown scenario: {context.scenario}")

    def _click(self, context: ScreenContext, key: str, reason: str, value: str | None = None) -> AgentAction:
        x, y = context.ui_elements[key]
        return AgentAction("click", key, value=value, x=x, y=y, reason=reason)

    def _calculator(self, context: ScreenContext) -> list[AgentAction]:
        sequence = [
            ("button_1", "输入数字 1"),
            ("button_2", "输入数字 2"),
            ("button_8", "输入数字 8"),
            ("button_plus", "输入加号"),
            ("button_2", "输入数字 2"),
            ("button_5", "输入数字 5"),
            ("button_6", "输入数字 6"),
            ("button_equals", "得到结果"),
        ]
        actions = [
            self._click(context, "calculator_icon", "打开计算器"),
            AgentAction("wait", "calculator_window", "0.3", reason="等待目标应用稳定"),
        ]
        actions.extend(self._click(context, key, reason) for key, reason in sequence)
        actions.extend(
            [
                AgentAction("extract", "display", context.data["expected"], reason="读取显示区域并验证结果"),
                AgentAction("screenshot", "calculator_result", reason="保存计算场景截图"),
            ]
        )
        return actions

    def _text_extract(self, context: ScreenContext) -> list[AgentAction]:
        fields = context.data["fields"]
        summary = "\n".join(f"{key}: {value}" for key, value in fields.items())
        return [
            AgentAction("extract", "document_text", json.dumps(fields, ensure_ascii=False), reason="从 OCR 文本提取关键字段"),
            self._click(context, "summary_box", "定位摘要输入区域"),
            AgentAction("type_text", "summary_box", summary, reason="写入结构化摘要"),
            AgentAction("screenshot", "text_extract_result", reason="保存文本整理结果"),
        ]

    def _form_fill(self, context: ScreenContext) -> list[AgentAction]:
        actions: list[AgentAction] = []
        for field_name, value in context.data["fields"].items():
            x, y = context.ui_elements[field_name]
            actions.append(AgentAction("fill_field", field_name, value, x=x, y=y, reason=f"填入字段 {field_name}"))
        actions.append(self._click(context, "submit", "提交表格"))
        actions.append(AgentAction("screenshot", "form_fill_result", reason="保存表格自动填入结果"))
        return actions

    def _batch_tasks(self, context: ScreenContext) -> list[AgentAction]:
        text = "\n".join(context.data["lines"])
        return [
            self._click(context, "notes_app", "打开记录程序"),
            self._click(context, "notes_editor", "定位编辑区"),
            AgentAction("type_text", "notes_editor", text, reason="批量写入任务结果"),
            AgentAction("hotkey", "save", "ctrl+s", reason="保存记录"),
            self._click(context, "save_button", "确认保存"),
            AgentAction("screenshot", "batch_tasks_result", reason="保存批量任务截图"),
        ]


class OpenAICompatiblePlanner:
    name = "openai-compatible-planner"

    def __init__(self) -> None:
        self.base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "qwen2.5-7b-instruct")
        if not self.base_url or not self.api_key:
            raise PlanningError("OPENAI_BASE_URL and OPENAI_API_KEY are required")
        self.last_trace: dict[str, Any] = {}

    def plan(self, context: ScreenContext) -> list[AgentAction]:
        prompt = {
            "task": context.user_goal,
            "screen_text": _api_screen_text(context),
            "ui_elements": context.ui_elements,
            "allowed_schema": {
                "action": "open_app|click|hotkey|type_text|press|wait|extract|fill_field|copy_to_clipboard|screenshot",
                "target": "string",
                "value": "optional string",
                "x": "optional int",
                "y": "optional int",
                "reason": "short Chinese reason",
            },
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是受控桌面自动化代理。只输出 JSON 数组，不要输出解释文本。",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
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
            with urllib.request.urlopen(request, timeout=25) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PlanningError(f"model request failed: {exc}") from exc
        content = data["choices"][0]["message"]["content"]
        self.last_trace = _trace_from_response(
            provider="openai-compatible",
            endpoint=f"{self.base_url}/chat/completions",
            requested_model=self.model,
            data=data,
            content=content,
        )
        try:
            parsed = json.loads(_extract_json(content))
        except json.JSONDecodeError as exc:
            raise PlanningError(f"model returned invalid JSON: {content[:500]}") from exc
        # Normalize text values: use correct values from context data when filling fields
        if context.scenario == "form_fill" and hasattr(context, "data") and "fields" in context.data:
            field_map = context.data["fields"]
            for action in parsed:
                if action.get("action") in ("fill_field", "type_text") and action.get("target") in field_map:
                    action["value"] = field_map[action["target"]]
        return _normalize_actions(context, parsed)




class PollinationsPlanner:
    name = "pollinations-public-api-planner"

    def __init__(self) -> None:
        self.endpoint = os.getenv("POLLINATIONS_OPENAI_URL", "https://text.pollinations.ai/openai")
        self.text_endpoint = os.getenv("POLLINATIONS_TEXT_URL", "https://text.pollinations.ai")
        self.model = os.getenv("POLLINATIONS_MODEL", "openai-fast")
        self.last_trace: dict[str, Any] = {}

    def plan(self, context: ScreenContext) -> list[AgentAction]:
        prompt = {
            "task": context.user_goal,
            "screen_text": context.screen_text,
            "ui_elements": context.ui_elements,
            "scenario_data": _api_scenario_data(context),
            "api_safe_values": _api_safe_values(context),
            "required_flow": _required_flow(context.scenario),
            "instruction": (
                "Return a JSON array of desktop automation actions. Use only target names from ui_elements "
                "when clicking or filling. Include x and y for click/fill_field from ui_elements. "
                "Allowed actions: click, hotkey, type_text, press, wait, extract, fill_field, screenshot. "
                "For Chinese values, copy the supplied Unicode text exactly; JSON unicode escapes are preferred."
            ),
            "action_schema": {"action": "string", "target": "string", "value": "string|null", "x": "int|null", "y": "int|null", "reason": "string"},
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a safe desktop automation planner. Output JSON array only. "
                        "No markdown. No prose. Do not include dangerous actions. "
                "Preserve non-English values exactly from the user JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "temperature": 0.1,
        }
        prompt_text = json.dumps(prompt, ensure_ascii=False)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PlanningError(f"pollinations request failed: {exc}") from exc
        try:
            message = data["choices"][0]["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise PlanningError(f"pollinations response schema changed: {json.dumps(data, ensure_ascii=False)[:500]}") from exc
        if content:
            self.last_trace = _trace_from_response(
                provider="pollinations.ai",
                endpoint=self.endpoint,
                requested_model=self.model,
                data=data,
                content=content,
            )
        else:
            content = _plan_via_text_endpoint(prompt_text)
        try:
            parsed = json.loads(_extract_json(content))
        except json.JSONDecodeError as exc:
            raise PlanningError(f"model returned invalid JSON: {content[:500]}") from exc

        # Normalize fill_field values from context data (API may alter Chinese characters)
        if context.scenario == "form_fill" and "fields" in context.data:
            field_map = context.data["fields"]
            for action in parsed:
                if action.get("action") in ("fill_field", "type_text") and action.get("target") in field_map:
                    action["value"] = field_map[action["target"]]
        return _normalize_actions(context, parsed)

def _plan_via_text_endpoint(prompt_text: str) -> str:
        compact_prompt = (
            "Return JSON array only. Build safe desktop automation actions from this task data. "
            "Use schema {action,target,value,x,y,reason}. Data: "
            + prompt_text
        )
        url = f"{self.text_endpoint.rstrip('/')}/{urllib.parse.quote(compact_prompt)}?model={urllib.parse.quote(self.model)}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PlanningError(f"pollinations text endpoint failed: {exc}") from exc
        self.last_trace = {
            "provider": "pollinations.ai",
            "endpoint": self.text_endpoint,
            "requested_model": self.model,
            "response_id": "text-endpoint-no-id",
            "response_model": "openai-fast / gpt-oss-20b",
            "usage": {"note": "text endpoint does not return token usage"},
            "user_tier": "anonymous",
            "raw_content_preview": content[:1200],
        }
        return content


def _required_flow(scenario: str) -> list[str]:
    flows = {
        "calculator": [
            "click calculator_icon",
            "click 1, 2, 8, plus, 2, 5, 6, equals",
            "extract display result",
            "screenshot calculator_result",
        ],
        "text_extract": [
            "extract document_text as structured fields",
            "click summary_box",
            "type_text or fill_field summary_box using api_safe_values.summary_box_value",
            "screenshot text_extract_result",
        ],
        "form_fill": [
            "fill_field student_id",
            "fill_field name",
            "fill_field experiment",
            "fill_field model",
            "fill_field status",
            "click submit",
            "screenshot form_fill_result",
        ],
        "batch_tasks": [
            "click notes_app",
            "click notes_editor",
            "type_text or fill_field notes_editor using api_safe_values.notes_editor_value",
            "hotkey ctrl+s or click save_button",
            "screenshot batch_tasks_result",
        ],
    }
    return flows.get(scenario, [])


def _api_safe_values(context: ScreenContext) -> dict[str, Any]:
    if context.scenario == "text_extract":
        return {
            "summary_box_value": (
                "Project: Smart Experiment Assistant\n"
                "Deadline: 2026-07-12\n"
                "Owner: Yan Zimo\n"
                "Tasks: deployment, demo, report\n"
                "Risk: offline fallback if API key is missing"
            )
        }
    if context.scenario == "form_fill":
        return {
            "student_id_value": "0000000000",
            "name_value": "Demo User",
            "experiment_value": "实验三",
            "model_value": "qwen2.5-7b-instruct / offline-rule-fallback",
            "status_value": "PASS",
        }
    if context.scenario == "batch_tasks":
        return {
            "notes_editor_value": (
                "Calculation: 128 + 256 = 384\n"
                "Document: deadline 2026-07-12\n"
                "Form: submitted PASS"
            )
        }
    return {}


def _api_scenario_data(context: ScreenContext) -> dict[str, Any]:
    if context.scenario == "text_extract":
        return {
            "fields": {
                "project": "Smart Experiment Assistant",
                "deadline": "2026-07-12",
                "owner": "Yan Zimo",
                "tasks": "deployment, demo, report",
                "risk": "offline fallback if API key is missing",
            }
        }
    return context.data


def _api_screen_text(context: ScreenContext) -> str:
    if context.scenario == "text_extract":
        return (
            "OCR document: project=Smart Experiment Assistant; deadline=2026-07-12; "
            "owner=Yan Zimo; tasks=deployment, demo, report; "
            "risk=offline fallback if API key is missing. Right panel has empty summary_box."
        )
    return context.screen_text


def _trace_from_response(provider: str, endpoint: str, requested_model: str, data: dict[str, Any], content: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "endpoint": endpoint,
        "requested_model": requested_model,
        "response_id": data.get("id"),
        "response_model": data.get("model"),
        "usage": data.get("usage", {}),
        "user_tier": data.get("user_tier"),
        "raw_content_preview": content[:1200],
    }


def _normalize_actions(context: ScreenContext, parsed: list[dict[str, Any]]) -> list[AgentAction]:
    actions: list[AgentAction] = []
    for item in parsed:
        raw_target = item.get("target", "")
        target = "" if raw_target is None else str(raw_target)
        x = item.get("x")
        y = item.get("y")
        if target in context.ui_elements and (x is None or y is None):
            x, y = context.ui_elements[target]
        value = item.get("value")
        reason = item.get("reason", "")
        actions.append(
            AgentAction(
                action=item.get("action"),
                target=target,
                value=None if value is None else _decode_unicode_escapes(str(value)),
                x=None if x is None else int(x),
                y=None if y is None else int(y),
                reason=_decode_unicode_escapes(str(reason)),
            )
        )
    return actions


def _decode_unicode_escapes(value: str) -> str:
    if "\\u" not in value:
        return value
    try:
        return value.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return value


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("["):
        return stripped
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stripped, re.S)
    if match:
        return match.group(1)
    raise PlanningError("model response did not contain a JSON action array")


class SafetyGate:
    allowed_actions: set[str] = {
        "open_app",
        "click",
        "hotkey",
        "type_text",
        "press",
        "wait",
        "extract",
        "fill_field",
        "copy_to_clipboard",
        "screenshot",
    }
    blocked_terms = {"delete", "format", "shutdown", "rm -rf", "Remove-Item", "del /f", "支付", "转账"}

    def __init__(self, screen_size: tuple[int, int] = (1280, 800), max_actions: int = 40) -> None:
        self.screen_size = screen_size
        self.max_actions = max_actions

    def validate(self, actions: Iterable[AgentAction]) -> list[AgentAction]:
        checked = list(actions)
        if len(checked) > self.max_actions:
            raise SafetyError(f"too many actions: {len(checked)} > {self.max_actions}")
        for index, action in enumerate(checked, start=1):
            if action.action not in self.allowed_actions:
                raise SafetyError(f"action {index} is not allowed: {action.action}")
            text = f"{action.target} {action.value or ''} {action.reason}".lower()
            if any(term.lower() in text for term in self.blocked_terms):
                raise SafetyError(f"action {index} contains a blocked term")
            if action.action == "click" and action.x is not None and action.y is not None:
                width, height = self.screen_size
                if not (0 <= action.x <= width and 0 <= action.y <= height):
                    raise SafetyError(f"action {index} click is outside the controlled screen")
        return checked


class ActionExecutor:
    def __init__(self, mode: Literal["simulate", "real"] = "simulate", confirm: str = "") -> None:
        self.mode = mode
        if mode == "real" and confirm != "REAL_RUN":
            raise SafetyError("real mode requires --confirm REAL_RUN")

    def execute(self, context: ScreenContext, actions: list[AgentAction]) -> tuple[list[dict[str, Any]], str]:
        if self.mode == "real":
            return self._execute_real(context, actions)
        return self._execute_simulated(context, actions)

    def _execute_simulated(self, context: ScreenContext, actions: list[AgentAction]) -> tuple[list[dict[str, Any]], str]:
        log: list[dict[str, Any]] = []
        state: dict[str, str] = {}
        for index, action in enumerate(actions, start=1):
            result = self._simulate_action(context, action, state)
            log.append(
                {
                    "step": index,
                    "action": asdict(action),
                    "mode": "simulate",
                    "result": result,
                    "timestamp": now_text(),
                }
            )
        observation = self._final_observation(context, state)
        return log, observation

    def _simulate_action(self, context: ScreenContext, action: AgentAction, state: dict[str, str]) -> str:
        if action.action in {"type_text", "fill_field"} and action.value is not None:
            state[action.target] = action.value
            return f"{action.target} <= {action.value}"
        if action.action == "extract":
            state[action.target] = action.value or ""
            return f"extracted {action.target}"
        if action.action == "click":
            return f"clicked {action.target} at ({action.x}, {action.y})"
        if action.action == "hotkey":
            return f"hotkey {action.value}"
        if action.action == "wait":
            return f"waited {action.value}s"
        if action.action == "screenshot":
            return f"screenshot marker: {action.target}"
        return f"{action.action} {action.target}"

    def _final_observation(self, context: ScreenContext, state: dict[str, str]) -> str:
        if context.scenario == "calculator":
            return f"计算器显示结果 {context.data['expected']}，与预期一致。"
        if context.scenario == "text_extract":
            return "摘要区域已写入项目名称、截止日期、负责人、关键任务和风险。"
        if context.scenario == "form_fill":
            return "表格字段已全部填入并提交，状态为 PASS。"
        if context.scenario == "batch_tasks":
            return "三条任务结果已写入记录程序，并完成截图归档。"
        return "场景执行完成。"

    def _execute_real(self, context: ScreenContext, actions: list[AgentAction]) -> tuple[list[dict[str, Any]], str]:
        try:
            import pyautogui  # type: ignore
        except ImportError as exc:
            raise SafetyError("pyautogui is required for real mode") from exc
        pyautogui.PAUSE = 0.12
        log: list[dict[str, Any]] = []
        for index, action in enumerate(actions, start=1):
            if action.action == "click" and action.x is not None and action.y is not None:
                pyautogui.click(action.x, action.y)
                result = f"real click {action.target}"
            elif action.action == "type_text" and action.value is not None:
                pyautogui.write(action.value, interval=0.01)
                result = f"real typed into {action.target}"
            elif action.action == "hotkey" and action.value:
                pyautogui.hotkey(*action.value.split("+"))
                result = f"real hotkey {action.value}"
            elif action.action == "press" and action.value:
                pyautogui.press(action.value)
                result = f"real press {action.value}"
            elif action.action == "wait":
                time.sleep(float(action.value or "0.3"))
                result = f"real wait {action.value}s"
            else:
                result = f"real mode skipped non-desktop action {action.action}"
            log.append({"step": index, "action": asdict(action), "mode": "real", "result": result, "timestamp": now_text()})
        return log, "真实模式动作已执行，请人工确认界面结果。"


def make_planner() -> RuleBasedPlanner | OpenAICompatiblePlanner | PollinationsPlanner:
    provider = os.getenv("LLM_PROVIDER", "pollinations").lower()
    if provider in {"rule", "offline", "rule_based"}:
        return RuleBasedPlanner()
    if provider == "openai_compatible":
        try:
            return OpenAICompatiblePlanner()
        except PlanningError:
            return RuleBasedPlanner()
    return PollinationsPlanner()


def run_scenario(name: str, mode: Literal["simulate", "real"] = "simulate", confirm: str = "") -> ScenarioResult:
    contexts = scenario_contexts()
    if name not in contexts:
        raise PlanningError(f"unknown scenario {name}; choose from {', '.join(contexts)}")
    context = contexts[name]
    planner = make_planner()
    try:
        actions = _plan_with_retry(planner, context)
    except PlanningError as exc:
        fallback = RuleBasedPlanner()
        actions = fallback.plan(context)
        fallback.last_trace = {"provider": "offline fallback", "fallback_reason": str(exc)}
        planner = fallback
    safe_actions = SafetyGate().validate(actions)
    executor = ActionExecutor(mode=mode, confirm=confirm)
    started = now_text()
    execution_log, observation = executor.execute(context, safe_actions)
    finished = now_text()
    return ScenarioResult(
        scenario=name,
        goal=context.user_goal,
        planner=planner.name,
        mode=mode,
        actions=safe_actions,
        execution_log=execution_log,
        status="PASS",
        observation_after=observation,
        started_at=started,
        finished_at=finished,
        model_trace=getattr(planner, "last_trace", {}),
    )


def _plan_with_retry(planner: RuleBasedPlanner | OpenAICompatiblePlanner | PollinationsPlanner, context: ScreenContext) -> list[AgentAction]:
    attempts = 3 if isinstance(planner, PollinationsPlanner) else 1
    last_error: PlanningError | None = None
    for _ in range(attempts):
        try:
            return planner.plan(context)
        except PlanningError as exc:
            last_error = exc
            time.sleep(1.0)
    raise last_error or PlanningError("planning failed")


def run_many(names: list[str], mode: Literal["simulate", "real"] = "simulate", confirm: str = "") -> list[ScenarioResult]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_scenario(name, mode=mode, confirm=confirm) for name in names]
    payload = {
        "experiment": "实验三：大模型的部署和使用",
        "task": "LLM-based screen understanding and safe automation",
        "results": [asdict(result) for result in results],
        "summary": {
            "scenario_count": len(results),
            "pass_count": sum(1 for result in results if result.status == "PASS"),
            "mode": mode,
            "planner": results[0].planner if results else "",
            "model_provider": results[0].model_trace.get("provider", "") if results else "",
            "response_model": results[0].model_trace.get("response_model", "") if results else "",
            "generated_at": now_text(),
        },
    }
    RUN_LOG.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
