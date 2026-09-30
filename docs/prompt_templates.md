# 提示模板设计文档 (Prompt Template Design)

## 实验四：任务编排与系统集成实战 - Branch B

---

## 一、模板体系概述

本实验设计了三层提示模板体系，用于大模型（LLM）在桌面自动化闭环中的决策与规划：

| 层级 | 用途 | 调用时机 |
|---|---|---|
| **系统提示词** | 设定代理角色与行为约束 | 每次LLM请求 |
| **任务规划提示** | 将屏幕状态转化为操作序列 | 决策阶段 |
| **验证提示** | 校验执行结果是否正确 | 验证阶段 |

---

## 二、系统提示词 (System Prompt)

### 模板内容


你是桌面自动化编排代理。根据当前屏幕状态、用户目标和历史上下文，
生成下一步可执行的动作序列。只输出JSON数组，不要输出解释文本。

动作类型：screenshot截图, ocr_extract文字提取, click点击,
type_text输入, hotkey快捷键, press按键, open_app打开应用,
wait等待, verify验证, save保存

JSON Schema:
{"action":"string","target":"string","value":"string|null",
 "x":"int|null","y":"int|null","reason":"中文理由",
 "expected":"预期值|null"}


### 设计要点
- **角色限定**：明确代理为"桌面自动化编排代理"，避免通用回答
- **输出格式约束**：严格要求 JSON 数组，禁止解释文本，便于程序解析
- **安全约束**：隐式约束——只生成白名单内的动作类型
- **Schema 示例**：提供完整的 JSON 结构，降低模型输出格式错误率

---

## 三、任务规划提示 (Task Planning Prompt)

### 模板结构


【用户目标】{用户任务描述}
【任务描述】{当前场景说明}
【当前阶段】{perceive / decide / execute / verify}
【已执行循环】第N次
【当前屏幕文字】{OCR提取的屏幕文本}
【可操作的目标元素】{UI元素名 -> 坐标的JSON映射}
【已提取数据】{历史提取的结构化数据}
【最近操作历史】
  [OK] perceive: ocr_extract screen -> ...
  [FAIL] execute: click button_1 -> ...


### 设计要点

| 字段 | 作用 | 示例 |
|---|---|---|
| 用户目标 | 提供任务的顶层目标 | 打开记事本并输入文字保存 |
| 当前阶段 | 告知闭环的当前位置 | perceive / decide / execute / verify |
| 已执行循环 | 避免无限循环 | 第3次 |
| 屏幕文字 | 感知阶段的输入 | "OCR: Notepad window detected" |
| 目标元素 | 可点击/可操作的 UI 坐标 | {"start_button": [60, 1050]} |
| 已提取数据 | 上下文记忆 | {"filename": "实验报告.txt"} |
| 操作历史 | 反馈前序步骤结果 | 帮助模型从错误中恢复 |

---

## 四、场景专用提示模板

### 场景1：记事本自动化 (Notepad)

**任务规划提示示例：**

【用户目标】打开记事本，输入一段实验报告文字，保存到桌面。
【当前屏幕文字】OCRscreenshot: Windows desktop. Recycle Bin, This PC, User Folder icons visible.
Start button at bottom-left. No Notepad window open.
【可操作的目标元素】
{
  "start_button": [60, 1050],
  "notepad_search": [600, 150],
  "notepad_icon_result": [400, 350],
  "notepad_editor": [800, 400],
  "file_menu": [80, 60],
  "save_as": [200, 300],
  "file_name_input": [600, 450],
  "save_button": [800, 550]
}


**期望输出JSON：**
```json
[
  {"action": "open_app", "target": "notepad", "reason": "打开记事本应用"},
  {"action": "wait", "target": "app_ready", "value": "1.0", "reason": "等待应用启动"},
  {"action": "type_text", "target": "editor",
   "value": "实验四：任务编排与系统集成实战\n学号：0000000000  姓名：Demo User",
   "reason": "输入实验报告文字"},
  {"action": "hotkey", "target": "save", "value": "ctrl+s", "reason": "保存文件"},
  {"action": "type_text", "target": "filename",
   "value": "demo_closed_loop.txt", "reason": "输入文件名"},
  {"action": "press", "target": "confirm", "value": "enter", "reason": "确认保存"}
]


### 场景4：扫雷自动化 (Minesweeper)

**任务规划提示示例：**

【用户目标】自动完成一轮扫雷游戏：识别游戏区域、标记地雷、点击安全格。
【当前屏幕文字】OCRscreenshot: Minesweeper 9x9 grid. Mine counter = 10.
All cells unopened. First move is safe.
【可操作的目标元素】
{
  "game_window": [400, 350],
  "center_cell": [400, 400],
  "cell_top_left": [250, 200],
  "smiley_face": [400, 150]
}


**期望输出JSON：**
```json
[
  {"action": "click", "target": "center_cell", "x": 400, "y": 400,
   "reason": "点击中央单元格开局", "expected": "1"},
  {"action": "wait", "target": "game_update", "value": "0.5",
   "reason": "等待游戏状态更新"}
]


---

## 五、验证提示模板 (Verification Prompt)

### 模板内容

验证阶段的提示词较为简洁，使用系统提示 + 当前截图OCR结果 + 预期值：


验证以下操作是否成功执行：
【目标操作】{最近执行的动作描述}
【预期结果】{expected字段的值}
【当前屏幕OCR】{验证截图提取的文字}
【判断标准】如果OCR结果包含预期值，则验证通过；否则失败。


### 设计要点
- 使用截图 OCR 结果作为客观验证依据
- 支持多预期值匹配（任一匹配即通过）
- 验证失败触发重试循环

---

## 六、API调用格式

### 请求结构 (OpenAI-compatible)

```json
{
  "model": "qwen2.5-7b-instruct",
  "messages": [
    {"role": "system", "content": "你是受控桌面自动化代理..."},
    {"role": "user", "content": "{\"user_goal\":\"打开记事本...\""}
  ],
  "temperature": 0.1
}


### 响应解析流程


模型响应 -> JSON提取 -> 动作类型标准化 -> 坐标映射 -> SafetyGate校验 -> 执行


### 超时与重试策略

| 参数 | 值 |
|---|---|
| 请求超时 | 30s (OpenAI) / 45s (Pollinations) |
| 最大重试次数 | 3 次 |
| 重试间隔 | 1s |
| 降级策略 | LLM失败 -> 规则规划器(RuleBasedPlanner) |

---

## 七、模板优化经验

1. **温度参数**：设为 0.1 降低随机性，确保动作序列的确定性
2. **历史上下文**：最近5步操作作为prompt上下文，帮助模型理解当前状态
3. **原子动作**：每个操作为最小不可分割的单位，便于安全校验和错误追踪
4. **中文理由**：要求模型为每个动作提供中文理由，增强可解释性
5. **坐标回填**：如果模型只返回目标名未返回坐标，系统自动从可用元素表回填坐标
