"""
实验四：场景注册表
定义所有可用的任务场景及其初始状态、目标元素和目标描述
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneDef:
    name: str
    goal: str
    description: str
    initial_screen_text: str
    targets: dict[str, tuple[int, int]]
    data: dict[str, Any] = field(default_factory=dict)


# ── 场景1：记事本 - 打开→输入文字→保存到桌面 ─────────────────
SCENE_NOTEPAD = SceneDef(
    name="notepad",
    goal="打开记事本，输入一段实验报告文字，保存到桌面。",
    description="场景1：从桌面截图识别记事本是否存在，若不存在则打开开始菜单搜索Notepad启动，然后输入文字，最后保存到桌面。",
    initial_screen_text=(
        "OCRscreenshot: Windows desktop. Desktop icons include: Recycle Bin, This PC, "
        "User Folder. Taskbar visible at bottom. Start button at bottom-left. "
        "No Notepad window is currently open. Desktop area is clear."
    ),
    targets={
        "start_button": (60, 1050),
        "notepad_search": (600, 150),
        "notepad_icon_result": (400, 350),
        "notepad_editor": (800, 400),
        "file_menu": (80, 60),
        "save_as": (200, 300),
        "file_name_input": (600, 450),
        "save_button": (800, 550),
        "desktop_shortcut": (150, 200),
        "notepad_title_bar": (600, 30),
    },
    data={
        "text_to_type": "实验四：任务编排与系统集成实战\n"
                       "学号：0000000000  姓名：Demo User\n\n"
                       "本实验通过集成屏幕识别（实验1）、自动化操作（实验2）和"
                       "大模型API（实验3），实现了完整的感知→决策→执行→验证闭环。",
        "filename": "demo_closed_loop.txt",
        "expected_on_save": ".txt",
    },
)


# ── 场景4：游戏自动化（模拟扫雷点击）───────────────────────────
SCENE_MINESWEEPER = SceneDef(
    name="minesweeper",
    goal="自动完成一轮扫雷游戏：识别游戏区域、标记地雷、点击安全格。",
    description="场景4：截图识别扫雷游戏状态，调用大模型决策下一步操作，循环直到游戏结束或地雷全部标记。",
    initial_screen_text=(
        "OCRscreenshot: Minesweeper game window. Grid size approximately 9x9 cells. "
        "Top row: mine counter = 10, smiley face. "
        "All cells are unopened (grey squares). First move is safe."
    ),
    targets={
        "game_window": (400, 350),
        "cell_top_left": (250, 200),
        "cell_top_right": (550, 200),
        "cell_bottom_left": (250, 600),
        "cell_bottom_right": (550, 600),
        "smiley_face": (400, 150),
        "mine_counter": (350, 150),
        "center_cell": (400, 400),
    },
    data={
        "grid_size": (9, 9),
        "total_mines": 10,
        "safe_first_move": True,
    },
)


# ── 场景字典 ────────────────────────────────────────────────
ALL_SCENES: dict[str, SceneDef] = {
    "notepad": SCENE_NOTEPAD,
    "minesweeper": SCENE_MINESWEEPER,
}


def get_scene(name: str) -> SceneDef:
    if name not in ALL_SCENES:
        raise ValueError(f"未知场景: {name}，可用场景: {list(ALL_SCENES.keys())}")
    return ALL_SCENES[name]
