"""实验三 Branch B — 自动化代理测试用例"""

import json
import os
import time
import sys
import tempfile
from pathlib import Path

# 确保测试可导入 src 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_auto_agent import (
    scenario_contexts,
    RuleBasedPlanner,
    SafetyGate,
    ActionExecutor,
    SafetyError,
    PlanningError,
    AgentAction,
    ScreenContext,
)


# ===== 测试场景定义 =====

def test_scenario_contexts_complete():
    """验证四个场景定义完整"""
    contexts = scenario_contexts()
    assert len(contexts) == 4
    for name in ["calculator", "text_extract", "form_fill", "batch_tasks"]:
        assert name in contexts, f"缺少场景: {name}"


def test_scenario_context_structure():
    """验证场景上下文数据结构"""
    for name, ctx in scenario_contexts().items():
        assert isinstance(ctx, ScreenContext)
        assert ctx.scenario == name
        assert len(ctx.user_goal) > 0
        assert len(ctx.screen_text) > 0
        assert len(ctx.ui_elements) > 0


# ===== 测试规则规划器 =====

class TestRuleBasedPlanner:
    planner = RuleBasedPlanner()

    def test_calculator_plan(self):
        ctx = scenario_contexts()["calculator"]
        actions = self.planner.plan(ctx)
        assert len(actions) > 5
        # 验证包含关键动作
        actions_text = " ".join(a.action for a in actions)
        assert "click" in actions_text, "计算器规划缺少 click 动作"
        assert "screenshot" in actions_text, "计算器规划缺少 screenshot"

    def test_text_extract_plan(self):
        ctx = scenario_contexts()["text_extract"]
        actions = self.planner.plan(ctx)
        assert len(actions) >= 3
        actions_text = " ".join(a.action for a in actions)
        assert "extract" in actions_text
        assert "type_text" in actions_text or "fill_field" in actions_text

    def test_form_fill_plan(self):
        ctx = scenario_contexts()["form_fill"]
        actions = self.planner.plan(ctx)
        assert len(actions) >= 5, f"表格填入动作数不足: {len(actions)}"
        fill_count = sum(1 for a in actions if a.action == "fill_field")
        assert fill_count >= 4, f"字段填入动作数不足: {fill_count}"

    def test_batch_tasks_plan(self):
        ctx = scenario_contexts()["batch_tasks"]
        actions = self.planner.plan(ctx)
        assert len(actions) >= 4
        actions_text = " ".join(a.action for a in actions)
        assert "click" in actions_text
        assert "screenshot" in actions_text


# ===== 测试安全门 =====

class TestSafetyGate:
    gate = SafetyGate()

    def test_valid_actions_pass(self):
        ctx = scenario_contexts()["calculator"]
        planner = RuleBasedPlanner()
        actions = planner.plan(ctx)
        validated = self.gate.validate(actions)
        assert len(validated) == len(actions)

    def test_blocked_action_rejected(self):
        blocked = AgentAction("delete", "file", value="test", reason="should be blocked")
        try:
            self.gate.validate([blocked])
            assert False, "应阻止危险动作"
        except SafetyError:
            pass

    def test_too_many_actions_rejected(self):
        actions = [AgentAction("click", "x") for _ in range(50)]
        try:
            self.gate.validate(actions)
            assert False, "应阻止超量动作"
        except SafetyError:
            pass

    def test_out_of_bounds_click_rejected(self):
        out = AgentAction("click", "outside", x=9999, y=9999, reason="outside screen")
        try:
            self.gate.validate([out])
            assert False, "应阻止越界点击"
        except SafetyError:
            pass


# ===== 测试动作执行器 =====

class TestActionExecutor:
    executor_sim = ActionExecutor(mode="simulate")

    def test_simulate_calculator(self):
        ctx = scenario_contexts()["calculator"]
        planner = RuleBasedPlanner()
        actions = planner.plan(ctx)
        log, obs = self.executor_sim.execute(ctx, actions)
        assert len(log) == len(actions)
        assert "384" in obs
        for entry in log:
            assert "step" in entry
            assert "action" in entry
            assert "result" in entry

    def test_simulate_text_extract(self):
        ctx = scenario_contexts()["text_extract"]
        planner = RuleBasedPlanner()
        actions = planner.plan(ctx)
        log, obs = self.executor_sim.execute(ctx, actions)
        assert len(log) > 0
        assert "摘要" in obs or "项目" in obs

    def test_simulate_form_fill(self):
        ctx = scenario_contexts()["form_fill"]
        planner = RuleBasedPlanner()
        actions = planner.plan(ctx)
        log, obs = self.executor_sim.execute(ctx, actions)
        assert len(log) > 0
        assert "PASS" in obs

    def test_real_mode_requires_confirm(self):
        try:
            ActionExecutor(mode="real")
            assert False, "真实模式应要求确认词"
        except SafetyError as e:
            assert "REAL_RUN" in str(e)


# ===== 测试完整的场景运行流程 =====

class TestScenarioRun:
    def test_run_calculator(self):
        """测试完整运行计算器场景"""
        from llm_auto_agent import run_scenario
        result = run_scenario("calculator", mode="simulate")
        assert result.status == "PASS"
        assert len(result.actions) > 0
        assert len(result.execution_log) > 0

    def test_run_text_extract(self):
        from llm_auto_agent import run_scenario
        result = run_scenario("text_extract", mode="simulate")
        assert result.status == "PASS"

    def test_run_form_fill(self):
        from llm_auto_agent import run_scenario
        result = run_scenario("form_fill", mode="simulate")
        assert result.status == "PASS"

    def test_run_batch_tasks(self):
        from llm_auto_agent import run_scenario
        result = run_scenario("batch_tasks", mode="simulate")
        assert result.status == "PASS"

    def test_run_many(self):
        """测试批量运行"""
        from llm_auto_agent import run_many
        results = run_many(["calculator", "text_extract"], mode="simulate")
        assert len(results) == 2
        for r in results:
            assert r.status == "PASS"


# ===== 测试错误处理 =====

class TestErrorHandling:
    def test_unknown_scenario(self):
        from llm_auto_agent import run_scenario
        try:
            run_scenario("unknown_scenario_xyz", mode="simulate")
            assert False, "未知场景应抛出异常"
        except PlanningError:
            pass

    def test_unsafe_action_detection(self):
        """测试安全门能检测各种危险动作"""
        for term in ["delete", "format", "shutdown", "rm -rf", "支付", "转账"]:
            action = AgentAction("click", "test", value=term, reason="test")
            try:
                SafetyGate().validate([action])
                assert False, f"应阻止包含 {term} 的动作"
            except SafetyError:
                pass


# ===== 测试数据完整性 =====

class TestDataIntegrity:
    def test_action_dataclass_defaults(self):
        """验证 AgentAction 默认值"""
        a = AgentAction("click", "target")
        assert a.value is None
        assert a.x is None
        assert a.y is None
        assert a.reason == ""

    def test_scenario_result_complete(self):
        """验证 ScenarioResult 包含所有必要字段"""
        from llm_auto_agent import run_scenario
        result = run_scenario("calculator", mode="simulate")
        assert result.scenario is not None
        assert result.planner is not None
        assert result.status is not None
        assert result.mode is not None
        assert result.observation_after is not None


# ===== 测试环境变量配置 =====

class TestConfiguration:
    def test_make_planner_default(self):
        """验证默认生成 Pollinations 规划器"""
        from llm_auto_agent import make_planner
        planner = make_planner()
        assert planner is not None

    def test_make_planner_rule(self):
        """验证环境变量可切换规划器"""
        from llm_auto_agent import make_planner
        old = os.environ.get("LLM_PROVIDER")
        try:
            os.environ["LLM_PROVIDER"] = "rule"
            planner = make_planner()
            assert planner.name == "offline-rule-planner"
        finally:
            if old:
                os.environ["LLM_PROVIDER"] = old
            else:
                os.environ.pop("LLM_PROVIDER", None)


# ===== 测试输出产物 =====

class TestOutputArtifacts:
    def test_run_log_json(self):
        """验证运行日志可导出为 JSON"""
        from llm_auto_agent import run_many, RUN_LOG
        result = run_many(["calculator"], mode="simulate")
        assert RUN_LOG.exists()
        data = json.loads(RUN_LOG.read_text(encoding="utf-8"))
        assert "experiment" in data
        assert "results" in data
        assert data["summary"]["scenario_count"] >= 1
        assert data["summary"]["pass_count"] >= 1


if __name__ == "__main__":
    # 手动运行测试（也可用 pytest）
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
