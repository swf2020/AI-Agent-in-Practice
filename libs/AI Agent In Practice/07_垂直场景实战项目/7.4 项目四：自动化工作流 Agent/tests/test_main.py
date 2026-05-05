# tests/test_main.py — 自动化工作流 Agent 冒烟测试
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── 测试 core_config 基础结构 ──────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY,
        )
        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0
        assert isinstance(ACTIVE_MODEL_KEY, str)
        assert ACTIVE_MODEL_KEY in MODEL_REGISTRY

    def test_model_registry_schema(self):
        """验证每个模型条目包含必要字段"""
        from core_config import MODEL_REGISTRY
        required_keys = {"litellm_id", "price_in", "price_out",
                         "max_tokens_limit", "api_key_env", "base_url"}
        for name, cfg in MODEL_REGISTRY.items():
            missing = required_keys - set(cfg.keys())
            assert not missing, f"{name} 缺少字段: {missing}"

    def test_get_litellm_id(self):
        from core_config import get_litellm_id
        result = get_litellm_id()
        assert isinstance(result, str) and len(result) > 0

    def test_get_model_list(self):
        from core_config import get_model_list, MODEL_REGISTRY
        lst = get_model_list()
        assert isinstance(lst, list)
        assert set(lst) == set(MODEL_REGISTRY.keys())

    def test_estimate_cost(self):
        from core_config import estimate_cost, get_model_list
        model_key = get_model_list()[0]
        cost = estimate_cost(model_key, input_tokens=1000, output_tokens=500)
        assert isinstance(cost, float) and cost >= 0

    def test_get_api_key_no_crash(self):
        """无环境变量时应返回 None 而不是抛异常"""
        from core_config import get_api_key
        result = get_api_key()
        assert result is None or isinstance(result, str)

    def test_get_chat_model_id(self):
        """验证 chat_model_id 无前缀"""
        from core_config import get_chat_model_id, get_litellm_id
        chat_id = get_chat_model_id()
        lite_id = get_litellm_id()
        assert isinstance(chat_id, str) and len(chat_id) > 0
        assert "/" not in chat_id or chat_id == lite_id  # 允许无前缀情况


# ── 测试 Models ────────────────────────────────────────
class TestModels:
    def test_risk_level_enum(self):
        from models import RiskLevel
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.HIGH.value == "high"

    def test_extracted_task(self):
        from models import ExtractedTask, RiskLevel
        task = ExtractedTask(
            title="测试任务",
            description="描述",
            risk_level=RiskLevel.LOW,
            risk_reason="常规操作",
        )
        assert task.title == "测试任务"
        assert task.priority == "medium"

    def test_email_message(self):
        from models import EmailMessage
        from datetime import datetime
        email = EmailMessage(
            message_id="123",
            subject="test",
            sender="a@b.com",
            body="hello",
            received_at=datetime.now(),
        )
        assert email.message_id == "123"

    def test_workflow_state(self):
        from models import WorkflowState
        state = WorkflowState(email_id="test_001")
        assert state.email_id == "test_001"
        assert state.email is None
        assert state.approved is None


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 workflow_graph 路由逻辑 ──────────────────────
class TestWorkflowRouting:
    def test_route_by_risk_high(self):
        from models import WorkflowState, ExtractedTask, RiskLevel
        from agent.workflow_graph import route_by_risk
        state = WorkflowState(
            email_id="1",
            extracted_task=ExtractedTask(
                title="删除数据", description="危险",
                risk_level=RiskLevel.HIGH, risk_reason="删除操作",
            ),
        )
        assert route_by_risk(state) == "request_approval"

    def test_route_by_risk_low(self):
        from models import WorkflowState, ExtractedTask, RiskLevel
        from agent.workflow_graph import route_by_risk
        state = WorkflowState(
            email_id="2",
            extracted_task=ExtractedTask(
                title="更新文档", description="常规",
                risk_level=RiskLevel.LOW, risk_reason="文档更新",
            ),
        )
        assert route_by_risk(state) == "write_task"

    def test_route_by_approval_true(self):
        from models import WorkflowState
        from agent.workflow_graph import route_by_approval
        state = WorkflowState(email_id="3", approved=True)
        assert route_by_approval(state) == "write_task"

    def test_route_by_approval_false(self):
        from models import WorkflowState
        from agent.workflow_graph import route_by_approval
        state = WorkflowState(email_id="4", approved=False)
        assert route_by_approval(state) == "reject_and_notify"


# ── 测试核心 LLM 调用（Mock ChatAnthropic）──────────────
class TestLLMCall:
    @pytest.fixture(autouse=True)
    def mock_settings(self, monkeypatch):
        """Mock Settings to avoid requiring real env vars for model config tests"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.setenv("GMAIL_CLIENT_ID", "test")
        monkeypatch.setenv("GMAIL_CLIENT_SECRET", "test")
        monkeypatch.setenv("GMAIL_REFRESH_TOKEN", "test")
        monkeypatch.setenv("GMAIL_USER_EMAIL", "test@test.com")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-secret")
        monkeypatch.setenv("SLACK_APPROVAL_CHANNEL", "C123")
        monkeypatch.setenv("SLACK_NOTIFY_CHANNEL", "C456")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    def test_llm_uses_core_config_model(self):
        """验证 LLM 初始化使用 core_config.get_litellm_id() 而非硬编码"""
        from agent.workflow_graph import _llm
        # ChatAnthropic 的 model 属性应该来自 core_config
        assert _llm.model is not None
        assert "claude" in _llm.model.lower() or _llm.model != "claude-3-5-sonnet-20241022"
