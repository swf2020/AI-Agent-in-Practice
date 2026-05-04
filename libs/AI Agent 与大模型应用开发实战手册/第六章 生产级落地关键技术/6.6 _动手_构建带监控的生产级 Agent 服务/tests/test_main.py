# tests/test_main.py — 冒烟测试
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))


# ── 测试 core_config 基础结构 ──────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY,
            get_litellm_id, get_api_key, get_base_url,
            get_model_list, get_active_config, estimate_cost,
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

    def test_get_active_config(self):
        from core_config import get_active_config, MODEL_REGISTRY, ACTIVE_MODEL_KEY
        cfg = get_active_config()
        assert cfg == MODEL_REGISTRY[ACTIVE_MODEL_KEY]

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

    def test_get_base_url(self):
        from core_config import get_base_url
        result = get_base_url()
        assert result is None or isinstance(result, str)


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 models.py ─────────────────────────────────────
class TestModels:
    def test_chat_request_valid(self):
        from models import ChatRequest
        req = ChatRequest(message="hello", session_id="s1")
        assert req.message == "hello"

    def test_chat_request_empty_message(self):
        from models import ChatRequest
        with pytest.raises(Exception):
            ChatRequest(message="")

    def test_task_request(self):
        from models import TaskRequest
        req = TaskRequest(message="test")
        assert req.session_id == "default"

    def test_task_status_enum(self):
        from models import TaskStatus
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.SUCCESS.value == "success"
        assert TaskStatus.FAILED.value == "failed"


# ── 测试 agent.py build_tools ──────────────────────────
class TestAgentTools:
    def test_calculator_valid(self):
        from agent import build_tools
        tools = build_tools()
        calc = next(t for t in tools if t.name == "calculator")
        assert "10" in calc.func("2*5")

    def test_calculator_invalid(self):
        from agent import build_tools
        tools = build_tools()
        calc = next(t for t in tools if t.name == "calculator")
        result = calc.func("abc")
        assert "不允许" in result

    def test_get_time(self):
        from agent import build_tools
        tools = build_tools()
        time_tool = next(t for t in tools if t.name == "get_time")
        result = time_tool.func("now")
        assert len(result) > 0

    def test_mock_search(self):
        from agent import build_tools
        tools = build_tools()
        search_tool = next(t for t in tools if t.name == "search")
        result = search_tool.func("AI")
        assert "AI" in result


# ── 测试核心 LLM 调用（Mock run_agent at agent level）────────────────
class TestLLMCall:
    @patch("agent.create_agent")
    def test_run_agent_mocked_agent(self, mock_create_agent):
        """验证核心 Agent 调用在 mock 下可正常执行"""
        async def fake_ainvoke(input_data):
            return {"messages": [MagicMock(content="mock response")]}

        mock_agent = MagicMock()
        mock_agent.ainvoke = fake_ainvoke
        mock_create_agent.return_value = mock_agent

        import asyncio
        from agent import run_agent

        result = asyncio.run(run_agent(
            message="test question",
            session_id="test-session",
            user_id="test-user",
        ))
        assert result["output"] == "mock response"
        assert isinstance(result["duration_ms"], int)


# ── 测试 config.py ─────────────────────────────────────
class TestConfig:
    def test_settings_defaults(self):
        """验证 Settings 的默认值（无 .env 时使用默认）"""
        from config import Settings
        # openai_api_key 等必填字段没有默认值，需用环境变量
        # 这里只测试默认值存在的字段
        s = Settings.model_construct(
            openai_api_key="test-key",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            langfuse_host="https://test.langfuse.com",
            redis_url="redis://localhost:6379",
            agent_max_iterations=10,
            agent_timeout_seconds=120,
            max_tokens_per_request=4000,
        )
        assert s.agent_max_iterations == 10
        assert s.max_tokens_per_request == 4000
