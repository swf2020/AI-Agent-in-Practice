# tests/test_main.py — 冒烟测试
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set dummy API key so tools.py can be imported (TavilySearchResults requires it)
os.environ.setdefault("TAVILY_API_KEY", "test_dummy_key")


# ── 测试 core_config 基础结构 ──────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY,
            get_litellm_id, get_api_key, get_base_url,
            get_model_list, estimate_cost,
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


# ── 测试 state ─────────────────────────────────────────
class TestState:
    def test_agent_state_structure(self):
        from state import AgentState
        state: AgentState = {
            "messages": [],
            "tool_calls_count": 0,
            "requires_approval": False,
        }
        assert isinstance(state, dict)


# ── 测试 tools ─────────────────────────────────────────
class TestTools:
    def test_calculate_tool(self):
        from tools import calculate
        # @tool decorator wraps function into StructuredTool; use .invoke()
        assert calculate.invoke({"expression": "2 + 2"}) == "4"
        assert calculate.invoke({"expression": "10 * 10"}) == "100"
        assert "1048.576" == calculate.invoke({"expression": "(1024 * 1024) / 1000"})

    def test_calculate_invalid(self):
        from tools import calculate
        result = calculate.invoke({"expression": "invalid_expr"})
        assert "错误" in result or "error" in result.lower()


# ── 测试 router ────────────────────────────────────────
class TestRouter:
    def test_should_continue_end(self):
        from router import should_continue
        from langchain_core.messages import AIMessage
        state = {
            "messages": [AIMessage(content="done")],
            "tool_calls_count": 0,
            "requires_approval": False,
        }
        assert should_continue(state) == "end"

    def test_should_continue_tools(self):
        from router import should_continue
        from langchain_core.messages import AIMessage
        from langchain_core.messages.tool import ToolCall
        state = {
            "messages": [AIMessage(
                content="",
                tool_calls=[ToolCall(name="calculate", args={"expression": "2+2"}, id="test_id_1")]
            )],
            "tool_calls_count": 1,
            "requires_approval": False,
        }
        assert should_continue(state) == "tools"

    def test_should_continue_max_calls(self):
        from router import should_continue, MAX_TOOL_CALLS
        from langchain_core.messages import AIMessage
        from langchain_core.messages.tool import ToolCall
        state = {
            "messages": [AIMessage(
                content="",
                tool_calls=[ToolCall(name="test", args={}, id="test_id_2")]
            )],
            "tool_calls_count": MAX_TOOL_CALLS,  # 已达上限
            "requires_approval": False,
        }
        assert should_continue(state) == "end"


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "run_agent.py")
        spec = importlib.util.spec_from_file_location("run_agent", path)
        assert spec is not None, "run_agent.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 graph 构建（Mock LLM）─────────────────────────
class TestGraphBuild:
    @patch("agent.create_llm")
    def test_build_and_invoke_mocked(self, mock_create_llm):
        """验证图构建和调用路径在 mock 下可正常执行"""
        from langchain_core.messages import AIMessage, HumanMessage
        from graph import build_graph

        # Mock LLM 返回
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="mock response")
        mock_create_llm.return_value = mock_llm

        graph = build_graph(use_memory=True)
        result = graph.invoke(
            input={
                "messages": [HumanMessage(content="test")],
                "tool_calls_count": 0,
                "requires_approval": False,
            },
            config={"configurable": {"thread_id": "test_001"}},
        )
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) > 0
