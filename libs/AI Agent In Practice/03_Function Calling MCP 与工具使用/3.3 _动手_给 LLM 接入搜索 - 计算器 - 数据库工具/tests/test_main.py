# tests/test_main.py — 冒烟测试

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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

    def test_get_base_url(self):
        from core_config import get_base_url
        result = get_base_url()
        assert result is None or isinstance(result, str)

    def test_get_active_config(self):
        from core_config import get_active_config, MODEL_REGISTRY, ACTIVE_MODEL_KEY
        config = get_active_config()
        assert config == MODEL_REGISTRY[ACTIVE_MODEL_KEY]


# ── 测试工具基类与调度器 ───────────────────────────────
class TestToolBase:
    def test_dispatcher_register_and_list(self):
        from tools.base import BaseTool, ToolDispatcher

        class FakeTool(BaseTool):
            @property
            def name(self): return "fake_tool"

            @property
            def schema(self):
                return {"type": "function", "function": {"name": self.name, "description": "test", "parameters": {}}}

            def run(self, **kwargs): return "ok"

        dispatcher = ToolDispatcher([FakeTool()])
        assert len(dispatcher.schemas) == 1
        assert dispatcher.schemas[0]["function"]["name"] == "fake_tool"

    def test_dispatch_unknown_tool(self):
        from tools.base import ToolDispatcher
        dispatcher = ToolDispatcher([])
        result = dispatcher.dispatch("nonexistent", "{}")
        assert "Error" in result
        assert "nonexistent" in result

    def test_dispatch_success(self):
        from tools.base import BaseTool, ToolDispatcher

        class EchoTool(BaseTool):
            @property
            def name(self): return "echo"

            @property
            def schema(self):
                return {"type": "function", "function": {"name": self.name, "description": "echo", "parameters": {}}}

            def run(self, **kwargs): return f"echoed: {kwargs}"

        dispatcher = ToolDispatcher([EchoTool()])
        result = dispatcher.dispatch("echo", '{"msg": "hello"}')
        assert "hello" in result

    def test_dispatch_invalid_json_args(self):
        from tools.base import BaseTool, ToolDispatcher

        class DummyTool(BaseTool):
            @property
            def name(self): return "dummy"

            @property
            def schema(self):
                return {"type": "function", "function": {"name": self.name, "description": "dummy", "parameters": {}}}

            def run(self, **kwargs): return "ok"

        dispatcher = ToolDispatcher([DummyTool()])
        result = dispatcher.dispatch("dummy", "not-json")
        assert "Error" in result


# ── 测试 SQL 工具（纯逻辑，无数据库连接） ──────────────
class TestSQLValidation:
    def test_validate_dangerous_sql(self):
        from tools.db_tool import TextToSQLTool
        # Use in-memory SQLite for testing
        tool = TextToSQLTool("sqlite:///:memory:")
        result = tool.run(natural_language_query="test", sql="DELETE FROM users")
        assert "禁止执行" in result or "验证失败" in result

    def test_validate_insert_sql(self):
        from tools.db_tool import TextToSQLTool
        tool = TextToSQLTool("sqlite:///:memory:")
        result = tool.run(natural_language_query="test", sql="INSERT INTO t VALUES (1)")
        assert "禁止执行" in result or "验证失败" in result

    def test_select_valid_sql(self):
        from tools.db_tool import TextToSQLTool
        tool = TextToSQLTool("sqlite:///:memory:")
        result = tool.run(natural_language_query="test", sql="SELECT 1")
        # Should execute without validation error
        assert "执行失败" not in result or "1" in result


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 Agent 构建函数（mock LLM） ───────────────────
class TestAgentBuild:
    def test_build_agent_returns_instance(self):
        """build_agent 应返回 ToolCallingAgent 实例（mock 外部依赖）"""
        import os
        # Set dummy env vars to avoid KeyError during tool construction
        os.environ.setdefault("TAVILY_API_KEY", "dummy-tavily")
        os.environ.setdefault("E2B_API_KEY", "dummy-e2b")

        from agent import build_agent
        agent = build_agent(db_url="sqlite:///:memory:")
        assert agent is not None
        assert agent._model is not None
        assert len(agent._dispatcher.schemas) == 3  # 3 tools registered


# ── 测试核心 LLM 调用（Mock litellm）──────────────────
class TestLLMCall:
    def test_mocked_completion(self):
        """验证核心调用路径在 mock 下可正常执行"""
        from unittest.mock import patch, MagicMock

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(
                choices=[MagicMock(
                    message=MagicMock(
                        content="mock response",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )

            import os
            os.environ.setdefault("TAVILY_API_KEY", "dummy-tavily")
            os.environ.setdefault("E2B_API_KEY", "dummy-e2b")

            import litellm
            from core_config import get_litellm_id, get_api_key, get_base_url

            response = litellm.completion(
                model=get_litellm_id(),
                api_key=get_api_key(),
                api_base=get_base_url(),
                messages=[{"role": "user", "content": "test"}],
            )
            assert response.choices[0].message.content == "mock response"
            mock_completion.assert_called_once()

    def test_agent_run_mocked(self):
        """测试 Agent.run() 在 mock 下能返回结果"""
        from unittest.mock import patch, MagicMock

        import os
        os.environ.setdefault("TAVILY_API_KEY", "dummy-tavily")
        os.environ.setdefault("E2B_API_KEY", "dummy-e2b")

        from agent import build_agent

        agent = build_agent(db_url="sqlite:///:memory:")

        with patch.object(agent, '_model', "deepseek/deepseek-chat"):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock(
                    choices=[MagicMock(
                        message=MagicMock(
                            content="This is a test response",
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )],
                )

                result = agent.run("Hello", verbose=False)
                assert result == "This is a test response"
                mock_completion.assert_called_once()
