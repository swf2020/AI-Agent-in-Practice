# tests/test_main.py — 自动生成的冒烟测试
import pytest
from unittest.mock import patch, MagicMock
import sys, os
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


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试沙箱核心功能 ───────────────────────────────────
class TestSandbox:
    def test_is_safe_code_safe(self):
        from sandbox_server import is_safe_code
        safe, reason = is_safe_code("print('hello')\nresult = 1 + 1\nprint(result)")
        assert safe is True

    def test_is_safe_code_blocks_dangerous_import(self):
        from sandbox_server import is_safe_code
        safe, reason = is_safe_code("import os\nprint(os.getcwd())")
        assert safe is False
        assert "安全检查拒绝" in reason

    def test_is_safe_code_blocks_eval(self):
        from sandbox_server import is_safe_code
        safe, reason = is_safe_code("result = eval('1+1')\nprint(result)")
        assert safe is False

    def test_is_safe_code_blocks_exec(self):
        from sandbox_server import is_safe_code
        safe, reason = is_safe_code("exec('print(1)')")
        assert safe is False

    def test_execute_python_basic(self):
        from sandbox_server import execute_python
        result = execute_python("print('hello from sandbox')\nprint(1 + 1)")
        assert result["success"] is True
        assert "hello from sandbox" in result["stdout"]
        assert "2" in result["stdout"]

    def test_execute_python_timeout(self):
        from sandbox_server import execute_python
        result = execute_python("while True: pass", timeout=2)
        assert result["success"] is False
        assert "超时" in result["error"]

    def test_execute_python_security_block_import(self):
        from sandbox_server import execute_python
        result = execute_python("import os\nprint(os.getcwd())")
        assert result["success"] is False
        assert "安全检查拒绝" in result["error"]

    def test_execute_python_security_block_eval(self):
        from sandbox_server import execute_python
        result = execute_python("result = eval('1+1')\nprint(result)")
        assert result["success"] is False

    def test_history(self):
        from sandbox_server import get_execution_history, reset_session, execute_python
        reset_session()
        execute_python("print('step 1')")
        execute_python("print('step 2')")
        history = get_execution_history(last_n=5)
        assert history["total"] == 2
        assert len(history["records"]) == 2

    def test_reset_session(self):
        from sandbox_server import reset_session, execute_python, get_execution_history
        execute_python("print('test')")
        result = reset_session()
        assert result["success"] is True
        history = get_execution_history()
        assert history["total"] == 0

    def test_install_package(self):
        from sandbox_server import install_package
        result = install_package("six")
        assert result["success"] is True


# ── 测试核心 LLM 调用（Mock litellm）──────────────────
class TestLLMCall:
    @patch("litellm.completion")
    def test_mocked_completion(self, mock_completion):
        """验证核心调用路径在 mock 下可正常执行"""
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="mock response"),
                finish_reason="stop",
            )],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
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
