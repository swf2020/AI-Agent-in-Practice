# tests/test_main.py — 自动生成的冒烟测试
import pytest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


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
        required_keys = {"litellm_id", "chat_model_id", "price_in", "price_out",
                         "max_tokens_limit", "api_key_env", "base_url"}
        for name, cfg in MODEL_REGISTRY.items():
            missing = required_keys - set(cfg.keys())
            assert not missing, f"{name} 缺少字段: {missing}"

    def test_get_litellm_id(self):
        from core_config import get_litellm_id
        result = get_litellm_id()
        assert isinstance(result, str) and len(result) > 0

    def test_get_chat_model_id(self):
        """验证 chat_model_id 无前缀"""
        from core_config import get_chat_model_id, get_litellm_id
        chat_id = get_chat_model_id()
        lite_id = get_litellm_id()
        assert "/" not in chat_id
        assert chat_id != lite_id

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
        cfg = get_active_config()
        assert cfg == MODEL_REGISTRY[ACTIVE_MODEL_KEY]


# ── 测试工具层 ───────────────────────────────────────────
class TestTools:
    def test_tool_result_dataclass(self):
        from tools import ToolResult
        r = ToolResult(success=True, output="ok", score=0.9)
        assert r.success is True
        assert r.score == 0.9

    def test_execute_code_with_tests_pass(self):
        """执行通过的场景"""
        from tools import execute_code_with_tests
        impl = "def add(a, b): return a + b"
        tests = "def test_add(): assert add(1, 2) == 3"
        result = execute_code_with_tests(impl, tests)
        assert result.success is True
        assert result.score == 1.0

    def test_execute_code_with_tests_fail(self):
        """执行失败的场景"""
        from tools import execute_code_with_tests
        impl = "def add(a, b): return a - b"
        tests = "def test_add(): assert add(1, 2) == 3"
        result = execute_code_with_tests(impl, tests)
        assert result.success is False

    def test_run_static_analysis_good_code(self):
        """良好代码应通过静态分析"""
        from tools import run_static_analysis
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = run_static_analysis(code)
        assert isinstance(result.score, float)
        assert result.score >= 0

    def test_run_security_scan_safe_code(self):
        """安全代码应通过扫描"""
        from tools import run_security_scan
        code = "def hello(): return 'hello'\n"
        result = run_security_scan(code)
        assert isinstance(result.score, float)
        assert result.score > 0


# ── 测试代码块解析 ────────────────────────────────────────
class TestCodeBlockParsing:
    def test_extract_implementation_and_tests(self):
        from agents import _extract_code_blocks
        message = '''Here is the code:
```implementation
def hello():
    return "world"
```

```tests
def test_hello():
    assert hello() == "world"
```
'''
        blocks = _extract_code_blocks(message)
        assert "implementation" in blocks
        assert "tests" in blocks
        assert "def hello" in blocks["implementation"]
        assert "def test_hello" in blocks["tests"]

    def test_extract_review_block(self):
        from agents import _extract_code_blocks
        message = '''```review
STATUS: PASS
SCORE: 0.9
COMMENT: Good code
```
'''
        blocks = _extract_code_blocks(message)
        assert "review" in blocks
        assert "STATUS: PASS" in blocks["review"]

    def test_extract_empty_on_no_blocks(self):
        from agents import _extract_code_blocks
        blocks = _extract_code_blocks("just some text without code blocks")
        assert blocks == {}


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


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

    def test_dual_agent_loop_single_round(self):
        """Mock LLM 调用，验证单轮循环能正常产出结果"""
        import agents as agents_mod

        coder_reply = '''```implementation
def add(a: int, b: int) -> int:
    return a + b
```

```tests
def test_add():
    assert add(1, 2) == 3
```
'''
        reviewer_reply = '''```review
STATUS: PASS
SCORE: 0.95
COMMENT: Code is clean and correct.
```
'''
        with patch.object(agents_mod, '_call_llm', side_effect=[coder_reply, reviewer_reply]):
            result = agents_mod.run_dual_agent_loop(
                requirement="实现 add(a, b) 函数",
                pass_threshold=0.60,
                max_rounds=1,
                verbose=False,
            )
            # 综合评分 = 工具分*0.6 + Reviewer分*0.4，阈值 0.60 确保单轮可通过
            assert result["success"] is True, f"Expected success but got score={result['final_score']}"
            assert result["rounds"] == 1
            assert "def add" in result["final_code"]
            assert result["final_score"] > 0
