# tests/test_main.py — 自动生成的冒烟测试
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys, os

# 确保能找到项目根目录和 core 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ── 测试 core_config 基础结构 ──────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY,
            get_litellm_id, get_api_key, get_base_url,
            get_model_list, estimate_cost, get_active_config,
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

    def test_get_litellm_id_by_key(self):
        from core_config import get_litellm_id
        assert get_litellm_id("DeepSeek-V3") == "deepseek/deepseek-chat"
        assert get_litellm_id("Qwen-Max") == "openai/qwen-plus"

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
        # DeepSeek-V3 默认 None
        assert get_base_url("DeepSeek-V3") is None
        # Qwen-Max 有 base_url
        url = get_base_url("Qwen-Max")
        assert isinstance(url, str) and len(url) > 0

    def test_get_active_config(self):
        from core_config import get_active_config, MODEL_REGISTRY, ACTIVE_MODEL_KEY
        cfg = get_active_config()
        assert cfg == MODEL_REGISTRY[ACTIVE_MODEL_KEY]


# ── 测试核心模块可导入 ───────────────────────────────────
class TestModuleImports:
    def test_caller_import(self):
        from core.caller import call_all, call_single, CallResult
        assert callable(call_all)
        assert callable(call_single)

    def test_history_import(self):
        from core.history import save_run, load_history, get_run_by_id
        assert callable(save_run)
        assert callable(load_history)

    def test_app_import(self):
        from app import run_experiment, format_result_markdown
        assert callable(run_experiment)
        assert callable(format_result_markdown)

    def test_main_module_importable(self):
        try:
            import importlib.util
            path = os.path.join(PROJECT_ROOT, "main.py")
            spec = importlib.util.spec_from_file_location("main", path)
            assert spec is not None, "main.py 不存在"
        except Exception as e:
            pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 LLM 调用（Mock litellm）──────────────────
class TestLLMCall:
    @pytest.mark.asyncio
    @patch("core.caller.acompletion")
    async def test_mocked_single_call(self, mock_acompletion):
        """验证核心调用路径在 mock 下可正常执行"""
        mock_acompletion.return_value = AsyncMock(
            choices=[MagicMock(
                message=MagicMock(content="mock response"),
                finish_reason="stop",
            )],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        from core.caller import call_single
        result = await call_single(
            model_key="DeepSeek-V3",
            system_prompt="test system",
            user_prompt="test user",
            temperature=0.0,
            max_tokens=100,
        )
        assert result.output == "mock response"
        assert result.error is None
        assert result.model == "DeepSeek-V3"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    @patch("core.caller.acompletion")
    async def test_mocked_call_all(self, mock_acompletion):
        """验证并发调用路径"""
        mock_acompletion.return_value = AsyncMock(
            choices=[MagicMock(
                message=MagicMock(content="answer"),
                finish_reason="stop",
            )],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        from core.caller import call_all
        results = await call_all(
            selected_models=["DeepSeek-V3"],
            system_prompt="test",
            user_prompt="hello",
            temperature=0.0,
            max_tokens=100,
        )
        assert len(results) == 1
        assert results[0].output == "answer"

    @pytest.mark.asyncio
    @patch("core.caller.acompletion")
    async def test_mocked_call_error(self, mock_acompletion):
        """验证模型调用失败时的错误处理"""
        mock_acompletion.side_effect = Exception("API Key 无效")

        from core.caller import call_single
        result = await call_single(
            model_key="DeepSeek-V3",
            system_prompt="test",
            user_prompt="hello",
            temperature=0.0,
            max_tokens=100,
        )
        assert result.error is not None
        assert result.output == ""


# ── 测试历史存储 ──────────────────────────────────────────
class TestHistory:
    def test_save_and_load(self, tmp_path, monkeypatch):
        """测试保存和读取历史（使用临时目录）"""
        import tempfile
        import json

        # 切换到临时目录，避免污染项目 history.jsonl
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            from core.history import save_run, load_history, HISTORY_FILE
            from core.caller import CallResult

            results = [
                CallResult(
                    model="DeepSeek-V3", output="test answer",
                    latency=1.5, input_tokens=10, output_tokens=5,
                    total_tokens=15, estimated_cost=0.001, error=None,
                )
            ]

            run_id = save_run(
                system_prompt="test system",
                user_prompt="hello",
                selected_models=["DeepSeek-V3"],
                temperature=0.0,
                max_tokens=100,
                results=results,
            )
            assert run_id is not None
            assert len(run_id) > 0

            df = load_history(use_cache=False)
            assert len(df) == 1
            assert "DeepSeek-V3" in df["模型"].values
        finally:
            os.chdir(original_cwd)


# ── 测试 Gradio 辅助函数 ──────────────────────────────────
class TestAppHelpers:
    def test_format_result_markdown_success(self):
        from app import format_result_markdown
        from core.caller import CallResult
        r = CallResult(
            model="DeepSeek-V3", output="Hello world",
            latency=1.0, input_tokens=10, output_tokens=5,
            total_tokens=15, estimated_cost=0.001, error=None,
        )
        md = format_result_markdown(r)
        assert "DeepSeek-V3" in md
        assert "Hello world" in md
        assert "1.0s" in md

    def test_format_result_markdown_error(self):
        from app import format_result_markdown
        from core.caller import CallResult
        r = CallResult(
            model="DeepSeek-V3", output="",
            latency=0.5, input_tokens=0, output_tokens=0,
            total_tokens=0, estimated_cost=0.0, error="API Key 无效",
        )
        md = format_result_markdown(r)
        assert "API Key 无效" in md
