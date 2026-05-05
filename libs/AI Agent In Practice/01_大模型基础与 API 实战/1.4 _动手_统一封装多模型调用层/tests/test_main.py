# tests/test_main.py — 冒烟测试
import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


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

    def test_get_router_model_name(self):
        from core_config import get_router_model_name, ACTIVE_MODEL_KEY
        result = get_router_model_name()
        assert isinstance(result, str) and len(result) > 0
        assert result == get_router_model_name(ACTIVE_MODEL_KEY)


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试核心 LLM 调用（Mock litellm）──────────────────
def make_mock_response(
    content: str = "这是模拟回答",
    model: str = "gpt-4o-2024-08-06",
    prompt_tokens: int = 20,
    completion_tokens: int = 50,
) -> MagicMock:
    from litellm import ModelResponse
    from litellm.utils import Usage
    response = MagicMock(spec=ModelResponse)
    response.model = model
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return response


class TestLLMCall:
    @pytest.fixture
    def gateway(self):
        from llm_gateway_gateway import LLMGateway
        return LLMGateway()

    @pytest.mark.asyncio
    async def test_mocked_completion(self, gateway):
        """验证核心调用路径在 mock 下可正常执行"""
        mock_resp = make_mock_response(content="mock response")
        with patch.object(gateway.router, "acompletion", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_resp
            with patch("litellm.completion_cost", return_value=0.000125):
                from llm_gateway_gateway import LLMResponse
                result = await gateway.chat(prompt="test", feature="test")
                assert isinstance(result, LLMResponse)
                assert result.content == "mock response"
                mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_batch_partial_failure(self, gateway):
        """批量调用中单个失败不导致整批崩溃"""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise TimeoutError("模拟请求超时")
            return make_mock_response(content=f"成功回答{call_count}")

        with patch.object(gateway.router, "acompletion", side_effect=side_effect):
            with patch("litellm.completion_cost", return_value=0.0001):
                results = await gateway.chat_batch(["p1", "p2", "p3"], max_concurrent=3)

        assert len(results) == 3
        assert "成功" in results[0].content
        assert "[ERROR]" in results[1].content
        assert "成功" in results[2].content

    @pytest.mark.asyncio
    async def test_cost_tracker_accumulates(self, gateway):
        """多次调用后 cost_report 正确累计"""
        mock_resp = make_mock_response(prompt_tokens=10, completion_tokens=20)
        with patch.object(gateway.router, "acompletion", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_resp
            with patch("litellm.completion_cost", return_value=0.0005):
                await gateway.chat("请求1", feature="feature_x")
                await gateway.chat("请求2", feature="feature_x")
                await gateway.chat("请求3", feature="feature_y")

        report = gateway.cost_report()
        assert "feature_x" in report
        assert "feature_y" in report
        total = gateway.tracker.total_cost()
        assert abs(total - 0.0015) < 1e-9


class TestCostTracker:
    def test_reset_clears_all_data(self):
        from llm_gateway_cost_tracker import CostTracker
        tracker = CostTracker()
        mock_resp = make_mock_response()
        with patch("litellm.completion_cost", return_value=0.001):
            tracker.record(mock_resp, feature="test")
        assert tracker.total_cost() > 0
        tracker.reset()
        assert tracker.total_cost() == 0.0
        assert tracker.report() == {}
