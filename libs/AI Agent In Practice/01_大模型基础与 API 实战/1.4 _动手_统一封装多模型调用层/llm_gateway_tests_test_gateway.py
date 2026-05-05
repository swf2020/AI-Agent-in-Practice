"""
单元测试：Mock LiteLLM API，不产生真实 API 调用。
运行：pytest llm_gateway/tests/ -v
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from litellm import ModelResponse
from litellm.utils import Usage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_gateway_gateway import LLMGateway, LLMResponse
from llm_gateway_cost_tracker import CostTracker


def make_mock_response(
    content: str = "这是模拟回答",
    model: str = "gpt-4o-2024-08-06",
    prompt_tokens: int = 20,
    completion_tokens: int = 50,
) -> ModelResponse:
    """
    构造一个结构与真实 LiteLLM 响应一致的 Mock 对象。
    直接 MagicMock() 会导致 cost 计算失败，这里精确构造所需字段。
    """
    response = MagicMock(spec=ModelResponse)
    response.model = model
    response.choices = [
        MagicMock(message=MagicMock(content=content))
    ]
    response.usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return response


class TestLLMGateway:
    """Gateway 核心功能测试"""

    @pytest.fixture
    def gateway(self):
        """每个测试用例共享一个 Gateway 实例"""
        return LLMGateway()

    @pytest.mark.asyncio
    async def test_chat_returns_llm_response(self, gateway):
        """chat() 应返回正确结构的 LLMResponse"""
        mock_resp = make_mock_response(content="北京是中国首都")

        # patch Router.acompletion，使其返回我们构造的 mock 响应
        # patch 路径要指向 gateway.py 中实际导入的对象
        with patch.object(gateway.router, "acompletion", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_resp

            # 同时 patch litellm.completion_cost，避免依赖真实定价接口
            with patch("litellm.completion_cost", return_value=0.000125):
                result = await gateway.chat("中国首都是哪里？", feature="test")

        assert isinstance(result, LLMResponse)
        assert result.content == "北京是中国首都"
        assert result.prompt_tokens == 20
        assert result.completion_tokens == 50
        assert result.cost_usd == 0.000125

    @pytest.mark.asyncio
    async def test_chat_batch_returns_ordered_results(self, gateway):
        """批量调用结果顺序应与输入 prompts 顺序一致"""
        prompts = ["问题A", "问题B", "问题C"]
        responses = [
            make_mock_response(content=f"回答{c}") for c in ["A", "B", "C"]
        ]

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            r = responses[call_count]
            call_count += 1
            return r

        with patch.object(gateway.router, "acompletion", side_effect=side_effect):
            with patch("litellm.completion_cost", return_value=0.0001):
                results = await gateway.chat_batch(
                    prompts, max_concurrent=3, feature="test_batch"
                )

        assert len(results) == 3
        # 验证顺序：并发执行但结果必须保序
        assert results[0].content == "回答A"
        assert results[1].content == "回答B"
        assert results[2].content == "回答C"

    @pytest.mark.asyncio
    async def test_chat_batch_handles_partial_failure(self, gateway):
        """批量调用中单个失败不应导致整批崩溃"""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # 第 2 个请求模拟超时
                raise TimeoutError("模拟请求超时")
            return make_mock_response(content=f"成功回答{call_count}")

        with patch.object(gateway.router, "acompletion", side_effect=side_effect):
            with patch("litellm.completion_cost", return_value=0.0001):
                results = await gateway.chat_batch(
                    ["p1", "p2", "p3"], max_concurrent=3
                )

        assert len(results) == 3
        assert "成功" in results[0].content
        assert "[ERROR]" in results[1].content  # 失败的变成错误占位
        assert "成功" in results[2].content

    @pytest.mark.asyncio
    async def test_cost_tracker_accumulates_correctly(self, gateway):
        """多次调用后 cost_report 应正确累计"""
        mock_resp = make_mock_response(prompt_tokens=10, completion_tokens=20)

        with patch.object(gateway.router, "acompletion", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_resp
            with patch("litellm.completion_cost", return_value=0.0005):
                await gateway.chat("请求1", feature="feature_x")
                await gateway.chat("请求2", feature="feature_x")
                await gateway.chat("请求3", feature="feature_y")

        report = gateway.cost_report()

        # feature_x 调用了两次
        model_key = list(report["feature_x"].keys())[0]
        assert report["feature_x"][model_key]["prompt_tokens"] == 20   # 10 * 2
        assert report["feature_x"][model_key]["completion_tokens"] == 40  # 20 * 2

        # feature_y 调用了一次
        assert "feature_y" in report

        # 总成本
        total = gateway.tracker.total_cost()
        assert abs(total - 0.0015) < 1e-9  # 0.0005 * 3


class TestCostTracker:
    """CostTracker 独立单元测试"""

    def test_reset_clears_all_data(self):
        tracker = CostTracker()
        mock_resp = make_mock_response()

        with patch("litellm.completion_cost", return_value=0.001):
            tracker.record(mock_resp, feature="test")

        assert tracker.total_cost() > 0
        tracker.reset()
        assert tracker.total_cost() == 0.0
        assert tracker.report() == {}