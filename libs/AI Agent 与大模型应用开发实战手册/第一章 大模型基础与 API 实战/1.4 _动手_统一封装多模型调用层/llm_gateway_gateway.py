"""
LLMGateway：统一多模型调用入口。
对业务层屏蔽底层模型切换、重试、Fallback 细节。
"""
import os
import asyncio
from typing import Any
from dotenv import load_dotenv
import litellm
from litellm import Router
from pydantic import BaseModel

from .cost_tracker import CostTracker
from .config.models import MODEL_LIST, FALLBACKS

load_dotenv()

# 关闭 LiteLLM 的 verbose 日志，生产环境用自己的日志体系
litellm.set_verbose = False


class LLMResponse(BaseModel):
    """
    统一响应结构。
    不直接返回 litellm.ModelResponse，原因：
    1. 避免业务代码依赖 litellm 内部类型，便于未来换底层库
    2. Pydantic 模型可直接序列化为 JSON，便于 API 返回
    """
    content: str
    model: str               # 实际使用的模型（可能是 Fallback 后的）
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class LLMGateway:
    """
    多模型统一调用网关。

    示例：
        gateway = LLMGateway()
        resp = await gateway.chat("你好", model="gpt-4o", feature="demo")
        print(resp.content, resp.cost_usd)
    """

    def __init__(
        self,
        model_list: list[dict] | None = None,
        fallbacks: list[dict] | None = None,
        num_retries: int = 2,
        timeout: float = 30.0,
    ) -> None:
        """
        Args:
            model_list:   模型配置列表，默认使用 config/models.py 中的配置
            fallbacks:    Fallback 策略，默认使用预设策略
            num_retries:  同一模型最大重试次数（不含 Fallback 切换）
            timeout:      单次请求超时秒数
        """
        self.router = Router(
            model_list=model_list or MODEL_LIST,
            fallbacks=fallbacks or FALLBACKS,
            num_retries=num_retries,
            timeout=timeout,
            # retry_after：遇到限流（429）时等待的秒数
            retry_after=5,
            # allowed_fails：某模型连续失败多少次后标记为不健康
            allowed_fails=3,
            # cooldown_time：不健康模型的冷却时间（秒），冷却后重新尝试
            cooldown_time=60,
        )
        self.tracker = CostTracker()

    async def chat(
        self,
        prompt: str,
        model: str = "gpt-4o",
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        feature: str = "default",
        **kwargs: Any,
    ) -> LLMResponse:
        """
        单轮对话调用（异步）。

        Args:
            prompt:      用户消息
            model:       模型别名，对应 MODEL_LIST 中的 model_name
            system:      系统提示词，None 则不传
            temperature: 采样温度
            max_tokens:  最大生成 Token 数
            feature:     业务功能标识，用于成本分组统计
            **kwargs:    透传给 litellm 的其他参数（如 response_format）

        Returns:
            LLMResponse 对象
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Router.acompletion 是异步版本，内部自动处理 Fallback
        raw: litellm.ModelResponse = await self.router.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # 记录本次消耗
        self.tracker.record(raw, feature=feature)

        return LLMResponse(
            content=raw.choices[0].message.content or "",
            model=raw.model or model,
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
            cost_usd=round(litellm.completion_cost(completion_response=raw), 6),
        )

    async def chat_batch(
        self,
        prompts: list[str],
        model: str = "gpt-4o",
        system: str | None = None,
        max_concurrent: int = 10,
        feature: str = "default",
        **kwargs: Any,
    ) -> list[LLMResponse]:
        """
        批量并发调用：自动控制并发数，避免触发限流。

        Args:
            prompts:        待处理的 prompt 列表
            max_concurrent: 最大并发数。建议：
                            - OpenAI Tier 1：5–10
                            - OpenAI Tier 3+：20–50
                            - Claude：5–10（更严格的限流策略）
            feature:        成本追踪标识

        Returns:
            与 prompts 等长的 LLMResponse 列表，顺序一致
        """
        # Semaphore 控制并发上限，防止同时发出几百个请求触发 429
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _call_with_limit(p: str) -> LLMResponse:
            async with semaphore:
                return await self.chat(p, model=model, system=system, feature=feature, **kwargs)

        # asyncio.gather 保序：返回结果与 prompts 下标一一对应
        results = await asyncio.gather(
            *[_call_with_limit(p) for p in prompts],
            return_exceptions=True,  # 单个失败不中断整批
        )

        # 将异常转换为带错误信息的占位响应，而非直接抛出
        final: list[LLMResponse] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                # 生产中这里应该打 error log + 上报 metrics
                final.append(LLMResponse(
                    content=f"[ERROR] prompt[{i}] failed: {type(r).__name__}: {r}",
                    model="error",
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost_usd=0.0,
                ))
            else:
                final.append(r)

        return final

    def cost_report(self) -> dict:
        """返回当前 Gateway 实例的成本汇总"""
        return self.tracker.report()