"""
LLM 并发调用层

设计原则：
  1. 每个 call_single 是独立协程，互不干扰
  2. call_all 用 asyncio.gather 并发，总耗时 ≈ 最慢模型的单次耗时
  3. return_exceptions=True 确保一个模型出错不影响其他模型的结果
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from litellm import acompletion

from core.config import MODEL_REGISTRY, estimate_cost


@dataclass
class CallResult:
    """单次模型调用的结构化结果"""
    model: str
    output: str
    latency: float        # 秒
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float # 美元
    error: str | None = None  # 非 None 表示调用失败


async def call_single(
    model_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> CallResult:
    """
    调用单个模型并返回结构化结果。

    Args:
        model_key: MODEL_REGISTRY 中的键名（如 "gpt-4o"）
        system_prompt: 系统提示词
        user_prompt: 用户输入
        temperature: 采样温度 [0, 2]
        max_tokens: 最大输出 Token 数

    Returns:
        CallResult 对象，error 字段非 None 表示失败
    """
    cfg = MODEL_REGISTRY[model_key]
    start = time.perf_counter()

    try:
        resp = await acompletion(
            model=cfg["litellm_id"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = round(time.perf_counter() - start, 2)
        usage = resp.usage
        input_tok = usage.prompt_tokens
        output_tok = usage.completion_tokens

        return CallResult(
            model=model_key,
            output=resp.choices[0].message.content or "",
            latency=latency,
            input_tokens=input_tok,
            output_tokens=output_tok,
            total_tokens=usage.total_tokens,
            estimated_cost=round(estimate_cost(model_key, input_tok, output_tok), 6),
        )

    except Exception as exc:
        # 捕获所有异常（限流、超时、Key 失效等），不让单个失败拖垮整批
        latency = round(time.perf_counter() - start, 2)
        return CallResult(
            model=model_key,
            output="",
            latency=latency,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost=0.0,
            error=_friendly_error(exc),
        )


def _friendly_error(exc: Exception) -> str:
    """将技术性异常转换为对用户友好的提示"""
    msg = str(exc).lower()
    if "rate limit" in msg or "429" in msg:
        return "❌ API 限流，请稍后重试（建议降低并发频率）"
    if "auth" in msg or "401" in msg or "invalid api key" in msg:
        return "❌ API Key 无效，请检查 .env 配置"
    if "timeout" in msg:
        return "❌ 请求超时，模型响应过慢"
    if "model_not_found" in msg or "404" in msg:
        return "❌ 模型不可用，请检查模型名称或账户权限"
    return f"❌ 调用失败：{exc.__class__.__name__}: {str(exc)[:100]}"


async def call_all(
    selected_models: list[str],
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> list[CallResult]:
    """
    并发调用所有选中的模型。

    关键设计：asyncio.gather 让所有请求同时发出，
    总耗时 ≈ max(各模型耗时)，而非 sum(各模型耗时)。
    例如：GPT-4o 需 3s，Claude 需 4s，DeepSeek 需 2s
          串行总计 9s，并发只需约 4s。
    """
    tasks = [
        call_single(m, system_prompt, user_prompt, temperature, max_tokens)
        for m in selected_models
    ]
    # return_exceptions=False 已被上层 try/except 处理，此处无需重复兜底
    results: list[CallResult] = await asyncio.gather(*tasks)
    return results