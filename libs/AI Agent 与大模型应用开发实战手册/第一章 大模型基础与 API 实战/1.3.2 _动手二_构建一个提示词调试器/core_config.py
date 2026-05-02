"""全局配置：模型注册表与定价信息"""
from typing import TypedDict


class ModelConfig(TypedDict):
    litellm_id: str       # LiteLLM 识别的模型字符串
    price_in: float       # 每 1K input tokens 价格（美元）
    price_out: float      # 每 1K output tokens 价格（美元）
    max_tokens_limit: int # 模型支持的最大 max_tokens


# 注册表：key 是界面显示名，value 是调用配置
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o": {
        "litellm_id": "openai/gpt-4o",
        "price_in": 0.0025,
        "price_out": 0.01,
        "max_tokens_limit": 4096,
    },
    "claude-3.5-sonnet": {
        "litellm_id": "anthropic/claude-3-5-sonnet-20241022",
        "price_in": 0.003,
        "price_out": 0.015,
        "max_tokens_limit": 8192,
    },
    "deepseek-v3": {
        "litellm_id": "deepseek/deepseek-chat",
        "price_in": 0.00027,
        "price_out": 0.0011,
        "max_tokens_limit": 4096,
    },
}


def estimate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """根据 Token 数估算调用费用（美元）"""
    cfg = MODEL_REGISTRY[model_key]
    return (
        input_tokens / 1000 * cfg["price_in"]
        + output_tokens / 1000 * cfg["price_out"]
    )