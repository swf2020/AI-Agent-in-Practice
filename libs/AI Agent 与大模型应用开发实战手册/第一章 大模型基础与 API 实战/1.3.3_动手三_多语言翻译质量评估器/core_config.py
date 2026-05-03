"""全局配置：模型注册表与定价信息"""
import os
from typing import TypedDict


class ModelConfig(TypedDict):
    litellm_id: str       # LiteLLM 识别的模型字符串
    price_in: float       # 每 1K input tokens 价格（美元）
    price_out: float      # 每 1K output tokens 价格（美元）
    max_tokens_limit: int # 模型支持的最大 max_tokens
    api_key_env: str | None  # API Key 环境变量名（None 表示使用默认）
    base_url: str | None     # API Base URL（None 表示使用默认）


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "DeepSeek-V3": {
        "litellm_id": "deepseek/deepseek-chat",
        "price_in": 0.00027,
        "price_out": 0.0011,
        "max_tokens_limit": 4096,
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": None,
    },
    "Qwen-Max": {
        "litellm_id": "openai/qwen-plus",
        "price_in": 0.001,
        "price_out": 0.004,
        "max_tokens_limit": 4096,
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
}


def estimate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """根据 Token 数估算调用费用（美元）"""
    cfg = MODEL_REGISTRY[model_key]
    return (
        input_tokens / 1000 * cfg["price_in"]
        + output_tokens / 1000 * cfg["price_out"]
    )


def get_model_list() -> list[str]:
    """获取所有可用模型列表"""
    return list(MODEL_REGISTRY.keys())
