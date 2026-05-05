"""全局配置：模型注册表与定价信息"""
import os
from typing import TypedDict


class ModelConfig(TypedDict, total=False):
    litellm_id: str          # LiteLLM 识别的模型字符串（含 provider 前缀）
    chat_model_id: str       # OpenAI/Anthropic SDK 直连时使用的模型名（无前缀）
    price_in: float          # 每 1K input tokens 价格（美元）
    price_out: float         # 每 1K output tokens 价格（美元）
    max_tokens_limit: int    # 模型支持的最大 max_tokens
    api_key_env: str | None  # API Key 环境变量名
    base_url: str | None     # API 基础 URL（None 表示使用默认）


# 注册表：key 是界面显示名，value 是调用配置
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "DeepSeek-V3": {
        "litellm_id": "deepseek/deepseek-chat",
        "chat_model_id": "deepseek-v4-flash",
        "price_in": 0.00027,
        "price_out": 0.0011,
        "max_tokens_limit": 4096,
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "Qwen-Max": {
        "litellm_id": "qwen/qwen-plus",
        "chat_model_id": "qwen-plus",
        "price_in": 0.001,
        "price_out": 0.004,
        "max_tokens_limit": 4096,
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    # 以下模型用于 LangChain 直连（不通过 LiteLLM），保留显示名供切换参考
    "Claude-Sonnet": {
        "litellm_id": "claude-sonnet-4-5",
        "chat_model_id": "claude-sonnet-4-5",
        "price_in": 0.003,
        "price_out": 0.015,
        "max_tokens_limit": 8192,
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
    "GPT-4o-Mini": {
        "litellm_id": "gpt-4o-mini",
        "chat_model_id": "gpt-4o-mini",
        "price_in": 0.00015,
        "price_out": 0.0006,
        "max_tokens_limit": 16384,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
}

# 当前激活模型 key — 修改此处全局切换
# 默认使用 LiteLLM 路由的 DeepSeek-V3，
# agent.py 中 create_llm(provider="default") 即使用此模型
ACTIVE_MODEL_KEY: str = "DeepSeek-V3"


def get_active_config() -> ModelConfig:
    """获取当前激活模型的完整配置"""
    return MODEL_REGISTRY[ACTIVE_MODEL_KEY]


def get_litellm_id(model_key: str | None = None) -> str:
    """获取指定模型的 LiteLLM SDK ID（含 provider 前缀，如 deepseek/deepseek-chat）"""
    key = model_key or ACTIVE_MODEL_KEY
    return MODEL_REGISTRY[key]["litellm_id"]


def get_chat_model_id(model_key: str | None = None) -> str:
    """获取 OpenAI/Anthropic SDK 直连时使用的模型名（无前缀，如 deepseek-v4-flash）"""
    key = model_key or ACTIVE_MODEL_KEY
    cfg = MODEL_REGISTRY[key]
    return cfg.get("chat_model_id", cfg["litellm_id"].split("/")[-1])


def get_api_key(model_key: str | None = None) -> str | None:
    """从环境变量读取指定模型的 API Key"""
    key = model_key or ACTIVE_MODEL_KEY
    env_var = MODEL_REGISTRY[key].get("api_key_env")
    return os.environ.get(env_var) if env_var else None


def get_base_url(model_key: str | None = None) -> str | None:
    """获取指定模型的 base_url（None 表示使用 SDK 默认值）"""
    key = model_key or ACTIVE_MODEL_KEY
    return MODEL_REGISTRY[key].get("base_url")


def get_model_list() -> list[str]:
    """获取所有已注册模型的显示名列表"""
    return list(MODEL_REGISTRY.keys())


def estimate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """根据 Token 数估算调用费用（美元）"""
    cfg = MODEL_REGISTRY[model_key]
    return (
        input_tokens / 1000 * cfg.get("price_in", 0)
        + output_tokens / 1000 * cfg.get("price_out", 0)
    )
