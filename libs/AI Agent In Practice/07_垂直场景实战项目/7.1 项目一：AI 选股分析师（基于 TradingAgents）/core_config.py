"""全局配置：模型注册表与定价信息

本模块同时提供各实验模块共用的工具函数，包括：
- create_default_config(): 创建 TradingAgentsConfig 的默认实例
- normalize_decision():  统一处理 TradingAgents 新旧版本 decision 返回值差异
"""
import os
from typing import Any, TypedDict

from tradingagents.config import TradingAgentsConfig


class ModelConfig(TypedDict):
    litellm_id: str          # LiteLLM 识别的模型字符串
    price_in: float          # 每 1K input tokens 价格（美元）
    price_out: float         # 每 1K output tokens 价格（美元）
    max_tokens_limit: int    # 模型支持的最大 max_tokens
    api_key_env: str | None  # API Key 环境变量名
    base_url: str | None     # API 基础 URL（None 表示使用默认）


# 注册表：key 是界面显示名，value 是调用配置
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "DeepSeek-V3": {
        "litellm_id": "deepseek/deepseek-chat",
        "price_in": 0.00027,
        "price_out": 0.0011,
        "max_tokens_limit": 4096,
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "Qwen-Max": {
        "litellm_id": "qwen/qwen-plus",
        "price_in": 0.001,
        "price_out": 0.004,
        "max_tokens_limit": 4096,
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "GPT-4o": {
        "litellm_id": "openai/gpt-4o",
        "price_in": 0.005,
        "price_out": 0.015,
        "max_tokens_limit": 4096,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
    "GPT-4o-Mini": {
        "litellm_id": "openai/gpt-4o-mini",
        "price_in": 0.00015,
        "price_out": 0.0006,
        "max_tokens_limit": 4096,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
}

# ✅ 当前激活模型 key — 修改此处全局生效，必须是 MODEL_REGISTRY 中的 key
ACTIVE_MODEL_KEY: str = "DeepSeek-V3"


def get_active_config() -> ModelConfig:
    """获取当前激活模型的完整配置"""
    return MODEL_REGISTRY[ACTIVE_MODEL_KEY]


def get_litellm_id(model_key: str | None = None) -> str:
    """获取指定模型（默认激活模型）的 LiteLLM ID"""
    key = model_key or ACTIVE_MODEL_KEY
    return MODEL_REGISTRY[key]["litellm_id"]


def get_api_key(model_key: str | None = None) -> str | None:
    """从环境变量读取指定模型的 API Key"""
    key = model_key or ACTIVE_MODEL_KEY
    env_var = MODEL_REGISTRY[key]["api_key_env"]
    return os.environ.get(env_var) if env_var else None


def get_base_url(model_key: str | None = None) -> str | None:
    """获取指定模型的 base_url（None 表示使用 SDK 默认值）"""
    key = model_key or ACTIVE_MODEL_KEY
    return MODEL_REGISTRY[key]["base_url"]


def get_model_list() -> list[str]:
    """获取所有已注册模型的显示名列表"""
    return list(MODEL_REGISTRY.keys())


# 定价单位：模型官方价格通常以"每 1K tokens"标示
# LiteLLM 的 cost tracker 也以此为单位，此处保持对齐
_TOKENS_PER_PRICE_UNIT = 1000

# TradingAgents 新版本 decision 返回字符串（不含置信度等字段）时使用的占位值
# 设为 None 明确表示"此值不代表真实分析置信度" [Fix #3]
UNKNOWN_CONFIDENCE = None


def estimate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """根据 Token 数估算调用费用（美元）"""
    cfg = MODEL_REGISTRY[model_key]
    return (
        input_tokens / _TOKENS_PER_PRICE_UNIT * cfg["price_in"]
        + output_tokens / _TOKENS_PER_PRICE_UNIT * cfg["price_out"]
    )


def create_default_config(reasoning_effort: str = "medium") -> TradingAgentsConfig:
    """
    创建各实验模块共用的默认 TradingAgentsConfig。 [Fix #4]

    统一收敛了原 experiment_1/2/3 中重复的 _make_config() 逻辑，
    各实验按需覆盖 reasoning_effort 即可（experiment_1 的风险偏好映射）。

    Args:
        reasoning_effort: 推理力度，支持 "low" / "medium" / "high"
    """
    return TradingAgentsConfig(
        llm_provider="litellm",
        deep_think_llm="deepseek/deepseek-chat",
        quick_think_llm="deepseek/deepseek-chat",
        reasoning_effort=reasoning_effort,
        max_debate_rounds=3,
        max_risk_discuss_rounds=3,
        max_recur_limit=100,
    )


def normalize_decision(decision: Any) -> dict[str, Any]:
    """
    统一处理 TradingAgents 新旧版本的 decision 返回值差异。 [Fix #3]

    新版本（>=0.3.x）返回纯字符串，旧版本返回包含 action/reasoning/confidence
    等字段的 dict。本函数确保下游代码始终拿到 dict 格式。

    Args:
        decision: graph.propagate() 返回的 decision 值

    Returns:
        标准化 decision dict，至少包含 action / reasoning / confidence 三个键
    """
    if isinstance(decision, str):
        return {
            "action": decision.lower(),
            "reasoning": (
                "（TradingAgents 新版本返回纯字符串评级，未包含结构化推理文本。"
                "建议检查框架版本或升级至支持 Structured Output 的版本。）"
            ),
            "confidence": UNKNOWN_CONFIDENCE,
        }
    return decision
