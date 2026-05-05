"""全局配置：模型注册表、Embedding、Reranker 与定价信息"""
import os
from typing import TypedDict


# ── 聊天模型注册表 ──────────────────────────────────────────────────
class ModelConfig(TypedDict, total=False):
    litellm_id: str          # LiteLLM SDK 使用的模型 ID（含 provider 前缀）
    chat_model_id: str       # OpenAI SDK 直连时使用的模型名（无前缀）
    price_in: float          # 每 1K input tokens 价格（美元）
    price_out: float         # 每 1K output tokens 价格（美元）
    max_tokens_limit: int    # 模型支持的最大 max_tokens
    api_key_env: str | None  # API Key 环境变量名
    base_url: str | None     # API 基础 URL（None 表示使用默认）


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
    "OpenAI-GPT-4o-mini": {
        "litellm_id": "gpt-4o-mini",
        "chat_model_id": "gpt-4o-mini",
        "price_in": 0.00015,
        "price_out": 0.0006,
        "max_tokens_limit": 16384,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
}

# ✅ 当前激活聊天模型 key — 修改此处全局生效
ACTIVE_MODEL_KEY: str = "DeepSeek-V3"


# ── Embedding 模型配置（DashScope）─────────────────────────────────
EMBEDDING_MODEL: str = "text-embedding-v4"
EMBEDDING_DIM: int = 1024          # text-embedding-v4 维度
EMBEDDING_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ── Reranker 模型配置（DashScope OpenAI 兼容接口）─────────────────
RERANKER_MODEL: str = "qwen3-rerank"
RERANKER_TOP_N: int = 5            # 默认精排后保留的数量
RERANKER_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ── 聊天模型辅助函数 ────────────────────────────────────────────────
def get_active_config() -> ModelConfig:
    """获取当前激活模型的完整配置"""
    return MODEL_REGISTRY[ACTIVE_MODEL_KEY]


def get_litellm_id(model_key: str | None = None) -> str:
    """获取指定模型的 LiteLLM SDK ID（含 provider 前缀，如 deepseek/deepseek-chat）"""
    key = model_key or ACTIVE_MODEL_KEY
    return MODEL_REGISTRY[key]["litellm_id"]


def get_chat_model_id(model_key: str | None = None) -> str:
    """获取 OpenAI SDK 直连时使用的模型名（无前缀，如 deepseek-v4-flash）"""
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


# ── DashScope 辅助函数（Embedding + Reranker 共用）─────────────────
def get_dashscope_api_key() -> str | None:
    """获取 DashScope API Key（Embedding 和 Reranker 共用）"""
    return os.environ.get("DASHSCOPE_API_KEY")


def get_embedding_model() -> str:
    """获取 Embedding 模型名称"""
    return EMBEDDING_MODEL


def get_embedding_dim() -> int:
    """获取 Embedding 维度"""
    return EMBEDDING_DIM


def get_embedding_base_url() -> str:
    """获取 Embedding 模型 base_url"""
    return EMBEDDING_BASE_URL


def get_reranker_model() -> str:
    """获取 Reranker 模型名称"""
    return RERANKER_MODEL


def get_reranker_top_n() -> int:
    """获取 Reranker 默认保留数量"""
    return RERANKER_TOP_N


def get_reranker_base_url() -> str:
    """获取 Reranker 模型 base_url"""
    return RERANKER_BASE_URL
