"""
模型配置：定义模型列表与 Fallback 策略。
将配置与代码解耦，方便在不改业务代码的情况下切换模型。
"""
from typing import Any

# LiteLLM Router 接受的模型配置格式
# model_name 是业务层使用的"别名"，litellm_params.model 才是实际调用的模型标识
MODEL_LIST: list[dict[str, Any]] = [
    {
        "model_name": "gpt-4o",           # 业务层调用名
        "litellm_params": {
            "model": "openai/gpt-4o",      # LiteLLM 标识：provider/model
            "api_key": "os.environ/OPENAI_API_KEY",   # 从环境变量读取
        },
        "model_info": {"id": "openai-gpt4o-primary"},
    },
    {
        "model_name": "claude-sonnet",
        "litellm_params": {
            "model": "anthropic/claude-sonnet-4-20250514",
            "api_key": "os.environ/ANTHROPIC_API_KEY",
        },
        "model_info": {"id": "anthropic-sonnet-primary"},
    },
    {
        "model_name": "deepseek-chat",
        "litellm_params": {
            "model": "deepseek/deepseek-chat",
            "api_key": "os.environ/DEEPSEEK_API_KEY",
        },
        "model_info": {"id": "deepseek-primary"},
    },
    {  # [Fix #2] 补充 Qwen-Max 配置，与 core_config.MODEL_REGISTRY 保持同步
        "model_name": "qwen-max",
        "litellm_params": {
            "model": "qwen/qwen-plus",
            "api_key": "os.environ/DASHSCOPE_API_KEY",
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "model_info": {"id": "qwen-max-primary"},
    },
    {  # [Fix #3] 补充 Gemini 配置，与教学文档保持一致性
        "model_name": "gemini-flash",
        "litellm_params": {
            "model": "gemini/gemini-2.5-flash",
            "api_key": "os.environ/GEMINI_API_KEY",
        },
        "model_info": {"id": "gemini-flash-primary"},
    },
]

# Fallback 策略：gpt-4o 失败时，依次尝试 claude-sonnet、deepseek-chat
# 这里用的是"别名"，而非具体模型 ID
FALLBACKS: list[dict[str, list[str]]] = [
    {"gpt-4o": ["claude-sonnet", "deepseek-chat"]},
    {"claude-sonnet": ["gpt-4o", "deepseek-chat"]},
]