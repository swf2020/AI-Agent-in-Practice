"""全局配置：模型注册表"""
MODEL_REGISTRY = {
    "DeepSeek-V3": {
        "litellm_id": "deepseek/deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": None,
    },
    "Qwen-Max": {
        "litellm_id": "openai/qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
}

def get_model_list():
    return list(MODEL_REGISTRY.keys())
