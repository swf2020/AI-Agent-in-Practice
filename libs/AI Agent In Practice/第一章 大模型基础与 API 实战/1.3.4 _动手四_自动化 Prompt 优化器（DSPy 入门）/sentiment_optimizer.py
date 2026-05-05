import os
import dspy
from dotenv import load_dotenv
from core_config import MODEL_REGISTRY, get_litellm_id, get_api_key, get_base_url, ACTIVE_MODEL_KEY

load_dotenv()  # 从 .env 读取环境变量

def configure_lm(model_key: str | None = None) -> dspy.LM:
    """
    初始化 LLM 后端。

    支持的模型：
      - "Qwen-Max"      → Qwen Plus（需设置 DASHSCOPE_API_KEY）
      - "DeepSeek-V3"   → DeepSeek V3（需设置 DEEPSEEK_API_KEY）

    temperature=0 在优化阶段保证确定性，便于对比实验。
    """
    if model_key is None:
        model_key = ACTIVE_MODEL_KEY

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_key}。可用模型: {list(MODEL_REGISTRY.keys())}")

    cfg = MODEL_REGISTRY[model_key]

    # 对于 Qwen 模型，使用 OpenAI 兼容模式
    if model_key == "Qwen-Max":
        api_key_env = cfg.get("api_key_env")
        base_url = cfg.get("base_url")

        api_key = os.getenv(api_key_env) if api_key_env else None

        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = api_key or ""
        os.environ["OPENAI_API_BASE"] = base_url or ""

        # 使用标准的 gpt-4o-mini 模型名，通过环境变量路由到 Qwen
        lm = dspy.LM(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=512,
            cache=True,
        )
        dspy.configure(lm=lm)
        print(f"✅ LLM 配置完成：{model_key} (通过 OpenAI 兼容模式)")
        return lm

    # 对于其他模型（如 DeepSeek），统一使用 core_config 辅助函数
    kwargs = {
        "model": get_litellm_id(model_key),
        "temperature": 0,
        "max_tokens": 512,
        "cache": True,
    }

    api_key = get_api_key(model_key)
    if api_key:
        kwargs["api_key"] = api_key
    base_url = get_base_url(model_key)
    if base_url:
        kwargs["base_url"] = base_url

    lm = dspy.LM(**kwargs)
    dspy.configure(lm=lm)
    print(f"✅ LLM 配置完成：{model_key} ({cfg['litellm_id']})")
    return lm

if __name__ == "__main__":
    lm = configure_lm()
    # 快速验证连通性
    response = lm("用一句话介绍 DSPy")
    print(response)
