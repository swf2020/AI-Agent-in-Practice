import os
import dspy
from dotenv import load_dotenv
from core_config import MODEL_REGISTRY

load_dotenv()  # 从 .env 读取环境变量

def configure_lm(model_key: str = "Qwen-Max") -> dspy.LM:
    """
    初始化 LLM 后端。
    
    支持的模型：
      - "Qwen-Max"      → Qwen Plus（需设置 DASHSCOPE_API_KEY）
      - "DeepSeek-V3"   → DeepSeek V3（需设置 DEEPSEEK_API_KEY）
    
    temperature=0 在优化阶段保证确定性，便于对比实验。
    """
    # 获取模型配置
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_key}。可用模型: {list(MODEL_REGISTRY.keys())}")
    
    cfg = MODEL_REGISTRY[model_key]
    
    # 构建参数字典
    kwargs = {
        "model": cfg["litellm_id"],
        "temperature": 0,
        "max_tokens": 512,
        "cache": True,
    }
    
    # 添加可选参数
    if cfg.get("api_key_env"):
        kwargs["api_key"] = os.getenv(cfg["api_key_env"])
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    
    lm = dspy.LM(**kwargs)
    dspy.configure(lm=lm)
    print(f"✅ LLM 配置完成：{model_key} ({cfg['litellm_id']})")
    return lm

if __name__ == "__main__":
    lm = configure_lm()
    # 快速验证连通性
    response = lm("用一句话介绍 DSPy")
    print(response)