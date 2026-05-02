import os
import dspy
from dotenv import load_dotenv

load_dotenv()  # 从 .env 读取 OPENAI_API_KEY

def configure_lm(model: str = "openai/gpt-4o-mini") -> dspy.LM:
    """
    初始化 LLM 后端。
    
    DSPy 使用 LiteLLM 格式的 model string：
      - "openai/gpt-4o-mini"     → OpenAI
      - "anthropic/claude-3-5-haiku-20241022" → Claude（需设置 ANTHROPIC_API_KEY）
      - "ollama/qwen2.5:7b"      → 本地 Ollama（无需 API Key）
    
    temperature=0 在优化阶段保证确定性，便于对比实验。
    """
    lm = dspy.LM(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,        # 优化阶段固定为0，推理阶段可调
        max_tokens=512,
        cache=True,           # 开启本地缓存，重跑时不重复花钱
    )
    dspy.configure(lm=lm)
    print(f"✅ LLM 配置完成：{model}")
    return lm

if __name__ == "__main__":
    lm = configure_lm()
    # 快速验证连通性
    response = lm("用一句话介绍 DSPy")
    print(response)