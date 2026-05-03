"""端到端冒烟测试，确认整个调用链路正常。需要真实 API Key。"""
import asyncio
import json
import sys
import os

# 将当前目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from llm_gateway_gateway import LLMGateway

load_dotenv()


async def main():
    gateway = LLMGateway()

    print("=== 单次调用测试 ===")
    resp = await gateway.chat(
        prompt="用一句话解释什么是 Transformer",
        model="deepseek-chat",  # 使用 DeepSeek 模型
        system="你是一位 AI 教育专家，用最通俗的语言解释技术概念。",
        feature="demo_single",
    )
    print(f"模型：{resp.model}")
    print(f"回答：{resp.content}")
    print(f"Token：{resp.prompt_tokens}p + {resp.completion_tokens}c")
    print(f"成本：${resp.cost_usd}")

    print("\n=== 批量并发调用测试（3 个 prompt）===")
    prompts = [
        "什么是向量数据库？",
        "RAG 和微调的区别是什么？",
        "什么是 Function Calling？",
    ]
    results = await gateway.chat_batch(
        prompts, model="deepseek-chat", max_concurrent=3, feature="demo_batch"
    )
    for i, r in enumerate(results):
        print(f"[{i}] {prompts[i][:15]}... → {r.content[:40]}...")

    print("\n=== 成本报告 ===")
    print(json.dumps(gateway.cost_report(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())