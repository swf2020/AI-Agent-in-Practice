"""
串行 vs 并发调用性能对比。
运行前确保 .env 中至少有一个有效的 API Key。
"""
import asyncio
import time
import sys
import os

# 将当前目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_gateway_gateway import LLMGateway


async def benchmark_serial_vs_concurrent(n_requests: int = 5) -> None:
    """对比串行和并发调用 n_requests 次的总耗时"""
    gateway = LLMGateway()
    prompts = [f"用一句话解释什么是机器学习（第{i+1}次）" for i in range(n_requests)]

    print(f"\n{'='*50}")
    print(f"测试 {n_requests} 次请求，模型：deepseek-chat")
    print(f"{'='*50}")

    # --- 串行调用 ---
    start = time.perf_counter()
    serial_results = []
    for p in prompts:
        r = await gateway.chat(p, model="deepseek-chat", feature="benchmark_serial")
        serial_results.append(r)
    serial_time = time.perf_counter() - start
    print(f"\n[串行]  总耗时：{serial_time:.2f}s  |  均摊：{serial_time/n_requests:.2f}s/req")

    # 重置统计，区分两次测试
    gateway.tracker.reset()

    # --- 并发调用 ---
    start = time.perf_counter()
    concurrent_results = await gateway.chat_batch(
        prompts, model="deepseek-chat", max_concurrent=n_requests, feature="benchmark_concurrent"
    )
    concurrent_time = time.perf_counter() - start
    speedup = serial_time / concurrent_time
    print(f"[并发]  总耗时：{concurrent_time:.2f}s  |  加速比：{speedup:.1f}x")

    # 打印成本报告
    print("\n--- 成本报告（本次并发测试）---")
    import json
    print(json.dumps(gateway.cost_report(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(benchmark_serial_vs_concurrent(n_requests=5))