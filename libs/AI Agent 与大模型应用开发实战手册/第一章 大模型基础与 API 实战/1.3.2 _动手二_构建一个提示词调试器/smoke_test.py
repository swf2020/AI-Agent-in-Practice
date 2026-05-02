# smoke_test.py — 端到端冒烟测试（不启动 UI，仅测试核心逻辑）
"""
运行：python smoke_test.py
预期：在 ~5 秒内并发拿到所有选中模型的响应
"""
import asyncio
import os
import sys

# 确保能找到 core 模块
sys.path.insert(0, os.path.dirname(__file__))

import core  # 触发 dotenv 加载
from core.caller import call_all
from core.history import save_run, load_history


async def main():
    print("🔬 Prompt 调试器 — 冒烟测试\n")

    system = "You are a concise assistant. Answer in one sentence."
    user = "What is the capital of France?"
    models = ["gpt-4o", "deepseek-v3"]  # 去掉 Claude 节省 Key 消耗

    print(f"📤 发送到模型：{models}")
    print(f"📝 User Prompt：{user}\n")

    results = await call_all(
        selected_models=models,
        system_prompt=system,
        user_prompt=user,
        temperature=0.0,  # 确定性输出便于验证
        max_tokens=100,
    )

    print("=" * 60)
    for r in results:
        if r.error:
            print(f"❌ {r.model}: {r.error}")
        else:
            print(f"✅ {r.model}")
            print(f"   输出: {r.output.strip()}")
            print(f"   耗时: {r.latency}s | Tokens: {r.total_tokens} | 费用: ${r.estimated_cost}")
    print("=" * 60)

    # 测试历史存储
    run_id = save_run(system, user, models, 0.0, 100, results)
    print(f"\n💾 历史已保存，Run ID: {run_id}")

    df = load_history()
    print(f"📚 当前历史记录数（行数）: {len(df)}")
    print(df.head(3).to_string())

    print("\n✅ 冒烟测试通过！运行 `python app.py` 启动完整 UI")


if __name__ == "__main__":
    asyncio.run(main())