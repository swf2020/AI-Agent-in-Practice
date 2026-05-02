# 端到端冒烟测试（直接复制运行）
# smoke_test.py

import os
os.environ["OPENAI_API_KEY"] = "your-key-here"  # 或从 .env 读取

from agents import run_dual_agent_loop

result = run_dual_agent_loop(
    requirement="实现 `add(a: int, b: int) -> int` 函数，对非整数输入抛出 TypeError。",
    pass_threshold=0.80,
    max_rounds=3,
    verbose=True,
)

assert result["final_code"] != "", "final_code 不应为空"
print(f"\n冒烟测试完成：{result['rounds']} 轮，评分 {result['final_score']:.2f}")