# 端到端冒烟测试（直接复制运行）
# smoke_test.py

import os
from dotenv import load_dotenv
load_dotenv()

# 确保当前激活模型所需的环境变量已设置
from core_config import ACTIVE_MODEL_KEY, MODEL_REGISTRY
_api_key_env = MODEL_REGISTRY[ACTIVE_MODEL_KEY].get("api_key_env")
if _api_key_env and not os.environ.get(_api_key_env):
    os.environ[_api_key_env] = "your-key-here"  # 仅作冒烟测试

from agents import run_dual_agent_loop

result = run_dual_agent_loop(
    requirement="实现 `add(a: int, b: int) -> int` 函数，对非整数输入抛出 TypeError。",
    pass_threshold=0.80,
    max_rounds=3,
    verbose=True,
)

assert result["final_code"] != "", "final_code 不应为空"
print(f"\n冒烟测试完成：{result['rounds']} 轮，评分 {result['final_score']:.2f}")
