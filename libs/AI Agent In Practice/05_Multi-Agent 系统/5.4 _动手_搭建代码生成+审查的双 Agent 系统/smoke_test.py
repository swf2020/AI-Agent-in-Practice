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

# [Fix #7] 检测到示例 Key 时跳过 LLM 调用，避免认证失败
if _api_key_env and os.environ.get(_api_key_env, "").startswith("your-"):
    print(f"⚠️  API Key 未配置（{_api_key_env} 仍为示例值），跳过 LLM 调用")
    print(f"   请复制 .env.example 为 .env，并填入真实的 API Key")
    # 只做不涉及 LLM 调用的基础检查
    from agents import _extract_code_blocks
    blocks = _extract_code_blocks("```implementation\ndef hello(): pass\n```")
    assert "implementation" in blocks
    print("✅ 冒烟测试（基础检查）通过")
else:
    result = run_dual_agent_loop(
        requirement="实现 `add(a: int, b: int) -> int` 函数，对非整数输入抛出 TypeError。",
        pass_threshold=0.80,
        max_rounds=3,
        verbose=True,
    )

    assert result["final_code"] != "", "final_code 不应为空"
    print(f"\n冒烟测试完成：{result['rounds']} 轮，评分 {result['final_score']:.2f}")
