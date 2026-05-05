"""
双 Agent 代码生成+审查系统入口。
"""

from agents import run_dual_agent_loop
from core_config import get_litellm_id, get_chat_model_id, ACTIVE_MODEL_KEY


def main() -> None:
    print(f"当前模型: {ACTIVE_MODEL_KEY} (LiteLLM: {get_litellm_id()}, 直连: {get_chat_model_id()})")

    # 示例需求：实现一个带缓存的斐波那契计算函数
    requirement = """
    实现一个 Python 函数 `fibonacci(n: int) -> int`，要求：
    1. 使用 LRU 缓存避免重复计算
    2. 对负数输入抛出 ValueError，错误信息为 "n must be non-negative"
    3. 支持 n=0（返回 0）和 n=1（返回 1）
    4. 函数本身不能有副作用（纯函数）
    同时实现 `fibonacci_sequence(count: int) -> list[int]`，返回前 count 个斐波那契数列。
    """

    result = run_dual_agent_loop(
        requirement,
        pass_threshold=0.85,
        max_rounds=6,
        verbose=True,
    )

    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    print(f"状态: {'通过' if result['success'] else '未达标'}")
    print(f"轮次: {result['rounds']}")
    print(f"综合评分: {result['final_score']:.2f}/1.00")
    print(f"\n最终代码：\n{result['final_code']}")


if __name__ == "__main__":
    main()
