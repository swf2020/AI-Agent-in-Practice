"""主入口 — 工具调用 Agent 演示程序"""

import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core_config import get_model_list, ACTIVE_MODEL_KEY
from agent import build_agent


def main():
    print(f"当前使用模型: {ACTIVE_MODEL_KEY} ({get_model_list()})")
    print("=" * 60)

    # 初始化 Agent（含搜索、代码执行、数据库三类工具）
    agent = build_agent(db_url="sqlite:///demo.db")

    print("可用工具:")
    for schema in agent._dispatcher.schemas:
        fn = schema["function"]
        print(f"  - {fn['name']}: {fn['description'][:80]}...")

    print("\n" + "=" * 60)
    print("交互式对话模式（输入 'quit' 或 'exit' 退出）")
    print("=" * 60)

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("再见!")
            break

        print("\n思考中...")
        try:
            answer = agent.run(user_input, verbose=True)
            print(f"\n回答: {answer}")
        except Exception as e:
            print(f"\n错误: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
