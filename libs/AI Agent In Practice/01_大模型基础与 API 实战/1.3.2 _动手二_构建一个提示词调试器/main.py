"""
Prompt 调试器 — 统一入口

用法：
  python main.py              # 启动 Gradio UI
  python main.py --cli        # CLI 模式（一次性调用）
  python main.py --test       # 运行冒烟测试
"""
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core  # 触发 dotenv 加载
from core_config import get_model_list


def run_cli():
    """CLI 模式：快速测试当前激活模型"""
    import asyncio
    from core.caller import call_single

    async def _main():
        result = await call_single(
            model_key="DeepSeek-V3",
            system_prompt="You are a concise assistant.",
            user_prompt="What is Python?",
            temperature=0.0,
            max_tokens=100,
        )
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Response: {result.output}")
            print(f"Tokens: {result.total_tokens} | Cost: ${result.estimated_cost}")

    asyncio.run(_main())


def run_ui():
    """启动 Gradio UI"""
    from app import demo
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        theme=None,
        css=".output-col { min-height: 300px; }",
        share=False,
        show_error=True,
    )


def run_test():
    """运行冒烟测试"""
    os.execl(sys.executable, sys.executable, "smoke_test.py")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    elif "--test" in sys.argv:
        run_test()
    else:
        run_ui()
