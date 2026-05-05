"""
主入口：流式输出 + 实时思维链可视化。
运行方式：
  python main.py              — 终端模式
  streamlit run web_app.py    — Web 模式
  python experiment.py        — 三模式对比实验
"""

from terminal_app import main as terminal_main


if __name__ == "__main__":
    terminal_main()
