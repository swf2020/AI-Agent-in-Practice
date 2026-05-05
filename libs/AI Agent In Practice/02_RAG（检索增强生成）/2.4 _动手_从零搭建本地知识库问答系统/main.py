"""
RAG 本地知识库问答系统 — 统一入口
用法：
  python main.py index <url_or_file>    # 索引文档
  python main.py ask "你的问题"          # 问答查询
  python main.py chainlit               # 启动 Chainlit Web 界面
"""
from __future__ import annotations

import sys


def index(source: str, strategy: str = "section") -> None:
    """索引一个文档到本地知识库"""
    from step3_index import index_document
    index_document(source, strategy=strategy)


def ask(question: str, top_k: int = 5, score_threshold: float = 0.5) -> None:
    """对知识库提问"""
    from step4_query import RAGPipeline
    pipeline = RAGPipeline(top_k=top_k, score_threshold=score_threshold)
    result = pipeline.ask(question)
    print(f"\n问题：{result.question}\n")
    print(f"回答：\n{result.answer}\n")
    print(f"引用来源（{len(result.sources)} 条）：")
    for c in result.sources:
        print(f"  [{c.score:.3f}] {c.source}")
    # 关闭 Qdrant 连接
    pipeline.qdrant.close()


def launch_chainlit() -> None:
    """启动 Chainlit Web 界面"""
    import subprocess
    subprocess.run([sys.executable, "-m", "chainlit", "run", "app.py"])


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "index":
        source = sys.argv[2] if len(sys.argv) > 2 else "https://docs.python.org/3/library/pathlib.html"
        strategy = sys.argv[3] if len(sys.argv) > 3 else "section"
        index(source, strategy)

    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("用法：python main.py ask \"你的问题\"")
            return
        ask(sys.argv[2])

    elif cmd == "chainlit":
        launch_chainlit()

    else:
        print(f"未知命令：{cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
