"""
RAG 本地知识库问答系统 — 统一入口
用法：
  python main.py index <url_or_file> [--strategy section|fixed]  # 索引文档
  python main.py ask "你的问题" [--top_k 5] [--threshold 0.5]   # 问答查询
  python main.py chainlit                                        # 启动 Chainlit Web 界面
"""  # [Fix #6] 使用 argparse 提供更好的帮助和参数校验
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(
        description="RAG 本地知识库问答系统 — 支持索引、查询和 Web 界面",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py index https://example.com/doc.html
  python main.py index paper.pdf --strategy fixed
  python main.py ask "什么是 RAG？"
  python main.py ask "如何调优？" --top_k 3 --threshold 0.6
  python main.py chainlit
        """,
    )  # [Fix #6]

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # index 子命令
    parser_index = subparsers.add_parser("index", help="索引文档到本地知识库")
    parser_index.add_argument(
        "source", nargs="?", default="https://docs.python.org/3/library/pathlib.html",
        help="文档 URL 或本地文件路径"
    )
    parser_index.add_argument(
        "--strategy", choices=["section", "fixed"], default="section",
        help="切块策略（默认: section）"
    )

    # ask 子命令
    parser_ask = subparsers.add_parser("ask", help="对知识库提问")
    parser_ask.add_argument("question", help="你的问题")
    parser_ask.add_argument("--top_k", type=int, default=5, help="检索返回的文档块数量（默认: 5）")
    parser_ask.add_argument("--threshold", type=float, default=0.5, help="相似度阈值（默认: 0.5）")

    # chainlit 子命令
    subparsers.add_parser("chainlit", help="启动 Chainlit Web 界面")

    args = parser.parse_args()

    if args.command == "index":
        index(args.source, strategy=args.strategy)
    elif args.command == "ask":
        ask(args.question, top_k=args.top_k, score_threshold=args.threshold)
    elif args.command == "chainlit":
        launch_chainlit()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
