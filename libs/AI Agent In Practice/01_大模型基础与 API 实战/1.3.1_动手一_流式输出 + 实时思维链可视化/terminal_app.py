"""
终端版思维链可视化。
运行方式：python terminal_app.py
支持 DeepSeek、Qwen 或 OpenAI 模型。
"""

import time
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from core import ChunkType, stream_cot_prompt, stream_extended_thinking

console = Console()


def run_terminal(
    prompt: str,
    use_extended_thinking: bool = False,
) -> dict:
    """
    在终端中实时渲染思维链，返回本次调用的统计数据。

    Args:
        prompt:                用户问题
        use_extended_thinking: True 时使用 DeepSeek 推理模型

    Returns:
        统计字典：ttft / total_time / token_count / tokens_per_sec
    """
    thinking_text = Text(style="dim cyan")
    answer_text = Text(style="bold white")

    stats = {
        "ttft": None,
        "total_time": 0.0,
        "token_count": 0,
        "thinking_tokens": 0,
        "answer_tokens": 0,
    }

    start_time = time.perf_counter()

    def build_layout() -> Columns:
        """每次 Live 刷新时重新构建布局。"""
        elapsed = time.perf_counter() - start_time
        tps = stats["token_count"] / elapsed if elapsed > 0 else 0

        stat_text = (
            f"⏱ {elapsed:.1f}s  "
            f"⚡ TTFT: {stats['ttft']:.3f}s  " if stats["ttft"] else f"⏱ {elapsed:.1f}s  "
        )
        stat_text += f"🔢 {stats['token_count']} tokens  📈 {tps:.1f} tok/s"

        return Columns(
            [
                Panel(
                    thinking_text,
                    title="🧠 [dim]思考过程[/dim]",
                    border_style="dim cyan",
                    padding=(0, 1),
                ),
                Panel(
                    answer_text,
                    title="✅ [bold green]最终回答[/bold green]",
                    border_style="green",
                    padding=(0, 1),
                ),
            ],
            equal=True,
        )

    stream_fn = (
        stream_extended_thinking(prompt, use_reasoner=True)
        if use_extended_thinking
        else stream_cot_prompt(prompt)
    )

    with Live(console=console, refresh_per_second=20, transient=False) as live:
        for chunk in stream_fn:
            now = time.perf_counter()

            if stats["ttft"] is None:
                stats["ttft"] = now - start_time

            stats["token_count"] += 1

            if chunk.chunk_type == ChunkType.THINKING:
                stats["thinking_tokens"] += 1
                thinking_text.append(chunk.content)
            else:
                stats["answer_tokens"] += 1
                answer_text.append(chunk.content)

            live.update(build_layout())

    stats["total_time"] = time.perf_counter() - start_time
    return stats


def main() -> None:
    console.rule("[bold]实时思维链可视化 · 终端版[/bold]")

    prompts = {
        "1": "一个农夫有17只羊，除了9只其余都死了，还剩几只？",
        "2": "小明有72块糖，要平均分给9个朋友，每人能分到几块？如果又来了3个朋友，重新分配后每人能分到几块？",
        "3": "一列火车从A城出发，以90km/h的速度行驶。另一列火车同时从B城出发，以60km/h的速度相向而行。AB两城距离450km，两列火车何时相遇？",
    }

    console.print("\n选择测试题（直接输入问题或选择编号）：")
    for k, v in prompts.items():
        console.print(f"  [{k}] {v}")

    user_input = console.input("\n> ").strip()
    prompt = prompts.get(user_input, user_input)

    use_extended_thinking = console.input("\n使用 DeepSeek 推理模型？(y/N): ").strip().lower() == "y"

    console.print()
    stats = run_terminal(prompt, use_extended_thinking=use_extended_thinking)

    console.print()
    console.rule("📊 本次统计")
    console.print(f"  TTFT:           {stats['ttft']:.3f}s")
    console.print(f"  总耗时:         {stats['total_time']:.2f}s")
    console.print(f"  总 token 数:    {stats['token_count']}")
    console.print(f"  思考 tokens:    {stats['thinking_tokens']}")
    console.print(f"  回答 tokens:    {stats['answer_tokens']}")
    tps = stats["token_count"] / stats["total_time"]
    console.print(f"  平均速率:       {tps:.1f} tok/s")


if __name__ == "__main__":
    main()
