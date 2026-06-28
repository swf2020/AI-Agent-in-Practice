"""
三模式对比实验：直接回答 / CoT Prompt / DeepSeek 思考模式（Prompt 驱动）
运行方式：python experiment.py
支持 DeepSeek、Qwen 或 OpenAI 模型。
"""

import time
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from core import (
    ChunkType,
    get_default_model,
    get_openai_client,
    stream_cot_prompt,
    stream_extended_thinking,
)

console = Console()
client = get_openai_client()
default_model = get_default_model()

# [Fix #12] 思考模式默认预算：4000 tokens 适合中等复杂度推理题，
# 题目简单时可降到 2000，复杂推理建议 8000+
DEFAULT_THINKING_BUDGET = 4000


@dataclass
class ExperimentResult:
    mode: str
    answer: str
    ttft: float
    total_time: float
    token_count: int
    thinking_tokens: int = 0
    answer_tokens: int = 0

    @property
    def tokens_per_sec(self) -> float:
        return self.token_count / self.total_time if self.total_time > 0 else 0


def run_direct_answer(prompt: str, model: str = None) -> ExperimentResult:
    """无 CoT 的直接回答，作为 baseline。"""
    model = model or default_model
    start = time.perf_counter()
    ttft = None
    full_text = ""
    token_count = 0

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.0,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        if ttft is None:
            ttft = time.perf_counter() - start
        full_text += delta
        token_count += 1

    return ExperimentResult(
        mode="直接回答（无 CoT）",
        answer=full_text.strip(),
        ttft=ttft or 0,
        total_time=time.perf_counter() - start,
        token_count=token_count,
        answer_tokens=token_count,
    )


def run_cot_prompt(prompt: str, model: str = None) -> ExperimentResult:
    """CoT Prompt 模式。"""
    model = model or default_model
    start = time.perf_counter()
    ttft = None
    think_text = ""
    answer_text = ""
    token_count = 0
    thinking_tokens = 0

    for chunk in stream_cot_prompt(prompt, model=model, temperature=0.0):
        if ttft is None:
            ttft = time.perf_counter() - start
        token_count += 1
        if chunk.chunk_type == ChunkType.THINKING:
            think_text += chunk.content
            thinking_tokens += 1
        else:
            answer_text += chunk.content

    return ExperimentResult(
        mode="CoT Prompt（<think> 标签）",
        answer=answer_text.strip(),
        ttft=ttft or 0,
        total_time=time.perf_counter() - start,
        token_count=token_count,
        thinking_tokens=thinking_tokens,
        answer_tokens=token_count - thinking_tokens,
    )


def run_extended_thinking(prompt: str, budget: int = DEFAULT_THINKING_BUDGET) -> ExperimentResult:
    """DeepSeek 思考模式（通过系统提示词开启思考）。"""
    start = time.perf_counter()
    ttft = None
    think_text = ""
    answer_text = ""
    token_count = 0
    thinking_tokens = 0

    for chunk in stream_extended_thinking(prompt, budget_tokens=budget):
        if ttft is None:
            ttft = time.perf_counter() - start
        token_count += 1
        if chunk.chunk_type == ChunkType.THINKING:
            think_text += chunk.content
            thinking_tokens += 1
        else:
            answer_text += chunk.content

    return ExperimentResult(
        mode="DeepSeek 思考模式（<thinking> 标签）",
        answer=answer_text.strip(),
        ttft=ttft or 0,
        total_time=time.perf_counter() - start,
        token_count=token_count,
        thinking_tokens=thinking_tokens,
        answer_tokens=token_count - thinking_tokens,
    )


def print_comparison(results: list[ExperimentResult], question: str) -> None:
    """打印横向对比报告。"""
    console.print()
    console.rule("[bold]对比实验报告[/bold]")
    console.print(f"[dim]问题：{question}[/dim]\n")

    for r in results:
        console.print(f"[bold]{r.mode}[/bold]")
        console.print(f"  答案：[green]{r.answer[:200]}[/green]")
        console.print()

    table = Table(title="📊 性能指标对比", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="dim", width=20)
    for r in results:
        table.add_column(r.mode, justify="right")

    metrics = [
        ("TTFT（首 token）", lambda r: f"{r.ttft:.3f}s"),
        ("总耗时", lambda r: f"{r.total_time:.2f}s"),
        ("总 tokens", lambda r: str(r.token_count)),
        ("思考 tokens", lambda r: str(r.thinking_tokens)),
        ("回答 tokens", lambda r: str(r.answer_tokens)),
        ("Token/s", lambda r: f"{r.tokens_per_sec:.1f}"),
    ]

    for metric_name, fn in metrics:
        table.add_row(metric_name, *[fn(r) for r in results])

    console.print(table)


def main() -> None:
    questions = [
        "小张每天能生产120个零件，小李每天能生产80个。工厂需要1200个零件，两人一起工作几天能完成？如果小张请假2天，总共需要几天？",
        "一个水池有进水管和出水管。进水管单独开8小时注满，出水管单独开12小时排完。如果同时打开两管，几小时能注满？",
    ]

    question = questions[0]
    console.print(f"\n[bold]实验题目：[/bold]{question}")
    console.print(f"[dim]使用模型：{default_model}[/dim]\n")
    console.print("[dim]正在运行三种模式（每次约 10-30 秒）...[/dim]\n")

    results = []

    console.print("[bold]1/3[/bold] 直接回答...")
    results.append(run_direct_answer(question))
    console.print("   ✓ 完成\n")

    console.print("[bold]2/3[/bold] CoT Prompt...")
    results.append(run_cot_prompt(question))
    console.print("   ✓ 完成\n")

    # [Fix #9] 文案与函数行为一致：此处走 Prompt 驱动模式（非原生 reasoner）
    console.print("[bold]3/3[/bold] DeepSeek 思考模式（Prompt 驱动）...")
    results.append(run_extended_thinking(question))
    console.print("   ✓ 完成\n")

    print_comparison(results, question)


if __name__ == "__main__":
    main()
