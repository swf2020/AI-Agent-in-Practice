"""
实验四：用相同股票/日期，对比 OpenAI GPT-4o vs DeepSeek-V3 的分析质量。

评估维度：
1. 评级一致性（是否给出相同的 BUY/HOLD/SELL）
2. 推理深度（核心论点数量与具体性）
3. 成本（Token 消耗与费用）
4. 延迟（端到端耗时）
"""
import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Literal
from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.config import TradingAgentsConfig
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()

ModelProvider = Literal["openai", "deepseek", "ollama"]


@dataclass
class ModelConfig:
    """模型配置，对应 TradingAgents config 的 llm_provider 相关字段。"""
    name: str
    provider: ModelProvider
    deep_model: str    # 用于高推理需求节点（研究员、风控）
    quick_model: str   # 用于低推理需求节点（数据汇总）
    cost_per_1m_input: float   # USD/1M input tokens（用于成本估算）
    cost_per_1m_output: float  # USD/1M output tokens


# 2025 年初定价（实际以官网为准）
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "gpt4o": ModelConfig(
        name="GPT-4o",
        provider="openai",
        deep_model="gpt-4o",
        quick_model="gpt-4o-mini",
        cost_per_1m_input=5.0,
        cost_per_1m_output=15.0,
    ),
    "deepseek": ModelConfig(
        name="DeepSeek-V3",
        provider="deepseek",
        deep_model="deepseek-chat",       # DeepSeek-V3 对应的模型 ID
        quick_model="deepseek-chat",
        cost_per_1m_input=0.27,           # DeepSeek 定价约为 GPT-4o 的 5%
        cost_per_1m_output=1.10,
    ),
}


def _make_config(mc: ModelConfig) -> TradingAgentsConfig:
    """根据 ModelConfig 创建 TradingAgentsConfig"""
    return TradingAgentsConfig(
        llm_provider=mc.provider,
        deep_think_llm=mc.deep_model,
        quick_think_llm=mc.quick_model,
        reasoning_effort="medium",
        max_debate_rounds=3,
        max_risk_discuss_rounds=3,
        max_recur_limit=100,
    )


@dataclass
class AnalysisResult:
    """单次分析结果的结构化记录。"""
    model_name: str
    ticker: str
    analysis_date: str
    action: str
    confidence: float
    target_price: float | None
    reasoning_length: int          # 推理文本字符数（粗略衡量深度）
    elapsed_seconds: float
    estimated_cost_usd: float
    raw_decision: dict = field(default_factory=dict)


def run_with_model(
    ticker: str,
    analysis_date: str,
    model_key: str,
) -> AnalysisResult:
    """
    用指定模型运行分析，返回结构化结果。

    Args:
        ticker: 股票代码
        analysis_date: 分析日期
        model_key: MODEL_CONFIGS 中的键，如 "gpt4o" 或 "deepseek"
    """
    mc = MODEL_CONFIGS[model_key]
    console.print(f"\n[bold cyan]使用 {mc.name} 分析 {ticker}...[/bold cyan]")

    config = _make_config(mc)

    graph = TradingAgentsGraph(
        selected_analysts=["fundamentals", "news", "technical"],
        config=config,
        debug=False,
    )

    start_time = time.time()
    state, decision = graph.propagate(ticker, analysis_date)
    elapsed = time.time() - start_time

    reasoning = decision.get("reasoning", "")

    # 粗略估算成本（token 数基于字符数估算，1 token ≈ 4 字符）
    estimated_tokens = len(reasoning) / 4
    estimated_cost = (estimated_tokens / 1_000_000) * mc.cost_per_1m_output

    return AnalysisResult(
        model_name=mc.name,
        ticker=ticker,
        analysis_date=analysis_date,
        action=decision.get("action", "unknown"),
        confidence=decision.get("confidence", 0.0),
        target_price=decision.get("target_price"),
        reasoning_length=len(reasoning),
        elapsed_seconds=elapsed,
        estimated_cost_usd=estimated_cost,
        raw_decision=decision,
    )


def compare_models(
    ticker: str,
    analysis_date: str,
    models: list[str] | None = None,
) -> list[AnalysisResult]:
    """
    多模型对比实验主函数。

    Args:
        ticker: 股票代码
        analysis_date: 分析日期
        models: 要对比的模型键列表，默认全部

    Returns:
        所有模型的分析结果列表
    """
    models = models or list(MODEL_CONFIGS.keys())
    results: list[AnalysisResult] = []

    for model_key in models:
        try:
            result = run_with_model(ticker, analysis_date, model_key)
            results.append(result)
            console.print(f"[green]✓ {result.model_name} 完成：{result.action.upper()}[/green]")
        except Exception as e:
            console.print(f"[red]✗ {MODEL_CONFIGS[model_key].name} 失败：{e}[/red]")

    return results


def display_comparison(results: list[AnalysisResult]) -> None:
    """展示多模型对比表格，包含评级、成本、延迟维度。"""
    table = Table(title="多模型分析质量对比", show_header=True, header_style="bold")

    columns = [
        ("模型", "cyan"),
        ("评级", "white"),
        ("置信度", "white"),
        ("目标价", "white"),
        ("推理深度（字符）", "white"),
        ("耗时（秒）", "yellow"),
        ("估算成本（USD）", "green"),
    ]
    for col_name, style in columns:
        table.add_column(col_name, style=style)

    for r in results:
        table.add_row(
            r.model_name,
            r.action.upper(),
            f"{r.confidence:.0%}",
            f"${r.target_price:.2f}" if r.target_price else "N/A",
            str(r.reasoning_length),
            f"{r.elapsed_seconds:.1f}s",
            f"${r.estimated_cost_usd:.4f}",
        )

    console.print(table)

    # 评级一致性分析
    actions = [r.action.lower() for r in results]
    if len(set(actions)) == 1:
        console.print(f"[bold green]✓ 所有模型评级一致：{actions[0].upper()}[/bold green]")
    else:
        console.print("[bold yellow]⚠ 模型评级存在分歧，建议人工复核[/bold yellow]")
        for r in results:
            console.print(f"  {r.model_name}: {r.action.upper()}")


if __name__ == "__main__":
    results = compare_models("NVDA", "2025-01-10", models=["gpt4o", "deepseek"])
    display_comparison(results)

    # 保存原始结果（含完整推理文本）供离线分析
    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            ensure_ascii=False,
            indent=2,
        )
    console.print("[dim]详细结果已保存至 comparison_results.json[/dim]")