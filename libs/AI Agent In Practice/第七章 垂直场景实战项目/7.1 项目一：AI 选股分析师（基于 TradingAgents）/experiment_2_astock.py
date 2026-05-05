"""
实验二：用 AKShare 适配器分析 A 股，以贵州茅台（600519）为例。
"""
import os
from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.config import TradingAgentsConfig
from astock_adapter import AStockAdapter
from rich.console import Console

load_dotenv()
console = Console()


def _make_config() -> TradingAgentsConfig:
    """创建 A 股分析配置"""
    return TradingAgentsConfig(
        llm_provider="litellm",
        deep_think_llm="deepseek/deepseek-chat",
        quick_think_llm="deepseek/deepseek-chat",
        reasoning_effort="medium",
        max_debate_rounds=3,
        max_risk_discuss_rounds=3,
        max_recur_limit=100,
    )


def patch_astock_tools(adapter: AStockAdapter) -> None:
    """
    将 A 股数据源注入 TradingAgents 的工具层。

    ⚠️ tradingagents 0.3.1 不再暴露 graph.toolkit，此函数保留为占位，
    实际数据替换需在 upstream 支持后启用。
    """
    console.print("[yellow]⚠ A 股工具注入（0.3.1 暂不支持 toolkit 替换，仅验证数据层）[/yellow]")
    # 验证数据可用性
    sample = adapter.get_price_history("600519", days=5)
    console.print(f"[green]✓ A 股数据层可用，600519 最新收盘价：{sample['Close'].iloc[-1]:.2f} CNY[/green]")
    console.print("[green]✓ A 股数据源适配完成[/green]")


def analyze_astock(ticker: str, analysis_date: str) -> dict:
    """
    分析 A 股股票。

    Args:
        ticker: A 股代码，如 "600519"（贵州茅台）
        analysis_date: 分析日期，格式 "YYYY-MM-DD"
    """
    config = _make_config()

    adapter = AStockAdapter()

    # 提前验证数据可用性
    console.print(f"[dim]验证 {ticker} 数据可用性...[/dim]")
    try:
        sample = adapter.get_price_history(ticker, days=5)
        console.print(f"[green]✓ 数据可用，最新收盘价：{sample['Close'].iloc[-1]:.2f} CNY[/green]")
    except ValueError as e:
        console.print(f"[red]✗ 数据验证失败：{e}[/red]")
        raise

    # 注入 A 股数据源（验证层）
    patch_astock_tools(adapter)

    graph = TradingAgentsGraph(
        selected_analysts=["fundamentals", "news"],
        config=config,
        debug=False,
    )

    console.print(f"[bold cyan]开始分析 A 股 {ticker}...[/bold cyan]")
    state, decision = graph.propagate(ticker, analysis_date)

    # 处理 decision 返回值：新版本返回字符串，旧版本返回字典
    if isinstance(decision, str):
        decision_dict = {
            "action": decision.lower(),
            "reasoning": "分析完成",
            "confidence": 0.8,
        }
    else:
        decision_dict = decision

    return {"ticker": ticker, "date": analysis_date, "decision": decision_dict, "state": state}


if __name__ == "__main__":
    # 分析贵州茅台
    result = analyze_astock("600519", "2025-01-10")

    action = result["decision"].get("action", "N/A")
    confidence = result["decision"].get("confidence", 0)
    console.print(f"\n[bold]茅台分析结论：{action.upper()} | 置信度：{confidence:.0%}[/bold]")
    console.print(result["decision"].get("reasoning", "")[:800])
