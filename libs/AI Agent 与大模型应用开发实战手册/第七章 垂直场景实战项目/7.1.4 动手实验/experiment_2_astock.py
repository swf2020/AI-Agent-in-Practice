"""
实验二：用 AKShare 适配器分析 A 股，以贵州茅台（600519）为例。
"""
import os
from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from astock_adapter import AStockAdapter
from rich.console import Console

load_dotenv()
console = Console()


def patch_astock_tools(graph: TradingAgentsGraph, adapter: AStockAdapter) -> None:
    """
    将 A 股数据源注入 TradingAgents 的工具层。

    TradingAgents 的 Analyst Tool 通过 graph.toolkit 注册，
    这里直接替换对应工具函数引用，绕过美股 API 依赖。

    ⚠️ 注意：此处依赖 TradingAgents 内部结构，版本升级后需验证接口兼容性。
    """
    # 替换价格历史工具
    graph.toolkit.get_price_data = lambda ticker, start, end: (
        adapter.get_price_history(ticker, days=90)
    )
    # 替换基本面工具
    graph.toolkit.get_company_profile = lambda ticker: (
        adapter.get_fundamental_info(ticker)
    )
    # 替换新闻工具
    graph.toolkit.get_stock_news = lambda ticker, limit=20: (
        adapter.get_news(ticker, limit=limit)
    )

    console.print("[green]✓ A 股数据源适配完成[/green]")


def analyze_astock(ticker: str, analysis_date: str) -> dict:
    """
    分析 A 股股票。

    Args:
        ticker: A 股代码，如 "600519"（贵州茅台）
        analysis_date: 分析日期，格式 "YYYY-MM-DD"
    """
    config = DEFAULT_CONFIG.copy()
    config.update({
        "llm_provider": "openai",
        "deep_think_llm": "gpt-4o",
        "quick_think_llm": "gpt-4o-mini",
        "risk_tolerance": "neutral",
        "online_tools": False,  # 关闭默认在线工具，使用我们的适配器
    })

    adapter = AStockAdapter()

    # 提前验证数据可用性，避免 Agent 运行一半才报错
    console.print(f"[dim]验证 {ticker} 数据可用性...[/dim]")
    try:
        sample = adapter.get_price_history(ticker, days=5)
        console.print(f"[green]✓ 数据可用，最新收盘价：{sample['Close'].iloc[-1]:.2f} CNY[/green]")
    except ValueError as e:
        console.print(f"[red]✗ 数据验证失败：{e}[/red]")
        raise

    graph = TradingAgentsGraph(
        selected_analysts=["fundamental", "news", "technical"],  # A 股暂不接入社媒情绪
        config=config,
        debug=False,
    )

    # 注入 A 股数据源
    patch_astock_tools(graph, adapter)

    console.print(f"[bold cyan]开始分析 A 股 {ticker}...[/bold cyan]")
    state, decision = graph.propagate(ticker, analysis_date)

    return {"ticker": ticker, "date": analysis_date, "decision": decision, "state": state}


if __name__ == "__main__":
    # 分析贵州茅台
    result = analyze_astock("600519", "2025-01-10")

    action = result["decision"].get("action", "N/A")
    confidence = result["decision"].get("confidence", 0)
    console.print(f"\n[bold]茅台分析结论：{action.upper()} | 置信度：{confidence:.0%}[/bold]")
    console.print(result["decision"].get("reasoning", "")[:800])