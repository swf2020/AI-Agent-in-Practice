"""
实验一：分析美股 NVDA / TSLA，解读 TradingAgents 五级评级输出。
依赖上一步配置好的 .env 文件。
"""
import os
from datetime import date, timedelta
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# TradingAgents 公开 API
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()
console = Console()


def run_analysis(
    ticker: str,
    analysis_date: str,
    risk_level: str = "neutral",  # 可选: aggressive / neutral / conservative
) -> dict:
    """
    对单支股票运行完整的 Multi-Agent 分析流程。

    Args:
        ticker: 股票代码，如 "NVDA"
        analysis_date: 分析基准日期，格式 "YYYY-MM-DD"
        risk_level: 风险偏好，影响 Risk Manager 的决策阈值

    Returns:
        包含决策和分析报告的字典
    """
    # 复制默认配置，避免污染全局状态
    config = DEFAULT_CONFIG.copy()
    config.update({
        "llm_provider": "openai",
        "deep_think_llm": "gpt-4o",       # 高推理需求节点（研究员/风控）
        "quick_think_llm": "gpt-4o-mini",  # 低推理需求节点（数据汇总）
        "risk_tolerance": risk_level,
        "online_tools": True,              # 启用实时数据获取
    })

    # 初始化图（懒加载，不调用不产生费用）
    graph = TradingAgentsGraph(
        selected_analysts=["fundamental", "sentiment", "news", "technical"],
        config=config,
        debug=False,  # 生产建议 False，调试时设 True 查看中间输出
    )

    console.print(f"[bold cyan]开始分析 {ticker}（{analysis_date}）...[/bold cyan]")

    # 执行分析——这里会触发完整的 Multi-Agent 调用链
    # 耗时通常 2-5 分钟，取决于模型响应速度
    state, decision = graph.propagate(ticker, analysis_date)

    return {
        "ticker": ticker,
        "date": analysis_date,
        "decision": decision,
        "state": state,
    }