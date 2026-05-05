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
from tradingagents.config import TradingAgentsConfig

load_dotenv()
console = Console()

# 风险偏好 → reasoning_effort 映射
RISK_TO_EFFORT = {
    "aggressive":   "high",
    "neutral":      "medium",
    "conservative": "low",
}


def _make_config(risk_level: str = "neutral") -> TradingAgentsConfig:
    """创建 TradingAgentsConfig 实例"""
    return TradingAgentsConfig(
        llm_provider="openai",
        deep_think_llm="gpt-4o",
        quick_think_llm="gpt-4o-mini",
        reasoning_effort=RISK_TO_EFFORT.get(risk_level, "medium"),
        max_debate_rounds=3,
        max_risk_discuss_rounds=3,
        max_recur_limit=100,
    )


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
    config = _make_config(risk_level)

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