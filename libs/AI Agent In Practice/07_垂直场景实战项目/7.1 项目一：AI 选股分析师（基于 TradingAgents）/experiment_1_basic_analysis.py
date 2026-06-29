"""
实验一：分析美股 NVDA / TSLA，解读 TradingAgents 五级评级输出。
依赖上一步配置好的 .env 文件。
"""
from dotenv import load_dotenv
from rich.console import Console

# TradingAgents 公开 API
from tradingagents.graph.trading_graph import TradingAgentsGraph
from core_config import create_default_config, normalize_decision  # [Fix #3][Fix #4]

load_dotenv()
console = Console()

# 风险偏好 → reasoning_effort 映射
RISK_TO_EFFORT = {
    "aggressive": "high",
    "neutral": "medium",
    "conservative": "low",
}


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
    config = create_default_config(  # [Fix #4] 复用 core_config 统一配置
        reasoning_effort=RISK_TO_EFFORT.get(risk_level, "medium")
    )

    # 初始化图（懒加载，不调用不产生费用）
    graph = TradingAgentsGraph(
        selected_analysts=["fundamentals", "news"],
        config=config,
        debug=False,  # 生产建议 False，调试时设 True 查看中间输出
    )

    console.print(f"[bold cyan]开始分析 {ticker}（{analysis_date}）...[/bold cyan]")

    # 执行分析——这里会触发完整的 Multi-Agent 调用链
    # 耗时通常 2-5 分钟，取决于模型响应速度
    state, decision = graph.propagate(ticker, analysis_date)

    # 统一 decision 格式（兼容新旧版本返回值） [Fix #3]
    decision_dict = normalize_decision(decision)

    return {
        "ticker": ticker,
        "date": analysis_date,
        "decision": decision_dict,
        "state": state,
    }
