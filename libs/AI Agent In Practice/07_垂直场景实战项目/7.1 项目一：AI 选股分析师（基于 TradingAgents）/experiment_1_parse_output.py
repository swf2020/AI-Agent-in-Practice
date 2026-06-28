"""
解析并可视化 TradingAgents 的决策输出。
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# 五级评级颜色映射（便于终端可视化）
RATING_COLORS = {
    "strong buy": "bold green",
    "buy": "green",
    "hold": "yellow",
    "sell": "red",
    "strong sell": "bold red",
}

# 风险等级文字说明（对应 Risk Manager 的三种风险偏好参数）
RISK_LABELS = {
    "aggressive": "激进型（高收益优先）",
    "neutral": "中性型（风险收益平衡）",
    "conservative": "保守型（资本保全优先）",
}


def display_decision(result: dict) -> None:
    """
    结构化展示分析决策，包括评级、理由和风险提示。

    Args:
        result: run_analysis() 返回的字典
    """
    decision = result["decision"]
    ticker = result["ticker"]

    # ---- 1. 核心决策面板 ----
    rating = decision.get("action", "unknown").lower()
    color = RATING_COLORS.get(rating, "white")

    rating_text = Text(rating.upper(), style=color)
    console.print(Panel(
        rating_text,
        title=f"[bold]{ticker} 投资评级[/bold]",
        subtitle=f"分析日期：{result['date']}",
        expand=False,
    ))

    # ---- 2. 量化指标表格 ----
    table = Table(title="关键决策指标", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="cyan", width=20)
    table.add_column("值", width=30)

    # 安全提取 risk_tolerance：兼容 dict 和对象两种 state 格式 [Fix #1]
    risk_tol_raw = "neutral"
    state_raw = result.get("state", {})
    if hasattr(state_raw, "risk_tolerance"):
        risk_tol_raw = state_raw.risk_tolerance  # 对象属性访问
    elif isinstance(state_raw, dict):
        risk_tol_raw = state_raw.get("risk_tolerance", "neutral")  # 字典键访问

    # 置信度展示：None 表示未获取到有效值 [Fix #3]
    conf_val = decision.get('confidence', 0)
    conf_str = "未知" if conf_val is None else f"{conf_val:.0%}"

    metrics = [
        ("目标买入价", decision.get("target_price", "N/A")),
        ("止损价", decision.get("stop_loss", "N/A")),
        ("止盈价", decision.get("take_profit", "N/A")),
        ("置信度", conf_str),
        ("风险偏好", RISK_LABELS.get(risk_tol_raw, "未知")),
    ]
    for name, value in metrics:
        table.add_row(name, str(value))

    console.print(table)

    # ---- 3. 核心论点摘要（来自研究员综合报告）----
    reasoning = decision.get("reasoning", "无可用分析")
    console.print(Panel(
        reasoning[:1500] + ("..." if len(reasoning) > 1500 else ""),
        title="[bold]核心投资论点[/bold]",
        border_style="blue",
    ))


def compare_risk_profiles(ticker: str, analysis_date: str) -> None:
    """
    用相同数据、三种风险偏好运行分析，对比评级差异。
    展示 Risk Manager 如何因风险阈值不同而给出不同建议。

    注意：此函数依赖 experiment_1_basic_analysis.run_analysis，
    函数内延迟导入以避免模块级循环依赖。
    """
    from experiment_1_basic_analysis import run_analysis  # [Fix #1] 延迟导入避免调用时 NameError

    results = {}
    for risk in ["aggressive", "neutral", "conservative"]:
        console.print(f"\n[dim]运行 {risk} 模式...[/dim]")
        results[risk] = run_analysis(ticker, analysis_date, risk_level=risk)

    # 汇总对比
    compare_table = Table(title=f"{ticker} 三种风险偏好评级对比")
    compare_table.add_column("风险偏好")
    compare_table.add_column("评级")
    compare_table.add_column("置信度")

    for risk, result in results.items():
        dec = result["decision"]
        rating = dec.get("action", "N/A").lower()
        color = RATING_COLORS.get(rating, "white")
        c_val = dec.get('confidence', 0)
        c_str = "未知" if c_val is None else f"{c_val:.0%}"  # [Fix #3]
        compare_table.add_row(
            RISK_LABELS[risk],
            Text(rating.upper(), style=color),
            c_str,
        )

    console.print(compare_table)


# ---- 主入口 ----
if __name__ == "__main__":
    from experiment_1_basic_analysis import run_analysis

    # 分析 NVDA（使用上周交易日，避免周末无数据问题）
    nvda_result = run_analysis(
        ticker="NVDA",
        analysis_date="2025-01-10",
        risk_level="neutral",
    )
    display_decision(nvda_result)

    # ⚠️ 生产注意：compare_risk_profiles 会触发 3 次完整分析链路
    # 约消耗 300-500K tokens，建议在调试完成后再运行
    # compare_risk_profiles("TSLA", "2025-01-10")
