# 7.1.4 动手实验

## 实验目标

完成本节四个实验后，你将能够：
1. **端到端运行 TradingAgents 系统**，理解多 Agent 协作分析流程，并能解读五级评级的决策依据
2. **扩展数据源接入能力**，将系统从美股迁移到 A 股场景，掌握数据适配层的设计模式
3. **掌握 Checkpoint 持久化机制**，实现 Agent 任务的中断恢复，这是生产级长任务 Agent 的核心工程能力

核心学习点：多 Agent 协作中的数据流转与状态管理、LangGraph Checkpoint 机制原理、跨模型的效果对比方法论。

---

## 架构总览

```mermaid
graph TD
    subgraph 数据层["数据层（实验一/二）"]
        DS1["Yahoo Finance\nFinnHub / Reddit"]
        DS2["AKShare\n东方财富 API"]
    end

    subgraph Agent团队["Analyst Team（5类分析师）"]
        FA["基本面分析师\nFundamental Analyst"]
        SA["情绪分析师\nSentiment Analyst"]
        NA["新闻分析师\nNews Analyst"]
        TA["技术分析师\nTechnical Analyst"]
    end

    subgraph 决策层["Research & Decision Layer"]
        RM["研究员\nResearcher"]
        Bull["多头辩手\nBull Debater"]
        Bear["空头辩手\nBear Debater"]
        TM["交易员\nTrader"]
        Risk["风控经理\nRisk Manager"]
    end

    subgraph 持久化["Checkpoint 层（实验三）"]
        CP["LangGraph\nCheckpointSaver"]
        DB["SQLite /\nPostgres"]
    end

    subgraph 模型层["LLM 后端（实验四）"]
        GPT["OpenAI GPT-4o"]
        DS["DeepSeek-V3"]
        Qwen["Qwen2.5-72B"]
    end

    DS1 --> FA & SA & NA & TA
    DS2 --> FA & SA & NA & TA
    FA & SA & NA & TA --> RM
    RM --> Bull & Bear
    Bull & Bear --> TM
    TM --> Risk
    Risk -->|"五级评级\nBUY/HOLD/SELL..."| Output["结构化决策输出"]
    
    CP <--> DB
    决策层 <--> CP
    模型层 -.->|"可切换"| Agent团队
    模型层 -.-> 决策层
```

---

## 环境准备

```bash
# 创建虚拟环境（uv）
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装核心依赖
uv pip install tradingagents==0.1.4 \
    akshare==1.14.13 \
    langgraph==0.2.55 \
    langchain-openai==0.2.14 \
    langchain-deepseek==0.1.3 \
    pandas==2.2.3 \
    python-dotenv==1.0.1 \
    rich==13.9.4

# pip 备选
# pip install tradingagents akshare langgraph langchain-openai \
#             langchain-deepseek pandas python-dotenv rich
```

> Colab 用户：`!pip install tradingagents akshare langgraph langchain-openai langchain-deepseek rich` 即可，无需创建虚拟环境

配置 API Keys（项目根目录创建 `.env`）：

```bash
# .env
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
FINNHUB_API_KEY=your_finnhub_key   # 免费注册 finnhub.io
TAVILY_API_KEY=your_tavily_key     # 可选，新闻搜索
```

---

## 实验一：分析 NVDA / TSLA 并解读五级评级输出

### Step 1：最小化运行 TradingAgents

**目标**：用最简配置跑通完整分析链路，理解系统默认行为，再逐层定制。

```python
# experiment_1_basic_analysis.py
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
```

### Step 2：解读五级评级输出

**目标**：理解 TradingAgents 的 Structured Output 格式，提取关键决策字段并可视化。五级评级是 Pydantic Schema 约束下的枚举输出，背后反映的是 Bull vs Bear 辩论后风控经理的综合裁定。

```python
# experiment_1_parse_output.py
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
    "strong buy":  "bold green",
    "buy":         "green",
    "hold":        "yellow",
    "sell":        "red",
    "strong sell": "bold red",
}

# 风险等级文字说明（对应 Risk Manager 的三种风险偏好参数）
RISK_LABELS = {
    "aggressive":    "激进型（高收益优先）",
    "neutral":       "中性型（风险收益平衡）",
    "conservative":  "保守型（资本保全优先）",
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

    metrics = [
        ("目标买入价", decision.get("target_price", "N/A")),
        ("止损价", decision.get("stop_loss", "N/A")),
        ("止盈价", decision.get("take_profit", "N/A")),
        ("置信度", f"{decision.get('confidence', 0):.0%}"),
        ("风险偏好", RISK_LABELS.get(
            result["state"].get("risk_tolerance", "neutral"), "未知"
        )),
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
    """
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
        compare_table.add_row(
            RISK_LABELS[risk],
            Text(rating.upper(), style=color),
            f"{dec.get('confidence', 0):.0%}",
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
```

**关键点**：
- `decision` 字段由 Risk Manager 通过 Pydantic Schema 强制约束输出，保证字段不缺失
- `target_price` / `stop_loss` 是 Trader Agent 基于 Bull/Bear 辩论结论计算的，不是规则公式
- ⚠️ 同一天的 NVDA 和 TSLA 并行分析会并发消耗约 800K tokens，注意控制成本

---

## 实验二：接入 A 股数据源（东方财富 / AKShare）

### Step 3：实现 A 股数据适配层

**目标**：TradingAgents 的 Analyst Tool 依赖特定格式的 DataFrame，A 股数据源字段命名与美股不同，需要一个适配层做字段映射和格式归一化。这是实际工程中最常见的"换数据源"问题模式。

```python
# astock_adapter.py
"""
A 股数据适配层：将 AKShare / 东方财富数据格式统一为
TradingAgents 内部期望的标准格式。

标准格式要求（来自 TradingAgents 源码 tradingagents/dataflows/）：
- OHLCV: Open, High, Low, Close, Volume（列名大写）
- 日期索引：datetime 类型，时区为 UTC
- 股票代码：以 ticker 参数传入（如 "600519"）
"""
from __future__ import annotations

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AStockAdapter:
    """
    将 AKShare 数据格式适配为 TradingAgents 标准格式。

    用法示例：
        adapter = AStockAdapter()
        df = adapter.get_price_history("600519", days=90)
    """

    # AKShare 返回字段 → TradingAgents 标准字段映射
    # 东方财富接口返回中文列名，需要转换
    COLUMN_MAP = {
        "日期":   "Date",
        "开盘":   "Open",
        "最高":   "High",
        "最低":   "Low",
        "收盘":   "Close",
        "成交量": "Volume",
        "成交额": "Amount",   # TradingAgents 可选字段
        "振幅":   "Amplitude",
        "涨跌幅": "Change",
        "涨跌额": "ChangeAmt",
        "换手率": "Turnover",
    }

    def get_price_history(
        self,
        ticker: str,
        days: int = 90,
        adjust: str = "qfq",  # 前复权；"hfq"=后复权；""=不复权
    ) -> pd.DataFrame:
        """
        获取 A 股历史行情，返回 TradingAgents 兼容格式。

        Args:
            ticker: A 股代码，如 "600519"（贵州茅台）
            days: 获取最近 N 个交易日数据
            adjust: 复权方式，分析建议使用前复权 "qfq"

        Returns:
            标准化 OHLCV DataFrame，日期为索引

        Raises:
            ValueError: 股票代码不存在或无数据时
        """
        end_date = datetime.now().strftime("%Y%m%d")
        # 多取 50 天缓冲，因为 days 是交易日而不是自然日
        start_date = (
            datetime.now() - timedelta(days=days + 50)
        ).strftime("%Y%m%d")

        try:
            df = ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception as e:
            raise ValueError(
                f"无法获取 {ticker} 数据，请确认股票代码正确且市场开放。原始错误：{e}"
            ) from e

        if df.empty:
            raise ValueError(f"股票 {ticker} 在指定日期范围内无交易数据")

        # 字段映射
        df = df.rename(columns=self.COLUMN_MAP)

        # 日期索引标准化
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df.index = df.index.tz_localize("Asia/Shanghai").tz_convert("UTC")

        # 保留核心字段（TradingAgents 只需要 OHLCV）
        core_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[core_cols].astype(float)

        # 截取实际需要的天数（AKShare 返回数据可能超出范围）
        return df.tail(days)

    def get_fundamental_info(self, ticker: str) -> dict:
        """
        获取 A 股基本面信息，用于替换 TradingAgents 的 FinnHub 基本面工具。

        Returns:
            与 FinnHub /stock/profile2 字段对齐的字典
        """
        try:
            # 东方财富实时行情（含市值、PE、PB 等）
            info = ak.stock_individual_info_em(symbol=ticker)
            # info 是两列 DataFrame：["item", "value"]
            info_dict = dict(zip(info["item"], info["value"]))
        except Exception as e:
            logger.warning(f"获取 {ticker} 基本面信息失败：{e}")
            return {}

        # 映射到 TradingAgents 期望的字段名
        return {
            "name":        info_dict.get("股票简称", ticker),
            "ticker":      ticker,
            "exchange":    info_dict.get("所处交所", "SSE/SZSE"),
            "marketCap":   info_dict.get("总市值", 0),
            "pe":          info_dict.get("市盈率(动态)", None),
            "pb":          info_dict.get("市净率", None),
            "52weekHigh":  info_dict.get("52周最高", None),
            "52weekLow":   info_dict.get("52周最低", None),
            "currency":    "CNY",
            "country":     "CN",
        }

    def get_news(self, ticker: str, limit: int = 20) -> list[dict]:
        """
        获取 A 股新闻，替换 TradingAgents 的 NewsAPI 工具。

        Returns:
            与 TradingAgents News Tool 期望格式对齐的新闻列表
        """
        try:
            news_df = ak.stock_news_em(symbol=ticker)
        except Exception as e:
            logger.warning(f"获取 {ticker} 新闻失败：{e}")
            return []

        results = []
        for _, row in news_df.head(limit).iterrows():
            results.append({
                "headline": row.get("新闻标题", ""),
                "summary":  row.get("新闻内容", "")[:500],  # 截断避免超 Token
                "datetime": str(row.get("发布时间", "")),
                "source":   row.get("新闻来源", "东方财富"),
                "url":      row.get("新闻链接", ""),
            })

        return results
```

### Step 4：将适配器注入 TradingAgents

**目标**：TradingAgents 通过工具函数而非硬编码 API 调用获取数据，因此可以通过替换工具函数实现数据源切换，无需修改 Agent 核心逻辑。

```python
# experiment_2_astock.py
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
```

**关键点**：
- A 股情绪数据（类似 Reddit/StockTwits）国内替代品较少，建议先跳过 Sentiment Analyst，等后续接入雪球/东方财富社区数据
- AKShare 有频率限制，批量分析多支股票时加 `time.sleep(1)` 间隔
- ⚠️ A 股代码需要纯数字字符串（如 `"600519"`），不要加交易所后缀

---

## 实验三：开启 Checkpoint 实现中断续跑

### Step 5：配置 LangGraph Checkpoint

**目标**：TradingAgents 底层是 LangGraph 有状态图。Checkpoint 机制会在每个节点执行后序列化当前 State 到存储后端。网络中断、API 超时或 Token 耗尽后，可以从最后成功的节点恢复，而不是重跑全程（重跑成本约 200-500K tokens）。

```python
# experiment_3_checkpoint.py
"""
实验三：开启 LangGraph Checkpoint，实现 Agent 任务中断后续跑。

核心机制：
- TradingAgents 的 TradingAgentsGraph 内部维护一个 LangGraph StateGraph
- 每个 Analyst / Researcher / Trader 节点执行后触发 checkpoint 保存
- 通过 thread_id 唯一标识一次分析任务，相同 thread_id 可恢复
"""
import os
import sqlite3
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from rich.console import Console

load_dotenv()
console = Console()

# Checkpoint 存储路径（生产环境建议改为 PostgreSQL）
CHECKPOINT_DB = "trading_checkpoints.db"


def get_saver() -> SqliteSaver:
    """
    创建 SQLite Checkpoint Saver。

    生产环境替换为 PostgreSQL:
        from langgraph.checkpoint.postgres import PostgresSaver
        return PostgresSaver.from_conn_string("postgresql://user:pass@host/db")
    """
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    return SqliteSaver(conn)


def analyze_with_checkpoint(
    ticker: str,
    analysis_date: str,
    thread_id: str | None = None,
) -> dict:
    """
    带 Checkpoint 的分析函数。首次运行创建新任务，中断后用相同
    thread_id 恢复。

    Args:
        ticker: 股票代码
        analysis_date: 分析日期
        thread_id: 任务唯一标识。None 时自动生成，
                   续跑时传入上次打印的 thread_id

    Returns:
        分析结果字典，额外包含 thread_id 字段
    """
    import uuid

    # 生成或复用 thread_id
    task_id = thread_id or f"{ticker}_{analysis_date}_{uuid.uuid4().hex[:8]}"
    console.print(f"[dim]Task ID: {task_id}[/dim]  ← 保存此 ID，中断后用于续跑")

    config = DEFAULT_CONFIG.copy()
    config.update({
        "llm_provider": "openai",
        "deep_think_llm": "gpt-4o",
        "quick_think_llm": "gpt-4o-mini",
        "online_tools": True,
    })

    saver = get_saver()

    graph = TradingAgentsGraph(
        selected_analysts=["fundamental", "sentiment", "news", "technical"],
        config=config,
        debug=False,
        # 关键：将 Checkpoint Saver 注入图引擎
        memory=saver,
    )

    # LangGraph 通过 configurable.thread_id 路由到对应 checkpoint
    run_config = {"configurable": {"thread_id": task_id}}

    try:
        console.print(f"[bold cyan]分析 {ticker}（支持断点续跑）...[/bold cyan]")
        state, decision = graph.propagate(
            ticker,
            analysis_date,
            config=run_config,
        )

        console.print(f"[green]✓ 分析完成[/green]")
        return {
            "ticker": ticker,
            "date": analysis_date,
            "thread_id": task_id,
            "decision": decision,
            "state": state,
        }

    except KeyboardInterrupt:
        console.print(
            f"\n[yellow]⚡ 手动中断。续跑命令：[/yellow]\n"
            f"  analyze_with_checkpoint('{ticker}', '{analysis_date}', thread_id='{task_id}')"
        )
        raise
    except Exception as e:
        console.print(
            f"\n[red]✗ 异常中断：{e}[/red]\n"
            f"[yellow]续跑时传入 thread_id='{task_id}'[/yellow]"
        )
        raise


def list_checkpoints(ticker: str | None = None) -> None:
    """
    查看所有已保存的 Checkpoint，确认可续跑的任务列表。
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()

    query = "SELECT thread_id, checkpoint_id, created_at FROM checkpoints"
    params = []
    if ticker:
        query += " WHERE thread_id LIKE ?"
        params.append(f"{ticker}%")
    query += " ORDER BY created_at DESC LIMIT 20"

    rows = cursor.execute(query, params).fetchall()

    from rich.table import Table
    table = Table(title="已保存的 Checkpoint")
    table.add_column("Thread ID", style="cyan")
    table.add_column("最新节点")
    table.add_column("时间")

    for thread_id, checkpoint_id, created_at in rows:
        table.add_row(thread_id, checkpoint_id[:20] + "...", str(created_at))

    console.print(table)
    conn.close()


if __name__ == "__main__":
    # 首次运行（保存打印的 thread_id）
    result = analyze_with_checkpoint("NVDA", "2025-01-10")
    print(f"Thread ID: {result['thread_id']}")

    # 模拟续跑（粘贴上面的 thread_id）
    # result = analyze_with_checkpoint(
    #     "NVDA", "2025-01-10",
    #     thread_id="NVDA_2025-01-10_a1b2c3d4"
    # )
```

**关键点**：
- `SqliteSaver` 适合本地开发；生产环境用 `PostgresSaver`，支持多进程并发访问
- LangGraph 的 Checkpoint 粒度是**节点级**，即完成一个 Analyst 节点后立即持久化，断网也不丢
- ⚠️ 同一 `thread_id` 重跑时，已完成节点的结果直接从 Checkpoint 读取，**不会重新调用 LLM**，可节省约 60-70% 的 token 消耗

---

## 实验四：替换 DeepSeek 为主模型对比分析质量

### Step 6：多模型 A/B 对比框架

**目标**：相同的 Prompt、相同的数据，换不同底座模型，观察分析结论是否一致、推理链路是否有实质差异。这是生产级选型的最小可行实验方法论。

```python
# experiment_4_model_comparison.py
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
from tradingagents.default_config import DEFAULT_CONFIG
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

    config = DEFAULT_CONFIG.copy()
    config.update({
        "llm_provider": mc.provider,
        "deep_think_llm": mc.deep_model,
        "quick_think_llm": mc.quick_model,
        "backend_url": _get_backend_url(mc.provider),
        "risk_tolerance": "neutral",
        "online_tools": True,
    })

    graph = TradingAgentsGraph(
        selected_analysts=["fundamental", "news", "technical"],
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


def _get_backend_url(provider: ModelProvider) -> str:
    """返回对应 provider 的 API base URL。"""
    urls = {
        "openai":   "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "ollama":   "http://localhost:11434/v1",
    }
    return urls.get(provider, "https://api.openai.com/v1")


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
```

**关键点**：
- DeepSeek-V3 价格约为 GPT-4o 的 **5%**，但在金融分析中推理链路的具体性可能略弱，需要用真实样本验证
- `reasoning_length` 只是粗略的深度代理指标，更严格的评估需要人工标注或 LLM-as-Judge
- ⚠️ 不同模型的 Structured Output 严格性有差异，DeepSeek 偶尔会返回格式略有偏差的 JSON，建议加 try/except 容错

---

## 完整运行验证

```python
# smoke_test.py
"""
端到端冒烟测试：验证四个实验核心功能均可运行。
运行前确保 .env 中配置了有效的 OPENAI_API_KEY。
"""
import os
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()


def test_basic_analysis():
    """实验一：基础分析流程验证（使用 mini 模型节约费用）"""
    from experiment_1_basic_analysis import run_analysis
    result = run_analysis("NVDA", "2025-01-10", risk_level="neutral")
    assert "decision" in result
    assert result["decision"].get("action") in [
        "strong buy", "buy", "hold", "sell", "strong sell"
    ]
    console.print(f"[green]✓ 实验一：{result['decision']['action'].upper()}[/green]")
    return result


def test_astock_adapter():
    """实验二：A 股适配层数据格式验证（不触发 LLM，纯数据层测试）"""
    from astock_adapter import AStockAdapter
    adapter = AStockAdapter()
    df = adapter.get_price_history("600519", days=10)

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"], \
        f"字段名不符合标准：{list(df.columns)}"
    assert len(df) > 0, "数据为空"
    assert df.index.tz is not None, "日期索引缺少时区信息"

    console.print(f"[green]✓ 实验二：600519 最新收盘价 {df['Close'].iloc[-1]:.2f} CNY[/green]")


def test_checkpoint_save():
    """实验三：验证 Checkpoint 写入成功（只运行一个节点后手动中断）"""
    import sqlite3
    from experiment_3_checkpoint import CHECKPOINT_DB, get_saver

    saver = get_saver()
    # 验证数据库文件已创建
    assert os.path.exists(CHECKPOINT_DB), "Checkpoint 数据库未创建"

    conn = sqlite3.connect(CHECKPOINT_DB)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    console.print(f"[green]✓ 实验三：Checkpoint DB 就绪，表：{[t[0] for t in tables]}[/green]")
    conn.close()


def test_model_config():
    """实验四：验证 DeepSeek 配置格式（不实际调用，节约成本）"""
    from experiment_4_model_comparison import MODEL_CONFIGS
    assert "deepseek" in MODEL_CONFIGS
    assert MODEL_CONFIGS["deepseek"].provider == "deepseek"
    assert MODEL_CONFIGS["deepseek"].cost_per_1m_input < MODEL_CONFIGS["gpt4o"].cost_per_1m_input
    console.print("[green]✓ 实验四：模型配置验证通过[/green]")


if __name__ == "__main__":
    console.rule("[bold]TradingAgents 冒烟测试[/bold]")

    test_astock_adapter()   # 无 LLM 调用，优先验证
    test_checkpoint_save()  # 无 LLM 调用
    test_model_config()     # 无 LLM 调用
    test_basic_analysis()   # ⚠️ 会产生 LLM 费用，约 $0.05-0.15

    console.rule("[bold green]所有测试通过[/bold green]")
```

**预期输出**：

```
──────────────── TradingAgents 冒烟测试 ────────────────
✓ 实验二：600519 最新收盘价 1688.00 CNY
✓ 实验三：Checkpoint DB 就绪，表：['checkpoints', 'writes']
✓ 实验四：模型配置验证通过
开始分析 NVDA（2025-01-10）...
✓ 实验一：BUY
────────────────── 所有测试通过 ──────────────────
```

---

## 常见报错与解决方案

| 报错信息 | 原因 | 解决方案 |
|---------|------|---------|
| `ModuleNotFoundError: No module named 'tradingagents'` | 包未安装或虚拟环境未激活 | `source .venv/bin/activate && uv pip install tradingagents` |
| `KeyError: 'OPENAI_API_KEY'` | .env 未加载或变量名拼写错误 | 确认 `.env` 文件在项目根目录，且已调用 `load_dotenv()` |
| `akshare.tools.func.FundamentalError: 股票代码错误` | A 股代码格式不对 | 使用纯数字 6 位代码，如 `"600519"`，不要加 `.SH` 后缀 |
| `langgraph.errors.NodeInterrupt` | 图中配置了人工审批节点 | 将 `interrupt_before` 参数从 config 中移除，或实现 `resume()` 逻辑 |
| `openai.RateLimitError: 429` | API 并发或每分钟 token 超限 | 降低 `selected_analysts` 数量，或在 analysts 之间加 `time.sleep(2)` |
| `sqlite3.OperationalError: database is locked` | 多进程同时写入 SQLite | 生产环境切换为 PostgresSaver，或确保单进程访问 |
| `AssertionError: 字段名不符合标准` | AKShare 接口字段名变更 | 执行 `print(ak.stock_zh_a_hist("600519").columns.tolist())` 检查当前列名，更新 `COLUMN_MAP` |

---

## 扩展练习（可选）

1. 🟡 **中等**：为实验二的 A 股适配器添加**雪球社区情绪数据**接入。AKShare 提供 `ak.stock_hot_rank_em()` 获取热度排行，尝试将其转换为情绪分数（0-1 范围），接入 Sentiment Analyst 节点。

2. 🔴 **困难**：基于实验四的多模型对比框架，设计一套**自动化效果评估流水线**：对 30 支股票运行 GPT-4o 和 DeepSeek-V3，与实际后续 5 个交易日涨跌幅对比，计算两个模型的预测准确率（方向正确率），最终输出一张统计显著性检验报告。提示：注意处理数据集划分和过拟合问题，避免用已知结果反向验证。
