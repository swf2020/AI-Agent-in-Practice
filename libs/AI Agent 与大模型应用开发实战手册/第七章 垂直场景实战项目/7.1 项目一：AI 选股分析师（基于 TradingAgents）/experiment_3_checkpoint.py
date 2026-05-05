"""
实验三：开启 LangGraph Checkpoint，实现 Agent 任务中断后续跑。

核心机制：
- TradingAgents 的 TradingAgentsGraph 内部维护一个 LangGraph StateGraph
- 每个 Analyst / Researcher / Trader 节点执行后触发 checkpoint 保存
- 通过 thread_id 唯一标识一次分析任务，相同 thread_id 可恢复

⚠️ tradingagents 0.3.1 暂不支持 memory/checkpoint 注入。
本实验改用 SQLite 手动记录分析状态，验证 checkpoint 概念。
"""
import os
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.config import TradingAgentsConfig
from rich.console import Console

load_dotenv()
console = Console()

# Checkpoint 存储路径（生产环境建议改为 PostgreSQL）
CHECKPOINT_DB = "trading_checkpoints.db"


def _init_checkpoint_db() -> sqlite3.Connection:
    """初始化 checkpoint 数据库"""
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT PRIMARY KEY,
            ticker TEXT,
            analysis_date TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result_json TEXT
        )
    """)
    conn.commit()
    return conn


def get_saver() -> sqlite3.Connection:
    """
    获取 Checkpoint 数据库连接。

    注：tradingagents 0.3.1 不支持 SqliteSaver 注入，
    此处返回连接供手动保存结果使用。
    """
    return _init_checkpoint_db()


def _make_config() -> TradingAgentsConfig:
    """创建配置"""
    return TradingAgentsConfig(
        llm_provider="openai",
        deep_think_llm="gpt-4o",
        quick_think_llm="gpt-4o-mini",
        reasoning_effort="medium",
        max_debate_rounds=3,
        max_risk_discuss_rounds=3,
        max_recur_limit=100,
    )


def _save_checkpoint(conn, thread_id, ticker, analysis_date, status, result=None):
    """手动保存 checkpoint"""
    conn.execute(
        "INSERT OR REPLACE INTO checkpoints (thread_id, ticker, analysis_date, status, result_json) VALUES (?, ?, ?, ?, ?)",
        (thread_id, ticker, analysis_date, status, json.dumps(result) if result else None),
    )
    conn.commit()


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

    config = _make_config()

    conn = get_saver()

    # 检查是否已有 checkpoint
    row = conn.execute(
        "SELECT status, result_json FROM checkpoints WHERE thread_id = ?",
        (task_id,)
    ).fetchone()
    if row and row[0] == "completed" and row[1]:
        console.print(f"[yellow]⚡ 找到已完成的 checkpoint，直接返回[/yellow]")
        result = json.loads(row[1])
        result["thread_id"] = task_id
        return result

    try:
        console.print(f"[bold cyan]分析 {ticker}（支持断点续跑）...[/bold cyan]")

        graph = TradingAgentsGraph(
            selected_analysts=["fundamentals", "sentiment", "news", "technical"],
            config=config,
            debug=False,
        )

        state, decision = graph.propagate(ticker, analysis_date)

        result = {
            "ticker": ticker,
            "date": analysis_date,
            "thread_id": task_id,
            "decision": decision,
            "state": state,
        }

        _save_checkpoint(conn, task_id, ticker, analysis_date, "completed", result)
        console.print(f"[green]✓ 分析完成，checkpoint 已保存[/green]")
        return result

    except KeyboardInterrupt:
        _save_checkpoint(conn, task_id, ticker, analysis_date, "interrupted")
        console.print(
            f"\n[yellow]⚡ 手动中断。续跑命令：[/yellow]\n"
            f"  analyze_with_checkpoint('{ticker}', '{analysis_date}', thread_id='{task_id}')"
        )
        raise
    except Exception as e:
        _save_checkpoint(conn, task_id, ticker, analysis_date, "error")
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

    query = "SELECT thread_id, status, created_at FROM checkpoints"
    params = []
    if ticker:
        query += " WHERE ticker = ?"
        params.append(ticker)
    query += " ORDER BY created_at DESC LIMIT 20"

    rows = cursor.execute(query, params).fetchall()

    from rich.table import Table
    table = Table(title="已保存的 Checkpoint")
    table.add_column("Thread ID", style="cyan")
    table.add_column("状态")
    table.add_column("时间")

    for thread_id, status, created_at in rows:
        table.add_row(thread_id, status, str(created_at))

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