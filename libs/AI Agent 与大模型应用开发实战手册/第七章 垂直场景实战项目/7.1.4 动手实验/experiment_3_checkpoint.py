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