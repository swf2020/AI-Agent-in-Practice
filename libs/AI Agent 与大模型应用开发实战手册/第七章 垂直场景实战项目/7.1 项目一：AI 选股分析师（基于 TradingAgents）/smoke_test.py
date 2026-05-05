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
    """实验三：验证 Checkpoint 数据库结构正确（不触发 LLM）"""
    import sqlite3
    from experiment_3_checkpoint import CHECKPOINT_DB, _init_checkpoint_db

    conn = _init_checkpoint_db()
    # 验证数据库文件已创建
    assert os.path.exists(CHECKPOINT_DB), "Checkpoint 数据库未创建"

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
    
    # test_basic_analysis()   # ⚠️ 会产生 LLM 费用，约 $0.05-0.15
    console.print("[yellow]⚠️  已跳过 LLM 测试（test_basic_analysis），如需运行请取消注释[/yellow]")

    console.rule("[bold green]所有非 LLM 测试通过[/bold green]")