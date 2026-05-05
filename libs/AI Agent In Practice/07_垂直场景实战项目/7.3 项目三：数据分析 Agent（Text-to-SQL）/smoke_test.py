from dotenv import load_dotenv
import os
from unittest.mock import patch, MagicMock

from core_config import get_litellm_id, get_api_key, get_base_url
from db_setup import create_demo_database
from schema_manager import SchemaManager
from sql_generator import SQLGenerator, Dialect, SQLGenerationResult
from sql_executor import SQLExecutor, SelfCorrectingExecutor
from visualizer import DataVisualizer
from visualizer import ChartDecision, ChartType

load_dotenv()


def _make_mock_client():
    """创建 mock OpenAI client，返回结构化输出的 fake response"""
    mock = MagicMock()
    mock.beta.chat.completions.parse.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                parsed=SQLGenerationResult(
                    sql="SELECT u.city AS 城市, ROUND(SUM(o.total_amount), 2) AS 订单总金额 "
                        "FROM orders o JOIN users u ON o.user_id = u.user_id "
                        "WHERE o.status = 'completed' GROUP BY u.city ORDER BY 订单总金额 DESC",
                    explanation="按城市分组汇总已完成订单的总金额",
                    confidence=0.95,
                    ambiguities=[],
                )
            )
        )]
    )
    return mock


def run_smoke_test():
    print("=== 数据分析 Agent - 冒烟测试 ===")

    print("\n1. 创建演示数据库...")
    create_demo_database("test_ecommerce.db")

    print("\n2. Schema 管理器测试...")
    schema_manager = SchemaManager("test_ecommerce.db")
    print(f"   ✓ 加载 {len(schema_manager.tables)} 张表")

    print("\n3. SQL 生成测试（Mock LLM）...")
    mock_client = _make_mock_client()

    with patch("sql_generator.get_openai_client", return_value=mock_client):
        sql_generator = SQLGenerator(
            model=get_litellm_id(), dialect=Dialect.SQLITE,
            api_key=get_api_key(), base_url=get_base_url(),
        )
        tables = schema_manager.retrieve_relevant_tables("各城市订单金额")
        schema_prompt = schema_manager.format_schema_prompt(tables)
        result = sql_generator.generate("各城市的订单总金额是多少？", schema_prompt)
        print(f"   ✓ SQL 生成成功：{result.sql[:60]}...")

        print("\n4. SQL 执行测试...")
        executor = SQLExecutor("test_ecommerce.db")
        correcting_executor = SelfCorrectingExecutor(executor, sql_generator, schema_prompt)
        exec_result, final_sql, retries = correcting_executor.execute_with_correction(
            "各城市订单金额", result.sql
        )
        print(f"   ✓ 执行成功（重试 {retries} 次）")
        print(f"   返回 {exec_result.row_count} 行数据")

        print("\n5. 可视化测试（Mock LLM）...")
        mock_viz_client = MagicMock()
        mock_viz_client.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    parsed=ChartDecision(
                        chart_type=ChartType.BAR,
                        x_column="城市",
                        y_column="订单总金额",
                        color_column=None,
                        reasoning="分类对比数据，适合柱状图",
                        title="各城市订单总金额对比",
                    )
                )
            )]
        )
        mock_viz_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="各城市中，北京订单总金额最高。"))]
        )

        visualizer = DataVisualizer(
            output_dir="test_charts",
            client=mock_viz_client,
            model="gpt-4o-mini",
        )
        viz_result = visualizer.visualize("各城市订单金额", final_sql, exec_result.data)
        print(f"   ✓ 图表类型：{viz_result['chart_type']}")
        print(f"   ✓ 摘要：{viz_result['summary']}")

    print("\n=== 测试完成 ✓ ===")

    os.remove("test_ecommerce.db")


if __name__ == "__main__":
    run_smoke_test()
