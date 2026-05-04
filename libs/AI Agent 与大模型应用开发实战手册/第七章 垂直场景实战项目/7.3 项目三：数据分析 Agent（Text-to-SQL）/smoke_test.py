from dotenv import load_dotenv
import os

from core_config import get_litellm_id, get_api_key, get_base_url
from db_setup import create_demo_database
from schema_manager import SchemaManager
from sql_generator import SQLGenerator, Dialect
from sql_executor import SQLExecutor, SelfCorrectingExecutor
from visualizer import DataVisualizer

load_dotenv()


def run_smoke_test():
    print("=== 数据分析 Agent - 冒烟测试 ===")

    print("\n1. 创建演示数据库...")
    create_demo_database("test_ecommerce.db")

    print("\n2. Schema 管理器测试...")
    schema_manager = SchemaManager("test_ecommerce.db")
    print(f"   ✓ 加载 {len(schema_manager.tables)} 张表")

    print("\n3. SQL 生成测试...")
    sql_generator = SQLGenerator(
        model=get_litellm_id(), dialect=Dialect.SQLITE,
        api_key=get_api_key(), base_url=get_base_url(),
    )
    tables = schema_manager.retrieve_relevant_tables("各城市订单金额")
    schema_prompt = schema_manager.format_schema_prompt(tables)
    result = sql_generator.generate("各城市的订单总金额是多少？", schema_prompt)
    print(f"   ✓ SQL 生成成功：{result.sql[:50]}...")

    print("\n4. SQL 执行测试...")
    executor = SQLExecutor("test_ecommerce.db")
    correcting_executor = SelfCorrectingExecutor(executor, sql_generator, schema_prompt)
    exec_result, final_sql, retries = correcting_executor.execute_with_correction(
        "各城市订单金额", result.sql
    )
    print(f"   ✓ 执行成功（重试 {retries} 次）")
    print(f"   返回 {exec_result.row_count} 行数据")

    print("\n5. 可视化测试...")
    visualizer = DataVisualizer(
        output_dir="test_charts",
        client=sql_generator.client,
        model=sql_generator.model,
    )
    viz_result = visualizer.visualize("各城市订单金额", final_sql, exec_result.data)
    print(f"   ✓ 图表类型：{viz_result['chart_type']}")
    print(f"   ✓ 摘要：{viz_result['summary']}")

    print("\n=== 测试完成 ✓ ===")

    os.remove("test_ecommerce.db")


if __name__ == "__main__":
    run_smoke_test()
