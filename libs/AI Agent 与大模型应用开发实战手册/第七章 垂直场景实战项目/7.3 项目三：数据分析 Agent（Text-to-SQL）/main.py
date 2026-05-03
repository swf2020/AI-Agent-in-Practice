from dotenv import load_dotenv
import os

from schema_manager import SchemaManager
from sql_generator import SQLGenerator, Dialect
from sql_executor import SQLExecutor, SelfCorrectingExecutor
from visualizer import DataVisualizer

load_dotenv()


def main():
    print("=== 数据分析 Agent（Text-to-SQL）===")
    
    db_path = os.getenv("DB_PATH", "ecommerce.db")
    
    schema_manager = SchemaManager(db_path)
    sql_generator = SQLGenerator(model="gpt-4o-mini", dialect=Dialect.SQLITE)
    executor = SQLExecutor(db_path)
    visualizer = DataVisualizer()
    
    questions = [
        "各城市的订单总金额是多少？",
        "过去三个月各品类销售额趋势",
        "VIP 用户的平均订单金额是多少？",
    ]
    
    for question in questions:
        print(f"\n---\n问题：{question}")
        
        tables = schema_manager.retrieve_relevant_tables(question)
        schema_prompt = schema_manager.format_schema_prompt(tables)
        
        print("🔧 生成 SQL...")
        sql_result = sql_generator.generate(question, schema_prompt)
        print(f"SQL：{sql_result.sql}")
        
        print("🔍 执行查询...")
        correcting_executor = SelfCorrectingExecutor(executor, sql_generator, schema_prompt)
        exec_result, final_sql, retries = correcting_executor.execute_with_correction(question, sql_result.sql)
        
        if exec_result.success:
            print(f"✅ 执行成功（重试 {retries} 次），返回 {exec_result.row_count} 行")
            print("📊 数据：")
            print(exec_result.data)
            
            print("\n📈 可视化...")
            viz_result = visualizer.visualize(question, final_sql, exec_result.data)
            print(f"摘要：{viz_result['summary']}")
        else:
            print(f"❌ 执行失败：{exec_result.error}")


if __name__ == "__main__":
    main()