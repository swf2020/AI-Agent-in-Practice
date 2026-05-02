"""
端到端冒烟测试：通过 MCP Python SDK 直接调用 Server
不依赖 Claude Desktop，适合 CI 环境
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_smoke_test():
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env={"DB_PATH": "data/sample.db"},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=" * 50)
            print("🧪 MCP Database Server 冒烟测试")
            print("=" * 50)

            # ── 测试 1：列出所有工具 ──────────────────────
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"\n✅ 工具列表：{tool_names}")
            assert set(tool_names) == {"execute_query", "describe_table", "get_sample_data"}, \
                f"工具列表不符合预期：{tool_names}"

            # ── 测试 2：列出所有 Resources ───────────────
            resources = await session.list_resources()
            resource_uris = [str(r.uri) for r in resources.resources]
            print(f"✅ Resources：{resource_uris}")
            assert "db://schema" in resource_uris

            # ── 测试 3：读取 Schema Resource ─────────────
            schema_result = await session.read_resource("db://schema")
            schema_text = schema_result.contents[0].text
            schema_data = json.loads(schema_text)
            tables_in_schema = list(schema_data["schema"].keys())
            print(f"✅ Schema 包含表：{tables_in_schema}")
            assert "users" in tables_in_schema and "orders" in tables_in_schema

            # ── 测试 4：describe_table Tool ──────────────
            desc_result = await session.call_tool("describe_table", {"table_name": "orders"})
            desc_data = json.loads(desc_result.content[0].text)
            col_names = [c["column"] for c in desc_data["columns"]]
            print(f"✅ orders 表字段：{col_names}")
            assert "order_id" in col_names and "total_amount" in col_names

            # ── 测试 5：execute_query 正常查询 ───────────
            query_result = await session.call_tool(
                "execute_query",
                {"sql": "SELECT COUNT(*) as user_count FROM users"}
            )
            query_data = json.loads(query_result.content[0].text)
            user_count = query_data["rows"][0]["user_count"]
            print(f"✅ 用户总数：{user_count}")
            assert user_count == 50, f"预期 50 个用户，实际 {user_count}"

            # ── 测试 6：安全拦截验证 ─────────────────────
            evil_result = await session.call_tool(
                "execute_query",
                {"sql": "DELETE FROM users WHERE 1=1"}
            )
            evil_data = json.loads(evil_result.content[0].text)
            print(f"✅ 危险 SQL 被拦截：{evil_data['error']}")
            assert evil_data["error"] == "SECURITY_REJECTED"

            # ── 测试 7：复杂业务查询 ─────────────────────
            biz_sql = """
                SELECT u.region,
                       COUNT(DISTINCT o.order_id) as order_count,
                       ROUND(SUM(o.total_amount), 2) as total_revenue
                FROM users u
                JOIN orders o ON u.user_id = o.user_id
                WHERE o.status = 'done'
                GROUP BY u.region
                ORDER BY total_revenue DESC
            """
            biz_result = await session.call_tool("execute_query", {"sql": biz_sql})
            biz_data = json.loads(biz_result.content[0].text)
            print(f"✅ 各地区完成订单统计（{biz_data['row_count']} 行）：")
            for row in biz_data["rows"]:
                print(f"   {row['region']}: {row['order_count']} 笔, ¥{row['total_revenue']}")

            print("\n🎉 所有测试通过！")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())