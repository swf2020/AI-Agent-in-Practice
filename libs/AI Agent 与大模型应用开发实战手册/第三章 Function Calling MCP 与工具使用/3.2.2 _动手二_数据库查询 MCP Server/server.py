"""
数据库查询 MCP Server
支持 Claude 通过自然语言查询 SQLite / PostgreSQL 数据库

运行方式：
  python server.py                    # stdio 模式（配合 Claude Desktop）
  python server.py --transport sse    # SSE 模式（配合 Web 客户端，默认端口 8000）
"""
import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from db_backend import (
    sqlite_execute,
    sqlite_get_schema,
    sqlite_describe_table,
    sqlite_get_sample,
)
from db_guard import SQLSecurityError, validate_sql

# ── 配置 ─────────────────────────────────────────────────
# 优先读环境变量，方便在不同环境（本地/容器/云函数）切换数据库路径
DB_PATH = os.environ.get("DB_PATH", "data/sample.db")

# 验证数据库文件存在（早失败原则：启动时就报错，而不是在第一次请求时才发现）
if not Path(DB_PATH).exists():
    sys.exit(f"❌ 数据库文件不存在：{DB_PATH}\n请先运行 python scripts/create_sample_db.py")

# ── 创建 MCP Server 实例 ──────────────────────────────────
mcp = FastMCP(
    name="database-server",
    # Server 的 instructions 会被注入到 Claude 的系统提示中
    # 告诉 Claude 这个 Server 的能力边界和使用约定
    instructions="""
    你有权限查询一个电商业务数据库（SQLite）。
    
    工作流程建议：
    1. 先通过 db://schema Resource 了解完整表结构
    2. 不确定字段含义时，用 describe_table 获取详细说明
    3. 不熟悉数据分布时，用 get_sample_data 查看样例
    4. 用 execute_query 执行最终的 SELECT 查询
    
    注意事项：
    - 仅支持 SELECT 查询，写操作会被拒绝
    - 结果最多返回 100 行，超出会被截断并提示
    - 涉及金额计算时注意使用 order_items.unit_price（快照价格）而非 products.price（当前价格）
    """,
)


# ═══════════════════════════════════════════════════════════
# TOOLS：Claude 主动调用，执行具体操作
# ═══════════════════════════════════════════════════════════

@mcp.tool()
def execute_query(sql: str) -> str:
    """
    执行 SQL SELECT 查询并返回结果。
    
    Args:
        sql: 标准 SQL SELECT 语句。支持 JOIN、子查询、聚合函数、CTE（WITH 子句）。
             示例：SELECT u.username, COUNT(o.order_id) as order_count
                   FROM users u LEFT JOIN orders o ON u.user_id = o.user_id
                   GROUP BY u.user_id ORDER BY order_count DESC LIMIT 10
    
    Returns:
        JSON 格式的查询结果，包含 row_count、truncated 标志和 rows 数组。
        若结果超过 100 行会被截断，truncated 字段为 true。
    """
    try:
        validated_sql = validate_sql(sql)
    except SQLSecurityError as e:
        # 安全拒绝：返回明确的错误信息让 Claude 理解原因
        return json.dumps({"error": "SECURITY_REJECTED", "message": str(e)}, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"error": "INVALID_SQL", "message": str(e)}, ensure_ascii=False)

    try:
        return sqlite_execute(DB_PATH, validated_sql)
    except Exception as e:
        # 数据库执行错误：把原始报错返回给 Claude，让它自行修正 SQL
        # 这是 Text-to-SQL 自修正循环的关键——错误信息本身就是上下文
        return json.dumps({
            "error": "EXECUTION_ERROR",
            "message": str(e),
            "hint": "请检查表名、列名是否正确，可以用 describe_table 确认表结构",
        }, ensure_ascii=False)


@mcp.tool()
def describe_table(table_name: str) -> str:
    """
    获取指定表的详细结构信息，包括列名、数据类型、是否可空、主键和外键关系。
    
    当你不确定某个表有哪些字段，或需要了解表间关联关系时使用此工具。
    
    Args:
        table_name: 表名（区分大小写）。如：users、orders、order_items、products
    
    Returns:
        JSON 格式的表结构描述，包含所有列的详细信息和外键引用关系。
    """
    try:
        result = sqlite_describe_table(DB_PATH, table_name)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except ValueError as e:
        return json.dumps({"error": "TABLE_NOT_FOUND", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def get_sample_data(table_name: str, limit: int = 5) -> str:
    """
    获取指定表的样例数据，帮助理解数据分布和字段取值范围。
    
    适用场景：
    - 不确定某个枚举字段有哪些可能值（如 status、region、category）
    - 需要了解数据的大致规模和质量
    - 构造 WHERE 条件前先看看实际数据长什么样
    
    Args:
        table_name: 表名
        limit: 返回行数，范围 1-20，默认 5
    
    Returns:
        JSON 格式的样例数据
    """
    try:
        return sqlite_get_sample(DB_PATH, table_name, limit)
    except ValueError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# RESOURCES：被动注入，Claude 主动读取作为背景知识
# ═══════════════════════════════════════════════════════════

@mcp.resource("db://schema")
def get_full_schema() -> str:
    """
    返回完整数据库 Schema，供 Claude 在生成 SQL 前参考。
    
    这是一个 Resource 而非 Tool：Claude 在对话开始时会自动读取它，
    不需要用户每次都在 Prompt 里粘贴表结构。
    
    Returns:
        JSON 格式的完整 Schema，结构为 {表名: [{列信息}]}
    """
    schema = sqlite_get_schema(DB_PATH)
    # 附加 Schema 使用说明，引导 LLM 正确理解
    result = {
        "database_type": "SQLite",
        "database_path": DB_PATH,
        "schema": schema,
        "important_notes": [
            "order_items.unit_price 是下单时的快照价格，用于历史金额计算",
            "products.price 是当前售价，可能与历史订单价格不同",
            "orders.status 枚举值：pending/paid/shipped/done/cancelled",
            "users.is_vip: 0=普通用户, 1=VIP用户",
        ],
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.resource("db://tables")
def get_table_list() -> str:
    """
    返回数据库中所有表的名称列表，适合快速了解数据库结构。
    
    Returns:
        JSON 格式的表名列表及简要说明
    """
    schema = sqlite_get_schema(DB_PATH)
    # 为每张表附加简要业务说明，帮助 LLM 快速定位目标表
    table_descriptions = {
        "users": "用户信息，包含大区、VIP等级",
        "products": "商品信息，包含分类、价格、库存",
        "orders": "订单主表，包含状态、总金额、时间",
        "order_items": "订单明细，记录每笔订单的商品、数量、成交价",
    }
    result = {
        "tables": [
            {
                "name": t,
                "description": table_descriptions.get(t, ""),
                "column_count": len(cols),
            }
            for t, cols in schema.items()
        ]
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        transport = sys.argv[idx + 1]

    print(f"🚀 Database MCP Server 启动中（transport={transport}, db={DB_PATH}）", file=sys.stderr)
    mcp.run(transport=transport)