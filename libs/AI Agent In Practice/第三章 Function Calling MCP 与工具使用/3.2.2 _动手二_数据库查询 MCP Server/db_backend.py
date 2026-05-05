"""
数据库后端抽象层：支持 SQLite（本地开发）和 PostgreSQL（生产）双模式
通过环境变量 DATABASE_URL 切换，不需要改代码
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Generator, Any

# 结果截断阈值（行数）：防止意外的全表扫描把几十万行数据塞入 LLM 上下文
MAX_ROWS = 100
# 单个字段值的最大字符数：防止 TEXT 字段里的长文本撑爆上下文
MAX_FIELD_LENGTH = 500


def _truncate_value(v: Any) -> Any:
    """截断过长的字段值，并标注截断标记方便 LLM 理解"""
    if isinstance(v, str) and len(v) > MAX_FIELD_LENGTH:
        return v[:MAX_FIELD_LENGTH] + f"... [截断，原始长度 {len(v)} 字符]"
    return v


def format_query_result(rows: list[dict], total_fetched: int) -> str:
    """
    将查询结果格式化为 JSON 字符串，附带元信息。
    
    元信息对 LLM 很重要：知道"只返回了 100 行中的 100 行"
    有助于它告知用户结果可能被截断，而不是编造"共有100条记录"。
    """
    result = {
        "row_count": len(rows),
        "truncated": total_fetched > MAX_ROWS,
        "rows": [{k: _truncate_value(v) for k, v in row.items()} for row in rows],
    }
    if result["truncated"]:
        result["note"] = f"查询结果已截断至 {MAX_ROWS} 行，请添加 LIMIT 子句获取精确结果"
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


# ── SQLite 后端 ──────────────────────────────────────────

@contextmanager
def get_sqlite_conn(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """SQLite 连接上下文管理器，自动处理关闭和异常回滚"""
    conn = sqlite3.connect(db_path, timeout=10)  # 10秒锁等待超时
    conn.row_factory = sqlite3.Row  # 让结果支持按列名访问
    conn.execute("PRAGMA query_only = ON")  # 数据库级只读保护（双重保险）
    try:
        yield conn
    finally:
        conn.close()


def sqlite_execute(db_path: str, sql: str) -> str:
    """在 SQLite 上执行已验证的 SELECT 查询"""
    with get_sqlite_conn(db_path) as conn:
        cursor = conn.execute(sql)
        # fetchmany 避免把百万行数据全部加载到内存
        rows = [dict(r) for r in cursor.fetchmany(MAX_ROWS + 1)]
        total = len(rows)
        return format_query_result(rows[:MAX_ROWS], total)


def sqlite_get_schema(db_path: str) -> dict:
    """提取 SQLite 完整 Schema：表名 → 列信息列表"""
    with get_sqlite_conn(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        schema = {}
        for (table_name,) in tables:
            cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            fks = conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
            fk_map = {fk[3]: f"{fk[2]}.{fk[4]}" for fk in fks}  # col → ref_table.ref_col

            schema[table_name] = [
                {
                    "column": col[1],
                    "type": col[2],
                    "nullable": not col[3],
                    "default": col[4],
                    "primary_key": bool(col[5]),
                    "references": fk_map.get(col[1]),  # 外键引用，帮助 LLM 理解关联关系
                }
                for col in cols
            ]
        return schema


def sqlite_describe_table(db_path: str, table_name: str) -> dict:
    """获取单表的详细结构信息"""
    schema = sqlite_get_schema(db_path)
    if table_name not in schema:
        available = list(schema.keys())
        raise ValueError(f"表 '{table_name}' 不存在。可用的表：{available}")
    return {"table": table_name, "columns": schema[table_name]}


def sqlite_get_sample(db_path: str, table_name: str, limit: int) -> str:
    """获取样例数据（复用 execute，但 table_name 需要额外验证）"""
    from db_guard import sanitize_identifier
    safe_name = sanitize_identifier(table_name)
    safe_limit = min(max(1, limit), 20)  # 样例数据强制限制 1-20 行
    sql = f"SELECT * FROM {safe_name} LIMIT {safe_limit}"
    with get_sqlite_conn(db_path) as conn:
        cursor = conn.execute(sql)
        rows = [dict(r) for r in cursor.fetchall()]
        return format_query_result(rows, len(rows))