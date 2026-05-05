"""Text-to-SQL 数据库查询工具"""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from tools.base import BaseTool


class TextToSQLTool(BaseTool):
    """将自然语言查询转换为 SQL 并安全执行。
    
    关键设计决策：
    1. 只读连接：使用只有 SELECT 权限的数据库用户
    2. Schema 注入：将表结构信息嵌入工具 description，让 LLM 知道"有什么表"
    3. 行数限制：强制 LIMIT，避免把整张大表返回给 LLM
    """

    MAX_ROWS = 50       # 返回给 LLM 的最大行数
    MAX_COL_LEN = 100   # 单个字段值的最大字符数（防止 blob 字段撑爆 context）

    def __init__(
        self,
        db_url: str,
        allowed_tables: list[str] | None = None,  # None 表示允许所有表
        max_rows: int = MAX_ROWS,
    ) -> None:
        # ⚠️ 生产中应使用只读账号的连接串，如 postgresql://readonly_user:pwd@host/db
        self._engine: Engine = create_engine(db_url)
        self._allowed_tables = set(allowed_tables) if allowed_tables else None
        self._max_rows = max_rows
        self._schema_description = self._build_schema_description()

    def _build_schema_description(self) -> str:
        """提取数据库 Schema 信息，生成 LLM 可理解的文本描述。
        
        这是 Text-to-SQL 质量的关键：Schema 信息越详细（含注释、样例值），
        LLM 生成的 SQL 越准确。
        """
        from sqlalchemy import inspect

        inspector = inspect(self._engine)
        tables = inspector.get_table_names()

        if self._allowed_tables:
            tables = [t for t in tables if t in self._allowed_tables]

        schema_parts: list[str] = []
        for table in tables:
            columns = inspector.get_columns(table)
            col_desc = ", ".join(
                f"{col['name']} ({str(col['type'])})" for col in columns
            )
            schema_parts.append(f"表名: {table}\n字段: {col_desc}")

        return "\n\n".join(schema_parts)

    @property
    def name(self) -> str:
        return "database_query"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    f"查询数据库中的结构化数据。当用户询问具体的数据统计、"
                    f"记录查找、数据分析时使用。\n\n"
                    f"数据库结构如下：\n{self._schema_description}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "natural_language_query": {
                            "type": "string",
                            "description": "用户的自然语言查询需求，如'查找销售额最高的前5个产品'",
                        },
                        "sql": {
                            "type": "string",
                            "description": (
                                "要执行的 SQL 查询语句。"
                                "必须是只读 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DROP。"
                                f"务必加 LIMIT，最多返回 {self.MAX_ROWS} 行。"
                            ),
                        },
                    },
                    "required": ["natural_language_query", "sql"],
                },
            },
        }

    def _validate_sql(self, sql: str) -> None:
        """防御性 SQL 验证：拦截危险操作。
        
        注意：这只是第一道防线，根本防线是数据库用户的权限控制。
        不要依赖正则来保证安全——有经验的攻击者可以绕过。
        """
        sql_upper = sql.upper().strip()

        # 检测写操作关键字
        dangerous_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
            "ALTER", "TRUNCATE", "EXEC", "EXECUTE", "--", ";"
        ]
        for keyword in dangerous_keywords:
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                raise ValueError(
                    f"禁止执行 '{keyword}' 操作，本工具仅支持 SELECT 查询"
                )

        if not sql_upper.lstrip().startswith("SELECT"):
            raise ValueError("SQL 必须以 SELECT 开头")

    def _format_results(self, rows: list[dict], columns: list[str]) -> str:
        """将查询结果格式化为 LLM 易读的文本表格。"""
        if not rows:
            return "查询成功，结果为空（0 行）"

        # 截断过长的字段值
        truncated_rows = []
        for row in rows:
            truncated_row = {}
            for k, v in row.items():
                v_str = str(v) if v is not None else "NULL"
                truncated_row[k] = v_str[:self.MAX_COL_LEN] + "..." if len(v_str) > self.MAX_COL_LEN else v_str
            truncated_rows.append(truncated_row)

        # 简单 Markdown 表格格式（LLM 更容易解析）
        header = " | ".join(columns)
        separator = " | ".join(["---"] * len(columns))
        data_rows = [" | ".join(str(row.get(col, "")) for col in columns) for row in truncated_rows]

        result = f"查询返回 {len(rows)} 行数据：\n\n"
        result += f"| {header} |\n| {separator} |\n"
        result += "\n".join(f"| {row} |" for row in data_rows)

        if len(rows) >= self._max_rows:
            result += f"\n\n⚠️ 结果已截断至 {self._max_rows} 行，实际可能有更多数据"

        return result

    def run(self, natural_language_query: str, sql: str) -> str:
        """验证并执行 SQL，返回格式化结果。"""
        try:
            self._validate_sql(sql)
        except ValueError as e:
            return f"SQL 验证失败：{e}"

        # 强制添加 LIMIT（即使 LLM 已经写了，再加一层保险）
        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + f" LIMIT {self._max_rows}"

        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(sql))
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchmany(self._max_rows)]
                return self._format_results(rows, columns)
        except Exception as e:
            # 将数据库错误返回给 LLM，让它自行修正 SQL
            return f"SQL 执行失败：{type(e).__name__}: {str(e)[:300]}\n原始 SQL：{sql}"