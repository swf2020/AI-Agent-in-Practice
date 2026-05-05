"""
SQL 安全守卫：使用 sqlparse 进行 AST 级别的语句类型验证
为何不用简单的 startswith("SELECT")：
  攻击者可以绕过：WITH evil AS (DELETE ...) SELECT 1
  sqlparse 解析后能拿到真实的 statement type
"""
import sqlparse
from sqlparse.tokens import Keyword, DDL, DML


class SQLSecurityError(ValueError):
    """SQL 安全违规异常，区别于普通的 ValueError"""
    pass


# 白名单：只允许这些 DML 操作
ALLOWED_STATEMENT_TYPES = {"SELECT"}

# 黑名单关键字（用于双重检查，防止 sqlparse 解析漏洞）
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
    "ALTER", "TRUNCATE", "REPLACE", "MERGE", "EXEC",
    "EXECUTE", "GRANT", "REVOKE", "ATTACH", "DETACH",
}


def validate_sql(sql: str) -> str:
    """
    验证 SQL 语句安全性，返回规范化后的 SQL。
    
    Args:
        sql: 待验证的 SQL 字符串
        
    Returns:
        规范化后的 SQL（去除首尾空白，统一大小写关键字）
        
    Raises:
        SQLSecurityError: 包含非 SELECT 操作时抛出
        ValueError: SQL 为空或无法解析时抛出
    """
    sql = sql.strip()
    if not sql:
        raise ValueError("SQL 不能为空")

    # ── 第一层：sqlparse AST 解析 ────────────────────────
    parsed = sqlparse.parse(sql)
    if not parsed:
        raise ValueError("无法解析 SQL 语句")

    for statement in parsed:
        stmt_type = statement.get_type()
        # get_type() 返回 None 表示无法识别，拒绝放行（fail-safe 原则）
        if stmt_type is None or stmt_type.upper() not in ALLOWED_STATEMENT_TYPES:
            raise SQLSecurityError(
                f"安全拒绝：检测到非 SELECT 操作（类型：{stmt_type}）。"
                f"本 Server 仅允许只读查询。"
            )

    # ── 第二层：关键字黑名单兜底 ────────────────────────
    # 防御 sqlparse 对复杂 CTE 解析不准确的边界情况
    tokens = sqlparse.parse(sql)[0].flatten()
    for token in tokens:
        if token.ttype in (Keyword, DDL, DML):
            if token.normalized.upper() in FORBIDDEN_KEYWORDS:
                raise SQLSecurityError(
                    f"安全拒绝：检测到禁止关键字 '{token.normalized}'。"
                )

    return sql


def sanitize_identifier(name: str) -> str:
    """
    验证表名/列名是否为合法标识符，防止 SQL 注入。
    仅允许字母、数字、下划线，且不能以数字开头。
    """
    import re
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        raise ValueError(f"非法标识符：'{name}'。只允许字母、数字和下划线。")
    return name