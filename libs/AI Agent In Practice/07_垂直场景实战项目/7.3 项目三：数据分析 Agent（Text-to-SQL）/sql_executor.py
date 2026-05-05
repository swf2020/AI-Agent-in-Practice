from __future__ import annotations

import sqlite3
import time
import signal
import sqlparse
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_fixed

from core_config import QUERY_TIMEOUT_SECONDS, MAX_RETRIES
from sql_generator import SQLGenerator


FORBIDDEN_KEYWORDS = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "REPLACE", "MERGE", "GRANT", "REVOKE",
    "ATTACH", "DETACH",
})


@dataclass
class ExecutionResult:
    success: bool
    data: Optional[pd.DataFrame]
    error: Optional[str]
    execution_time_ms: float
    row_count: int = 0


class SQLSafetyError(Exception):
    pass


class QueryTimeoutError(Exception):
    pass


def _check_sql_safety(sql: str) -> None:
    parsed = sqlparse.parse(sql)
    for statement in parsed:
        for token in statement.flatten():
            if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL,
                               sqlparse.tokens.Keyword.DML):
                if token.normalized.upper() in FORBIDDEN_KEYWORDS:
                    raise SQLSafetyError(
                        f"拒绝执行：检测到禁止操作 '{token.normalized.upper()}'。"
                        f"本系统仅允许 SELECT 查询。"
                    )

        stmt_type = statement.get_type()
        if stmt_type and stmt_type.upper() not in ("SELECT", "UNKNOWN", None):
            raise SQLSafetyError(
                f"拒绝执行：语句类型为 '{stmt_type}'，仅允许 SELECT。"
            )


class SQLExecutor:
    def __init__(self, db_path: str, timeout_seconds: int = QUERY_TIMEOUT_SECONDS):
        self.db_path = db_path
        self.timeout = timeout_seconds

    def execute(self, sql: str) -> ExecutionResult:
        try:
            _check_sql_safety(sql)
        except SQLSafetyError as e:
            return ExecutionResult(
                success=False, data=None,
                error=str(e), execution_time_ms=0
            )

        start = time.monotonic()

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {self.timeout * 1000}")

        try:
            def _timeout_handler(signum, frame):
                raise QueryTimeoutError(f"查询超时（>{self.timeout}s），疑似全表扫描或缺少索引。")

            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(self.timeout)
            except (AttributeError, OSError):
                pass

            df = pd.read_sql_query(sql + ";", conn)

            try:
                signal.alarm(0)
            except (AttributeError, OSError):
                pass

            elapsed_ms = (time.monotonic() - start) * 1000

            if elapsed_ms > 3000:
                print(f"⚠️  慢查询警告：执行耗时 {elapsed_ms:.0f}ms，考虑添加索引")

            return ExecutionResult(
                success=True,
                data=df,
                error=None,
                execution_time_ms=elapsed_ms,
                row_count=len(df),
            )

        except QueryTimeoutError as e:
            return ExecutionResult(success=False, data=None,
                                   error=str(e), execution_time_ms=self.timeout * 1000)
        except Exception as e:
            return ExecutionResult(
                success=False, data=None,
                error=f"{type(e).__name__}: {e}",
                execution_time_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            conn.close()


class SelfCorrectingExecutor:
    def __init__(
        self,
        executor: SQLExecutor,
        sql_generator: "SQLGenerator",
        schema_prompt: str,
    ):
        self.executor = executor
        self.generator = sql_generator
        self.schema_prompt = schema_prompt

    def execute_with_correction(
        self, original_question: str, initial_sql: str
    ) -> tuple[ExecutionResult, str, int]:
        current_sql = initial_sql
        correction_history: list[dict] = []

        for attempt in range(MAX_RETRIES + 1):
            result = self.executor.execute(current_sql)

            if result.success:
                if attempt > 0:
                    print(f"✅ 第 {attempt} 次修正后成功")
                return result, current_sql, attempt

            if attempt == MAX_RETRIES:
                print(f"❌ 达到最大修正次数（{MAX_RETRIES}），放弃")
                return result, current_sql, attempt

            print(f"🔄 第 {attempt + 1} 次修正，错误：{result.error[:100]}...")

            correction_prompt = f"""上一条 SQL 执行失败，请修正。

**原始问题：** {original_question}

**失败的 SQL：**
```sql
{current_sql}
```

**错误信息：**
{result.error}

**修正要求：**
- 分析错误原因后重新生成正确的 SQL
- 确保语法符合 SQLite 方言
- 不要解释，直接返回修正后的 SQL
"""
            correction_history.extend([
                {"role": "user", "content": f"修正 SQL：{correction_prompt}"},
            ])

            new_result = self.generator.generate(
                question=correction_prompt,
                schema_prompt=self.schema_prompt,
                conversation_history=correction_history,
            )
            current_sql = new_result.sql
            correction_history.append({
                "role": "assistant",
                "content": f"修正后的 SQL：{current_sql}"
            })

        return result, current_sql, MAX_RETRIES
