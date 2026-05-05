import re
import json
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from core_config import get_litellm_id, get_api_key, get_base_url

load_dotenv()


def get_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """创建 OpenAI 客户端，使用 core_config 中的配置"""
    key = api_key or get_api_key()
    url = base_url or get_base_url()
    kwargs = {}
    if key:
        kwargs["api_key"] = key
    if url:
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


class Dialect(str, Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class SQLGenerationResult(BaseModel):
    sql: str = Field(description="生成的 SQL 查询语句，不含 markdown 代码块标记")
    explanation: str = Field(description="用一句话解释这条 SQL 的查询逻辑")
    confidence: float = Field(ge=0.0, le=1.0, description="生成置信度，0~1")
    ambiguities: list[str] = Field(
        default_factory=list,
        description="问题中存在的模糊点，如'最近'未指定具体时间范围"
    )


def _parse_json_response(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON，处理可能的格式问题"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取 JSON 块
    import re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # 如果解析失败，返回默认值
    return {"sql": "", "explanation": "解析失败", "confidence": 0.0, "ambiguities": []}


SYSTEM_PROMPT_TEMPLATE = """你是一个专业的数据分析 SQL 专家。你的任务是将用户的自然语言问题转换为正确的 {dialect} SQL 查询。

## 数据库 Schema
{schema}

## 生成规则
1. 只生成 SELECT 查询，绝对不生成 INSERT/UPDATE/DELETE/DROP/ALTER 等写操作
2. 使用 {dialect} 方言语法（如 SQLite 日期函数用 strftime，PostgreSQL 用 DATE_TRUNC）
3. 金额字段保留 2 位小数：ROUND(amount, 2)
4. 时间范围若问题未指定，默认取最近 90 天
5. 结果集行数超过 100 行时，自动加 LIMIT 100
6. 对涉及多表查询，优先使用显式 JOIN 而非子查询（可读性更好）
7. 字段别名使用中文（方便最终用户读图）

## Few-shot 示例
用户：各城市的订单总金额是多少？
SQL：
SELECT u.city AS 城市, ROUND(SUM(o.total_amount), 2) AS 订单总金额
FROM orders o
JOIN users u ON o.user_id = u.user_id
WHERE o.status = 'completed'
GROUP BY u.city
ORDER BY 订单总金额 DESC

用户：上个月每天的新增用户数
SQL：
SELECT DATE(register_date) AS 日期, COUNT(*) AS 新增用户数
FROM users
WHERE register_date >= DATE('now', '-1 month', 'start of month')
  AND register_date < DATE('now', 'start of month')
GROUP BY DATE(register_date)
ORDER BY 日期
"""


class SQLGenerator:
    def __init__(
        self,
        model: str | None = None,
        dialect: Dialect = Dialect.SQLITE,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or get_litellm_id()
        self.dialect = dialect
        self.client = get_openai_client(api_key=api_key, base_url=base_url)

    def generate(
        self,
        question: str,
        schema_prompt: str,
        conversation_history: list[dict] | None = None,
    ) -> SQLGenerationResult:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            dialect=self.dialect.value,
            schema=schema_prompt,
        )

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history[-6:])

        messages.append({"role": "user", "content": question})

        # 优先尝试使用结构化输出
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=SQLGenerationResult,
                temperature=0.1,
            )
            result = response.choices[0].message.parsed
            result.sql = self._clean_sql(result.sql)
            return result
        except Exception:
            # 结构化输出失败时，回退到普通 completion + JSON 手动解析
            print("⚠️  结构化输出解析失败，回退到普通模式...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )
            content = response.choices[0].message.content or ""
            data = _parse_json_response(content)
            result = SQLGenerationResult(**data)
            result.sql = self._clean_sql(result.sql)
            return result

    @staticmethod
    def _clean_sql(sql: str) -> str:
        sql = re.sub(r"```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = sql.strip().rstrip(";")
        return sql
