from __future__ import annotations
import time
import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_litellm import ChatLiteLLM

from config import get_settings
from core_config import get_litellm_id, get_api_key, get_base_url

logger = logging.getLogger(__name__)
settings = get_settings()


# ─── 工具函数（模块级，避免每次调用 run_agent 时重建）─── [Fix #4]

def _calculator(expression: str) -> str:
    """安全计算数学表达式，仅允许数字和基本运算符。

    教学说明：通过在空 __builtins__ 字典中执行 eval，禁用了所有内置函数。
    这是章节演示计算器工具的权宜之计——生产环境应替换为
    ast.literal_eval 或专用数学表达式解析器（如 numexpr）。  # [Fix #8]
    """
    allowed_chars = set("0123456789+-*/()., ")
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式包含不允许的字符"
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


def _get_current_time(_: str) -> str:
    """返回当前时间，演示无参工具的用法。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC+8")


def _mock_search(query: str) -> str:
    """模拟搜索工具，生产环境替换为 Tavily/Brave Search。"""
    return f"搜索结果（模拟）：关于「{query}」，根据最新资料显示..."


def build_tools() -> list[Tool]:
    """
    构建工具列表。
    生产环境这里会接入 Tavily 搜索、数据库查询等真实工具。
    本节用简单工具确保代码可立即运行。
    """
    return [
        Tool(name="calculator", func=_calculator, description="计算数学表达式，输入: 数学表达式字符串"),
        Tool(name="get_time", func=_get_current_time, description="获取当前时间，输入: 任意字符串"),
        Tool(name="search", func=_mock_search, description="搜索网络信息，输入: 搜索关键词"),
    ]


# ReAct 系统提示词
REACT_SYSTEM_PROMPT = """You are a helpful assistant. You have access to a set of tools that you can use to answer questions.
Use your tools to solve problems step by step. When you have a final answer, provide it clearly to the user.

Available tools:
{tools}"""


async def run_agent(
    message: str,
    session_id: str = "default",
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    执行 Agent 并返回结果与元数据。

    返回值包含：
    - output: Agent 最终回答
    - duration_ms: 总耗时
    - token_usage: 各阶段 Token 消耗（由 LangFuse Handler 汇总）
    """
    start_time = time.monotonic()

    tools = build_tools()
    tools_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    system_prompt = REACT_SYSTEM_PROMPT.format(tools=tools_desc)

    llm = ChatLiteLLM(
        model=get_litellm_id(),
        temperature=0,
        max_tokens=settings.max_tokens_per_request,
        api_key=get_api_key(),
        api_base=get_base_url(),
        streaming=False,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    try:
        result = await agent.ainvoke(
            {"messages": [("user", message)]},
            config={
                "recursion_limit": settings.agent_max_iterations + 1
                # [Fix #2] 限制 Agent 最大递归次数，防止陷入无限 ReAct 循环
                # +1 因为首次调用不计入 recursion
            },
        )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(
            "agent_run_success",
            extra={"session_id": session_id, "duration_ms": duration_ms},
        )

        # 提取最终回复
        messages = result.get("messages", [])
        output = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                output = msg.content
                break

        return {
            "output": output,
            "duration_ms": duration_ms,
            "token_usage": {},
        }

    except Exception as e:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        logger.error(
            "agent_run_failed",
            extra={"session_id": session_id, "error": str(e), "duration_ms": duration_ms},
        )
        raise
