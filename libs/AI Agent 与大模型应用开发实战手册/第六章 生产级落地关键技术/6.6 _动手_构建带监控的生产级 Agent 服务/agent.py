from __future__ import annotations
import time
import logging
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def build_tools() -> list[Tool]:
    """
    构建工具列表。
    生产环境这里会接入 Tavily 搜索、数据库查询等真实工具。
    本节用简单工具确保代码可立即运行。
    """

    def calculator(expression: str) -> str:
        """安全计算数学表达式，仅允许数字和基本运算符。"""
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        try:
            result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
            return str(result)
        except Exception as e:
            return f"计算错误：{e}"

    def get_current_time(_: str) -> str:
        """返回当前时间，演示无参工具的用法。"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC+8")

    def mock_search(query: str) -> str:
        """模拟搜索工具，生产环境替换为 Tavily/Brave Search。"""
        return f"搜索结果（模拟）：关于「{query}」，根据最新资料显示..."

    return [
        Tool(name="calculator", func=calculator, description="计算数学表达式，输入: 数学表达式字符串"),
        Tool(name="get_time", func=get_current_time, description="获取当前时间，输入: 任意字符串"),
        Tool(name="search", func=mock_search, description="搜索网络信息，输入: 搜索关键词"),
    ]


def create_langfuse_handler(
    session_id: str,
    user_id: str | None,
    trace_name: str = "agent_run",
) -> LangfuseCallbackHandler:
    """
    为每次 Agent 执行创建独立的 LangFuse Handler。

    关键设计：每个请求用独立的 Handler，确保 trace 不会跨请求污染。
    session_id 和 user_id 传入后，LangFuse 界面可以按用户/会话过滤。
    """
    return LangfuseCallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        session_id=session_id,
        user_id=user_id,
        trace_name=trace_name,
        tags=["production", "react-agent"],
    )


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

    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 生产环境根据任务复杂度动态路由
        temperature=0,
        max_tokens=settings.max_tokens_per_request,
        api_key=settings.openai_api_key,
        streaming=False,
    )

    tools = build_tools()
    # 使用 LangChain Hub 的标准 ReAct 提示词，生产环境建议固定版本 hash
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    langfuse_handler = create_langfuse_handler(
        session_id=session_id,
        user_id=user_id,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=settings.agent_max_iterations,
        handle_parsing_errors=True,  # LLM 输出格式错误时不 crash，而是重试
        verbose=False,  # 生产环境关闭，避免日志膨胀
    )

    try:
        result = await agent_executor.ainvoke(
            {"input": message},
            config={"callbacks": [langfuse_handler]},
        )
        # 确保 LangFuse 数据已上报（异步批量上报，需主动 flush）
        langfuse_handler.flush()

        duration_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(
            "agent_run_success",
            extra={"session_id": session_id, "duration_ms": duration_ms},
        )

        return {
            "output": result["output"],
            "duration_ms": duration_ms,
            "token_usage": {},  # LangFuse Handler 已记录详细用量，此处简化
        }

    except Exception as e:
        langfuse_handler.flush()
        duration_ms = int((time.monotonic() - start_time) * 1000)
        logger.error(
            "agent_run_failed",
            extra={"session_id": session_id, "error": str(e), "duration_ms": duration_ms},
        )
        raise