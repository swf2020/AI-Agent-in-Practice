"""工具调用 Agent 主循环"""

from __future__ import annotations

import json
import os
from typing import Any

import litellm
from dotenv import load_dotenv

from core_config import get_api_key, get_base_url, get_litellm_id
from tools.base import ToolDispatcher
from tools.code_tool import E2BCodeExecutorTool
from tools.db_tool import TextToSQLTool
from tools.search_tool import TavilySearchTool

load_dotenv()


class ToolCallingAgent:
    """支持多工具调用的 LLM Agent。

    实现标准的 ReAct 循环：
    LLM 思考 → 选择工具 → 执行工具 → 观察结果 → 循环直到得出答案
    """

    MAX_TURNS = 10  # 防止死循环：最多 10 轮工具调用

    def __init__(self, dispatcher: ToolDispatcher, model: str | None = None) -> None:
        self._dispatcher = dispatcher
        self._model = model or get_litellm_id()
        self._api_key = get_api_key()
        self._base_url = get_base_url()
        self._system_prompt = (
            "你是一个有用的 AI 助手，可以使用搜索、代码执行和数据库查询工具来回答问题。\n"
            "工具使用原则：\n"
            "1. 需要最新信息时，使用搜索工具\n"
            "2. 需要精确计算时，使用代码执行工具（不要自己心算）\n"
            "3. 需要查询结构化数据时，使用数据库工具\n"
            "4. 能直接回答的问题不要无谓调用工具"
        )

    def run(self, user_input: str, verbose: bool = False) -> str:
        """执行工具调用循环，返回最终答案。

        Args:
            user_input: 用户输入
            verbose: 是否打印工具调用过程（调试用）

        Returns:
            LLM 的最终文本回答
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_input},
        ]

        for turn in range(self.MAX_TURNS):
            response = litellm.completion(
                model=self._model,
                api_key=self._api_key,
                api_base=self._base_url,
                messages=messages,
                tools=self._dispatcher.schemas,
                tool_choice="auto",  # 让 LLM 自主决定是否调用工具
            )

            message = response.choices[0].message
            # 将 assistant 消息追加到历史（包含 tool_calls 信息）
            messages.append(message)  # type: ignore[arg-type]

            # 没有工具调用 → LLM 给出了最终答案
            if not message.tool_calls:
                return message.content or ""

            # 执行所有工具调用（OpenAI 支持并行工具调用）
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                if verbose:
                    print(f"\n调用工具: {tool_name}")
                    print(f"   参数: {tool_args[:200]}")

                tool_result = self._dispatcher.dispatch(tool_name, tool_args)

                if verbose:
                    print(f"   结果: {tool_result[:300]}...")

                # 工具结果必须以 tool 角色回传，且 tool_call_id 要对应
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        return "达到最大工具调用轮次，无法完成任务"


def build_agent(db_url: str = "sqlite:///demo.db") -> ToolCallingAgent:
    """工厂函数：组装带三类工具的 Agent 实例。"""
    tools = [
        TavilySearchTool(max_results=3),
        E2BCodeExecutorTool(timeout=30),
        TextToSQLTool(db_url=db_url),
    ]
    dispatcher = ToolDispatcher(tools)
    return ToolCallingAgent(dispatcher=dispatcher)
