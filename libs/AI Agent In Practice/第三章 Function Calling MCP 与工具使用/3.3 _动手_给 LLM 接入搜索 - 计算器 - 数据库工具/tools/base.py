"""工具基类与调度器定义"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """所有工具必须继承此基类。

    设计原则：工具负责"做"，不负责"决策调用时机"——那是 LLM 的事。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，必须与 JSON Schema 中的 name 字段一致。"""
        ...

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """OpenAI Function Calling 格式的工具描述。

        description 字段极为关键：LLM 完全依赖它来判断是否调用该工具。
        描述要说清楚"什么时候用"，而不只是"能做什么"。
        """
        ...

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """执行工具逻辑，返回字符串结果（LLM 只接受文本反馈）。"""
        ...


class ToolDispatcher:
    """工具调度器：管理工具注册、LLM 调用循环、工具执行。"""

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools: dict[str, BaseTool] = {t.name: t for t in tools}

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """返回所有工具的 schema 列表，直接传给 OpenAI API 的 tools 参数。"""
        return [t.schema for t in self._tools.values()]

    def dispatch(self, tool_name: str, tool_args: str) -> str:
        """根据 LLM 的工具调用请求执行对应工具。

        Args:
            tool_name: LLM 选择的工具名称
            tool_args: JSON 格式的参数字符串（来自 LLM 的 function_call.arguments）

        Returns:
            工具执行结果的字符串表示
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            # 返回错误信息而非抛异常——让 LLM 知道工具不存在并自行纠正
            return f"Error: tool '{tool_name}' not found. Available: {list(self._tools.keys())}"

        try:
            args = json.loads(tool_args)
            result = tool.run(**args)
            logger.info("Tool '%s' executed successfully", tool_name)
            return result
        except json.JSONDecodeError as e:
            return f"Error: invalid arguments JSON: {e}"
        except Exception as e:
            logger.exception("Tool '%s' execution failed", tool_name)
            # 生产中不要把完整 traceback 返回给 LLM（信息泄露风险）
            return f"Error executing tool: {type(e).__name__}: {str(e)[:200]}"
