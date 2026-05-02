from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Agent 运行时的完整状态。
    
    字段说明：
    - messages: 对话历史，使用 add_messages reducer 自动合并而非覆盖
    - tool_calls_count: 本轮工具调用计数，用于防止无限循环
    - requires_approval: 标记是否需要人工审批后才能继续执行
    """
    # Annotated + add_messages 是 LangGraph 的核心约定：
    # 多个节点向 messages 写入时，自动 append 而非覆盖
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls_count: int
    requires_approval: bool