from typing import Literal
from langchain_core.messages import AIMessage
from state import AgentState

# 工具调用上限：防止 Agent 陷入无限工具调用循环
MAX_TOOL_CALLS = 5


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """条件边路由函数：返回值对应 conditional_edges 中定义的路由 key。
    
    路由逻辑（优先级从高到低）：
    1. 工具调用次数超上限 → 强制结束，防止死循环
    2. 最新消息有 tool_calls → 路由到 tools 节点
    3. 否则 → 结束
    
    Returns:
        "tools": 路由到工具执行节点
        "end": 路由到 END，对话结束
    """
    last_message = state["messages"][-1]
    
    # 安全阀：超过调用上限，强制结束
    if state["tool_calls_count"] >= MAX_TOOL_CALLS:
        return "end"
    
    # 检查 LLM 是否决定调用工具
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    return "end"


def check_approval(state: AgentState) -> Literal["approved", "pending"]:
    """人工审批卡点路由函数（用于高风险操作场景）。
    
    当 requires_approval=True 时，Agent 会在此节点暂停（interrupt），
    等待外部系统修改 state 后再继续执行。
    """
    if state.get("requires_approval", False):
        return "pending"
    return "approved"