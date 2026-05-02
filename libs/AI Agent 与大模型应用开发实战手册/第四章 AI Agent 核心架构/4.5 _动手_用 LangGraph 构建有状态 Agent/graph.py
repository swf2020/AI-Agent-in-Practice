from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from agent import agent_node
from router import should_continue
from tools import TOOLS


def build_graph(use_memory: bool = True) -> "CompiledGraph":
    """构建并编译 Agent 图。
    
    Args:
        use_memory: 是否启用内存 Checkpoint（开发调试用）
                   生产环境建议切换为 SqliteSaver 或 PostgresSaver
    
    Returns:
        编译后的图，可直接调用 .invoke() / .stream()
    """
    # 1. 初始化图，绑定 State Schema
    graph_builder = StateGraph(AgentState)
    
    # 2. 添加节点
    # agent_node: 我们自定义的 LLM 决策节点
    graph_builder.add_node("agent", agent_node)
    
    # ToolNode 是 LangGraph 预置的工具执行节点：
    # 自动解析 AIMessage 中的 tool_calls，执行对应工具，
    # 将结果包装成 ToolMessage 追加到 messages
    graph_builder.add_node("tools", ToolNode(TOOLS))
    
    # 3. 定义边（控制流）
    # 入口：START → agent
    graph_builder.add_edge(START, "agent")
    
    # 条件边：agent 执行完毕后，根据 should_continue 的返回值路由
    graph_builder.add_conditional_edges(
        source="agent",           # 从哪个节点出发
        path=should_continue,     # 路由函数
        path_map={                # 返回值 → 目标节点映射
            "tools": "tools",
            "end": END,
        },
    )
    
    # 固定边：工具执行完毕后，无条件回到 agent 节点（ReAct 循环）
    graph_builder.add_edge("tools", "agent")
    
    # 4. 配置 Checkpoint（状态持久化）
    checkpointer = MemorySaver() if use_memory else None
    
    # 5. 编译图（compile 会做静态验证：检查孤立节点、死路等）
    return graph_builder.compile(checkpointer=checkpointer)


# 模块级别的图实例（单例，避免重复编译）
agent_graph = build_graph(use_memory=True)