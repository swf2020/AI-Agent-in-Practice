from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from graph import build_graph


def demo_multi_turn_memory():
    """演示：Checkpoint 实现多轮对话状态持久化。
    
    thread_id 是会话标识符，同一 thread_id 的调用会共享状态历史。
    不同 thread_id 之间完全隔离，天然支持多用户场景。
    """
    graph = build_graph(use_memory=True)
    
    # thread_id 标识一个对话会话，可以是用户ID、会话UUID等
    config = {"configurable": {"thread_id": "user_001_session_1"}}
    
    # 第一轮对话
    print("=== 第一轮 ===")
    result = graph.invoke(
        input={
            "messages": [HumanMessage(content="LangGraph 是什么？它和 LangChain 有什么区别？")],
            "tool_calls_count": 0,
            "requires_approval": False,
        },
        config=config,
    )
    print(result["messages"][-1].content[:200])
    
    # 查看 Checkpoint 快照（调试用）
    snapshot = graph.get_state(config)
    print(f"\n当前状态快照：messages 数量={len(snapshot.values['messages'])}")
    
    # 第二轮对话：不需要重传历史，Checkpoint 自动恢复上下文
    print("\n=== 第二轮（自动携带上轮历史）===")
    result2 = graph.invoke(
        input={"messages": [HumanMessage(content="它支持哪些 Checkpoint 后端？")]},
        # LangGraph 会从 Checkpoint 中恢复完整状态，再 merge 新的 messages
        config=config,
    )
    print(result2["messages"][-1].content[:300])


def demo_human_approval():
    """演示：使用 interrupt_before 实现人工审批卡点。
    
    核心机制：compile(interrupt_before=["tools"]) 让 Agent 在即将执行工具前暂停，
    将控制权交还给调用方。调用方检查 pending tool_calls，决定是否批准继续执行。
    """
    # 创建带中断点的图：在执行 tools 节点前暂停
    from state import AgentState
    from agent import agent_node
    from router import should_continue
    from tools import TOOLS
    from langgraph.prebuilt import ToolNode
    from langgraph.graph import StateGraph, START, END

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(TOOLS))
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph_builder.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    # interrupt_before=["tools"] 是关键：Agent 决策后、工具执行前暂停
    approval_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )
    
    config = {"configurable": {"thread_id": "approval_demo_001"}}
    
    # Step 1：发起请求，Agent 决策调用工具后暂停
    print("=== 发起请求（Agent 即将调用工具）===")
    _result = approval_graph.invoke(
        input={
            "messages": [HumanMessage(content="搜索 LangGraph 最新版本号")],
            "tool_calls_count": 0,
            "requires_approval": False,
        },
        config=config,
    )
    
    # 此时图已暂停，检查 Agent 想做什么
    snapshot = approval_graph.get_state(config)
    last_msg = snapshot.values["messages"][-1]
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print("\n⏸️  Agent 暂停，待审批的工具调用：")
        for tc in last_msg.tool_calls:
            print(f"   工具: {tc['name']}, 参数: {tc['args']}")
        
        # Step 2：模拟人工审批（实际场景可接入 Slack Bot / 审批系统）
        approved = input("\n✅ 批准执行? (y/n): ").strip().lower() == "y"
        
        if approved:
            print("\n▶️  已批准，继续执行...")
            # 传入 None 作为 input，从 Checkpoint 恢复继续执行
            final_result = approval_graph.invoke(None, config=config)
            print(f"\n最终回答：{final_result['messages'][-1].content}")
        else:
            print("\n❌ 已拒绝，中止执行")


if __name__ == "__main__":
    demo_multi_turn_memory()
    demo_human_approval()