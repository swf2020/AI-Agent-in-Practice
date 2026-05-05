# run_agent.py —— 端到端冒烟测试，直接复制运行
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph import build_graph

load_dotenv()

def run_smoke_test():
    """端到端验证：搜索工具 + 多轮对话 + Checkpoint 恢复。"""
    graph = build_graph(use_memory=True)
    config = {"configurable": {"thread_id": "smoke_test_001"}}
    
    test_cases = [
        "今天是几号？LangGraph 0.2 版本有哪些重要更新？",
        "它的最新版本号是多少？",  # 测试多轮上下文
        "2 的 16 次方等于多少？",  # 测试计算器工具
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"[第{i}轮] {question}")
        print("-" * 60)
        
        # 流式输出：实时看到工具调用过程
        for event in graph.stream(
            input={
                "messages": [HumanMessage(content=question)],
                "tool_calls_count": 0,
                "requires_approval": False,
            },
            config=config,
            stream_mode="values",  # 每次 State 更新都输出
        ):
            last_msg = event["messages"][-1]
            msg_type = type(last_msg).__name__
            
            if msg_type == "AIMessage":
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"  🔧 调用工具: {tc['name']}({tc['args']})")
                elif last_msg.content:
                    print(f"  🤖 回答: {last_msg.content[:300]}")
            elif msg_type == "ToolMessage":
                print(f"  📥 工具结果: {str(last_msg.content)[:100]}...")


if __name__ == "__main__":
    run_smoke_test()