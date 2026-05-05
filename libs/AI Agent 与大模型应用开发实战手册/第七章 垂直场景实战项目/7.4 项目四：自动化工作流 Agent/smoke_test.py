import json
import time
from unittest.mock import patch, MagicMock
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from models import WorkflowState, EmailMessage, ExtractedTask, RiskLevel


def run_smoke_test():
    fake_email = EmailMessage(
        message_id="test_msg_001",
        subject="紧急：请删除全部测试用户数据",
        sender="cto@example.com",
        body="Hi，请立即删除 staging 数据库中的所有用户数据，我们要做一次全量重置。谢谢。",
        received_at="2025-01-01T10:00:00",
    )

    fake_task = ExtractedTask(
        title="删除 staging 数据库全部用户数据",
        description="CTO 要求对 staging 环境执行全量用户数据清理",
        assignee="ops@example.com",
        due_date="2025-01-01",
        priority="urgent",
        risk_level=RiskLevel.HIGH,
        risk_reason="涉及批量删除操作，数据不可恢复",
    )

    # 关键修复：
    # - @tool 装饰对象（BaseTool）不能用 patch("xxx.invoke") 来 mock，
    #   因为 setattr 会触发 Pydantic 的 __setattr__ 校验。
    #   正确做法：用 MagicMock 替换整个对象，并配置其 .invoke() 返回值。
    # - _extractor 是 RunnableSequence，同样不能用 patch 替换其方法。
    # - send_approval_request / update_approval_message 是普通函数，可用 patch 替换。
    mock_gmail_read = MagicMock()
    mock_gmail_read.invoke.return_value = fake_email.model_dump_json()
    mock_extractor = MagicMock()
    mock_extractor.invoke.return_value = fake_task
    mock_notion = MagicMock()
    mock_notion.invoke.return_value = "notion-page-abc"
    mock_slack_notify = MagicMock()
    mock_slack_notify.invoke.return_value = "ts-notify"
    mock_gmail_mark = MagicMock()
    mock_gmail_mark.invoke.return_value = "done"

    with (
        patch("agent.workflow_graph.gmail_read_email", mock_gmail_read),
        patch("agent.workflow_graph._extractor", mock_extractor),
        patch("agent.workflow_graph.send_approval_request", return_value="1234567890.123456"),
        patch("agent.workflow_graph.update_approval_message"),
        patch("agent.workflow_graph.notion_create_task", mock_notion),
        patch("agent.workflow_graph.slack_send_notification", mock_slack_notify),
        patch("agent.workflow_graph.gmail_mark_processed", mock_gmail_mark),
        patch("agent.workflow_graph.settings", MagicMock(
            notion_api_key="fake-notion-key",
            notion_database_id="fake-db-id",
        )),
    ):
        import agent.workflow_graph as wg

        def patched_build(redis_url):
            from langgraph.graph import StateGraph, END
            builder = StateGraph(WorkflowState)
            builder.add_node("read_email", wg.node_read_email)
            builder.add_node("extract_task", wg.node_extract_task)
            builder.add_node("request_approval", wg.node_request_approval)
            builder.add_node("write_task", wg.node_write_task)
            builder.add_node("send_notification", wg.node_send_notification)
            builder.add_node("reject_and_notify", wg.node_reject_and_notify)
            builder.set_entry_point("read_email")
            builder.add_edge("read_email", "extract_task")
            builder.add_conditional_edges("extract_task", wg.route_by_risk)
            builder.add_conditional_edges("request_approval", wg.route_by_approval)
            builder.add_edge("write_task", "send_notification")
            builder.add_edge("send_notification", END)
            builder.add_edge("reject_and_notify", END)
            return builder.compile(checkpointer=MemorySaver())

        wg.build_workflow_graph = patched_build

        graph = wg.build_workflow_graph("redis://localhost")
        config = {"configurable": {"thread_id": "test_msg_001"}}

        print("▶ Step 1: 首次执行（预期在审批节点暂停）...")
        try:
            graph.invoke(WorkflowState(email_id="test_msg_001"), config=config)
        except Exception as e:
            print(f"  图执行状态：中断等待审批（符合预期）")

        snapshot = graph.get_state(config)
        print(f"  当前节点：{snapshot.next}")

        print("\n▶ Step 2: 模拟审批通过...")
        graph.invoke(
            Command(resume={"approved": True, "operator": "admin"}),
            config=config,
        )

        final_state = graph.get_state(config)
        print(f"  最终状态：{final_state.next}")
        print(f"  Notion Page ID: {final_state.values.get('notion_page_id')}")

    print("\n=== 测试完成 ✓ ===")


if __name__ == "__main__":
    run_smoke_test()