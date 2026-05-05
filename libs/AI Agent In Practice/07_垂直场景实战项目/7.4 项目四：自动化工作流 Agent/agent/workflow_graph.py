from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
from langgraph.types import interrupt
from langgraph.config import RunnableConfig

if TYPE_CHECKING:
    from langgraph.pregel import CompiledGraph

from models import WorkflowState, ExtractedTask, RiskLevel, EmailMessage
from tools.gmail_tool import gmail_read_email, gmail_mark_processed
from tools.slack_tool import slack_send_notification, send_approval_request, update_approval_message
from tools.task_tool import notion_create_task, jira_create_issue
from config import settings
from core_config import get_chat_model_id, get_api_key

_api_key = get_api_key() or settings.anthropic_api_key
if not _api_key:
    raise RuntimeError(
        "未配置 ANTHROPIC_API_KEY。请在 .env 文件中设置 ANTHROPIC_API_KEY=your_key"
    )

_llm = ChatAnthropic(
    model=get_chat_model_id(),
    api_key=_api_key,
    temperature=0,
)

_extractor = _llm.with_structured_output(ExtractedTask)


def node_read_email(state: WorkflowState) -> dict:
    result = gmail_read_email.invoke({"message_id": state.email_id})
    email_data = json.loads(result)
    return {"email": EmailMessage(**email_data)}


def node_extract_task(state: WorkflowState) -> dict:
    if state.email is None:
        raise ValueError("node_extract_task: state.email 不能为空，请先调用 node_read_email")

    messages = [
        SystemMessage(content="""你是一个工作流自动化助手。
从邮件中提取任务信息，并判断风险等级：
- HIGH 风险：涉及删除、批量变更、对外发送大量通知、金额超过 10000 元
- LOW 风险：其他常规任务创建、提醒、记录
"""),
        HumanMessage(content=f"""
发件人：{state.email.sender}
主题：{state.email.subject}
内容：
{state.email.body}

请提取任务信息。
"""),
    ]

    task = _extractor.invoke(messages)
    return {"extracted_task": task}


def node_request_approval(state: WorkflowState, config: RunnableConfig) -> dict:
    if state.extracted_task is None or state.email is None:
        raise ValueError("node_request_approval: state.extracted_task 和 state.email 不能为空")

    run_id = config["configurable"]["thread_id"]

    ts = send_approval_request(
        task_title=state.extracted_task.title,
        task_description=state.extracted_task.description,
        risk_reason=state.extracted_task.risk_reason,
        email_sender=state.email.sender,
        workflow_run_id=run_id,
    )

    approval_result: dict = interrupt({"message": "等待人工审批", "ts": ts})

    approved: bool = approval_result.get("approved", False)
    operator: str = approval_result.get("operator", "unknown")

    update_approval_message(ts=ts, approved=approved, operator=operator)

    return {
        "approval_message_ts": ts,
        "approved": approved,
    }


def node_write_task(state: WorkflowState) -> dict:
    if state.extracted_task is None:
        raise ValueError("node_write_task: state.extracted_task 不能为空")

    task = state.extracted_task
    result = {}

    if settings.notion_api_key and settings.notion_database_id:
        page_id = notion_create_task.invoke({
            "title": task.title,
            "description": task.description,
            "assignee": task.assignee or "",
            "due_date": task.due_date or "",
            "priority": task.priority,
        })
        result["notion_page_id"] = page_id

    elif settings.jira_base_url and settings.jira_api_token:
        issue_key = jira_create_issue.invoke({
            "title": task.title,
            "description": task.description,
            "assignee": task.assignee or "",
            "due_date": task.due_date or "",
            "priority": task.priority,
        })
        result["jira_issue_key"] = issue_key

    else:
        raise RuntimeError("未配置 Notion 或 Jira，至少需要一个任务管理工具")

    return result


def node_send_notification(state: WorkflowState) -> dict:
    if state.extracted_task is None:
        raise ValueError("node_send_notification: state.extracted_task 不能为空")

    task_ref = state.notion_page_id or state.jira_issue_key or "未知"
    priority_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "urgent": "🔴"}.get(
        state.extracted_task.priority, "⚪"
    )

    msg = (
        f"{priority_emoji} *新任务已创建*\n"
        f"*标题*：{state.extracted_task.title}\n"
        f"*来源*：{state.email.sender if state.email else '未知'}\n"
        f"*任务 ID*：{task_ref}\n"
        f"*截止*：{state.extracted_task.due_date or '未设置'}"
    )
    slack_send_notification.invoke({"message": msg})

    gmail_mark_processed.invoke({"message_id": state.email_id})

    return {}


def node_reject_and_notify(state: WorkflowState) -> dict:
    slack_send_notification.invoke({
        "message": f"⛔ 任务「{state.extracted_task.title if state.extracted_task else '未知'}」已被拒绝审批"
    })
    gmail_mark_processed.invoke({"message_id": state.email_id})
    return {}


def route_by_risk(state: WorkflowState) -> Literal["request_approval", "write_task"]:
    if state.extracted_task and state.extracted_task.risk_level == RiskLevel.HIGH:
        return "request_approval"
    return "write_task"


def route_by_approval(state: WorkflowState) -> Literal["write_task", "reject_and_notify"]:
    if state.approved:
        return "write_task"
    return "reject_and_notify"


def build_workflow_graph(redis_url: str) -> CompiledGraph:
    builder = StateGraph(WorkflowState)

    builder.add_node("read_email", node_read_email)
    builder.add_node("extract_task", node_extract_task)
    builder.add_node("request_approval", node_request_approval)
    builder.add_node("write_task", node_write_task)
    builder.add_node("send_notification", node_send_notification)
    builder.add_node("reject_and_notify", node_reject_and_notify)

    builder.set_entry_point("read_email")
    builder.add_edge("read_email", "extract_task")
    builder.add_conditional_edges("extract_task", route_by_risk)
    builder.add_conditional_edges("request_approval", route_by_approval)
    builder.add_edge("write_task", "send_notification")
    builder.add_edge("send_notification", END)
    builder.add_edge("reject_and_notify", END)

    checkpointer = RedisSaver.from_conn_string(redis_url)

    return builder.compile(checkpointer=checkpointer)
