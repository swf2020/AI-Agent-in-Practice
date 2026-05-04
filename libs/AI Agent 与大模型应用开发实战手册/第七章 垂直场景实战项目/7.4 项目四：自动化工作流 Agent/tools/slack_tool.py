from __future__ import annotations
import json
from typing import Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_core.tools import tool

from config import settings

_client = WebClient(token=settings.slack_bot_token)


@tool
def slack_send_notification(message: str, channel: Optional[str] = None) -> str:
    """向指定 Slack 频道发送通知消息"""
    target = channel or settings.slack_notify_channel
    try:
        response = _client.chat_postMessage(channel=target, text=message)
        return response["ts"]
    except SlackApiError as e:
        raise RuntimeError(f"Slack 通知失败: {e.response['error']}") from e


def send_approval_request(
    task_title: str,
    task_description: str,
    risk_reason: str,
    email_sender: str,
    workflow_run_id: str,
) -> str:
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🔔 需要您的审批"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*任务标题*\n{task_title}"},
                {"type": "mrkdwn", "text": f"*发件人*\n{email_sender}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*任务描述*\n{task_description}"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"⚠️ *高风险原因*：{risk_reason}",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Approve"},
                    "style": "primary",
                    "action_id": f"approve_{workflow_run_id}",
                    "value": workflow_run_id,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ Reject"},
                    "style": "danger",
                    "action_id": f"reject_{workflow_run_id}",
                    "value": workflow_run_id,
                },
            ],
        },
    ]

    response = _client.chat_postMessage(
        channel=settings.slack_approval_channel,
        text=f"审批请求：{task_title}",
        blocks=blocks,
    )
    return response["ts"]


def update_approval_message(ts: str, approved: bool, operator: str) -> None:
    result_text = f"✅ 已批准（操作人：{operator}）" if approved else f"❌ 已拒绝（操作人：{operator}）"
    _client.chat_update(
        channel=settings.slack_approval_channel,
        ts=ts,
        text=result_text,
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": result_text},
            }
        ],
    )