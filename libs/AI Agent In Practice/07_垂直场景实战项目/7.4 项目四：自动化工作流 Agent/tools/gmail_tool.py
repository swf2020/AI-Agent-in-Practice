from __future__ import annotations
import base64
from datetime import datetime
from typing import Optional
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain_core.tools import tool

from models import EmailMessage
from config import settings


def _get_gmail_service():
    creds = Credentials(
        token=None,
        refresh_token=settings.gmail_refresh_token,
        client_id=settings.gmail_client_id,
        client_secret=settings.gmail_client_secret,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/gmail.modify"],
    )
    creds.refresh(Request())
    return build("gmail", "v1", credentials=creds)


def _decode_body(payload: dict) -> str:
    if payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    return ""


@tool
def gmail_read_email(message_id: str) -> str:
    """读取指定邮件的内容并返回 JSON 格式的 EmailMessage"""
    service = _get_gmail_service()
    msg = service.users().messages().get(
        userId="me",
        id=message_id,
        format="full"
    ).execute()

    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
    body = _decode_body(msg["payload"])
    received_at = datetime.fromtimestamp(int(msg["internalDate"]) / 1000).isoformat()

    email = EmailMessage(
        message_id=message_id,
        subject=headers.get("Subject", "(无主题)"),
        sender=headers.get("From", "未知发件人"),
        body=body[:4000],
        received_at=received_at,
    )
    return email.model_dump_json()


@tool
def gmail_mark_processed(message_id: str) -> str:
    """将指定邮件标记为已处理（添加 WORKFLOW_PROCESSED 标签并移除 UNREAD）"""
    service = _get_gmail_service()
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    label_id = next(
        (lb["id"] for lb in labels if lb["name"] == "WORKFLOW_PROCESSED"),
        None
    )
    if not label_id:
        new_label = service.users().labels().create(
            userId="me",
            body={"name": "WORKFLOW_PROCESSED", "labelListVisibility": "labelHide"}
        ).execute()
        label_id = new_label["id"]

    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={
            "addLabelIds": [label_id],
            "removeLabelIds": ["UNREAD"],
        }
    ).execute()
    return f"邮件 {message_id} 已标记为已处理"
