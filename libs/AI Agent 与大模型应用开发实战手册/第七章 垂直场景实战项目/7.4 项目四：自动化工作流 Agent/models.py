from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class RiskLevel(str, Enum):
    LOW = "low"
    HIGH = "high"


class ExtractedTask(BaseModel):
    title: str = Field(description="任务标题，50字以内")
    description: str = Field(description="任务详细描述")
    assignee: Optional[str] = Field(None, description="被分配人，邮件地址或姓名")
    due_date: Optional[str] = Field(None, description="截止日期，格式 YYYY-MM-DD")
    priority: str = Field(default="medium", description="优先级：low/medium/high/urgent")
    risk_level: RiskLevel = Field(description="操作风险等级")
    risk_reason: str = Field(description="风险等级判断依据")


class EmailMessage(BaseModel):
    message_id: str
    subject: str
    sender: str
    body: str
    received_at: datetime


class WorkflowState(BaseModel):
    email_id: str
    email: Optional[EmailMessage] = None
    extracted_task: Optional[ExtractedTask] = None
    approval_message_ts: Optional[str] = None
    approved: Optional[bool] = None
    rejection_reason: Optional[str] = None
    notion_page_id: Optional[str] = None
    jira_issue_key: Optional[str] = None
    error: Optional[str] = None