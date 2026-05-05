from __future__ import annotations
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class ChatRequest(BaseModel):
    """同步对话请求体。"""
    message: str = Field(..., min_length=1, max_length=2000, description="用户输入")
    session_id: str = Field(default="default", description="会话 ID，用于多轮记忆")
    user_id: str | None = Field(default=None, description="用户标识，用于成本归因")


class TaskRequest(BaseModel):
    """异步任务请求体。"""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = "default"
    user_id: str | None = None


class TaskResponse(BaseModel):
    """提交任务后立即返回的响应。"""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskResult(BaseModel):
    """任务结果查询响应。"""
    task_id: str
    status: TaskStatus
    result: str | None = None
    error: str | None = None
    duration_ms: int | None = None
    token_usage: dict[str, int] | None = None
    created_at: datetime
    completed_at: datetime | None = None