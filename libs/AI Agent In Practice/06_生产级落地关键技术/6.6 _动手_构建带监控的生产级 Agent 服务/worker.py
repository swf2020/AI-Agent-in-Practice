from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any

from arq.connections import RedisSettings

from agent import run_agent
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _redis_settings() -> RedisSettings:
    """从 REDIS_URL 解析 ARQ 需要的 RedisSettings 对象。"""
    from urllib.parse import urlparse
    parsed = urlparse(settings.redis_url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
    )


async def execute_agent_task(
    ctx: dict[str, Any],
    task_id: str,
    message: str,
    session_id: str,
    user_id: str | None,
) -> None:
    """
    ARQ Worker 执行的任务函数。

    ctx 由 ARQ 注入，包含 Redis 连接等上下文。
    任务结果通过 Redis 存储，供 FastAPI 查询接口读取。
    """
    redis = ctx["redis"]
    result_key = f"task_result:{task_id}"
    created_at = datetime.now(timezone.utc).isoformat()

    # 更新状态为 running
    await redis.set(
        result_key,
        json.dumps({
            "task_id": task_id,
            "status": "running",
            "created_at": created_at,
        }),
        ex=3600,  # 结果保留 1 小时
    )

    try:
        agent_result = await run_agent(
            message=message,
            session_id=session_id,
            user_id=user_id,
        )

        await redis.set(
            result_key,
            json.dumps({
                "task_id": task_id,
                "status": "success",
                "result": agent_result["output"],
                "duration_ms": agent_result["duration_ms"],
                "token_usage": agent_result.get("token_usage"),
                "created_at": created_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }),
            ex=3600,
        )

    except Exception as e:
        logger.exception("task_failed", extra={"task_id": task_id})
        await redis.set(
            result_key,
            json.dumps({
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "created_at": created_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }),
            ex=3600,
        )


class WorkerSettings:
    """ARQ Worker 全局配置。"""

    # 注册所有任务函数
    functions = [execute_agent_task]

    # Redis 连接配置
    redis_settings = _redis_settings()

    # 并发控制：同时最多执行 5 个 Agent 任务
    # 核心考量：LLM API 有速率限制，并发过高会触发 429
    max_jobs = 5

    # 任务超时：超过 2 分钟强制终止
    job_timeout = 120

    # 心跳间隔
    health_check_interval = 30