from __future__ import annotations
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

from agent import run_agent
from config import get_settings
from models import (
    ChatRequest,
    TaskRequest,
    TaskResponse,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# ─── Prometheus 指标定义 ────────────────────────────────────────────────────
# 每个指标的 label 设计直接影响 Grafana 的查询灵活度
REQUEST_COUNTER = Counter(
    "agent_requests_total",
    "Total number of agent requests",
    ["endpoint", "status"],  # 按接口和状态码分层
)

LATENCY_HISTOGRAM = Histogram(
    "agent_request_duration_seconds",
    "Agent request duration in seconds",
    ["endpoint"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],  # 适配 LLM 长尾延迟
)

TASK_QUEUE_GAUGE = Counter(
    "agent_tasks_enqueued_total",
    "Total tasks enqueued to ARQ worker",
)

# ─── 应用生命周期管理 ─────────────────────────────────────────────────────────
arq_pool: ArqRedis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI 生命周期管理：启动时建立 Redis 连接池，关闭时清理。"""
    global arq_pool
    logger.info("connecting_to_redis", extra={"url": settings.redis_url})

    from urllib.parse import urlparse
    parsed = urlparse(settings.redis_url)
    arq_pool = await create_pool(
        RedisSettings(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
        )
    )
    logger.info("redis_connected")
    yield
    # 关闭时释放连接池
    await arq_pool.close()
    logger.info("redis_disconnected")


app = FastAPI(
    title="Production Agent Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── 中间件：自动记录延迟 ─────────────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """对每个请求自动记录延迟和状态码，无需在每个路由里手动埋点。"""
    import time
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start

    endpoint = request.url.path
    LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(duration)
    REQUEST_COUNTER.labels(endpoint=endpoint, status=str(response.status_code)).inc()

    return response


# ─── 路由 ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check() -> dict:
    """健康检查接口，供 K8s/ECS 探针和 docker-compose healthcheck 使用。"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Prometheus 抓取接口，返回 text/plain 格式指标数据。"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/chat", summary="同步对话接口（适合 <30s 的短任务）")
async def chat(request: ChatRequest) -> dict:
    """
    同步执行 Agent 并返回结果。

    ⚠️ 适用场景：预期响应时间 < 30s 的请求。
    超过这个阈值建议切换 /task 异步接口，否则客户端容易超时。
    """
    try:
        result = await run_agent(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        return {
            "output": result["output"],
            "duration_ms": result["duration_ms"],
            "session_id": request.session_id,
        }
    except Exception as e:
        logger.exception("chat_endpoint_error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/task", response_model=TaskResponse, summary="异步任务接口（适合长任务）")
async def submit_task(request: TaskRequest) -> TaskResponse:
    """
    将 Agent 任务提交到 ARQ 队列，立即返回 task_id。
    客户端通过 GET /task/{task_id} 轮询结果。
    """
    if arq_pool is None:
        raise HTTPException(status_code=503, detail="任务队列未就绪")

    task_id = str(uuid.uuid4())

    # 在 Redis 中预先写入 pending 状态
    result_key = f"task_result:{task_id}"
    await arq_pool.set(
        result_key,
        json.dumps({
            "task_id": task_id,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }),
        ex=3600,
    )

    # 入队
    await arq_pool.enqueue_job(
        "execute_agent_task",
        task_id=task_id,
        message=request.message,
        session_id=request.session_id,
        user_id=request.user_id,
    )
    TASK_QUEUE_GAUGE.inc()

    return TaskResponse(task_id=task_id)


@app.get("/task/{task_id}", response_model=TaskResult, summary="查询异步任务结果")
async def get_task_result(task_id: str) -> TaskResult:
    """轮询任务状态。前端建议以 2s 间隔轮询，任务完成后停止。"""
    if arq_pool is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    result_key = f"task_result:{task_id}"
    raw = await arq_pool.get(result_key)

    if raw is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在或已过期")

    data = json.loads(raw)
    return TaskResult(
        task_id=data["task_id"],
        status=TaskStatus(data["status"]),
        result=data.get("result"),
        error=data.get("error"),
        duration_ms=data.get("duration_ms"),
        token_usage=data.get("token_usage"),
        created_at=datetime.fromisoformat(data["created_at"]),
        completed_at=(
            datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None
        ),
    )