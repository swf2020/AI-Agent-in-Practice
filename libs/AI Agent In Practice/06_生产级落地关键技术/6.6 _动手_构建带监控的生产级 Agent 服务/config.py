from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings  # pydantic v2 拆包，需 pip install pydantic-settings

from arq.connections import RedisSettings
from urllib.parse import urlparse


class Settings(BaseSettings):
    """应用配置，自动从 .env 文件读取。"""

    openai_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    redis_url: str = "redis://localhost:6379"

    # Agent 运行参数
    agent_max_iterations: int = 10
    agent_timeout_seconds: int = 120

    # 成本控制：单次请求最大 Token 消耗
    max_tokens_per_request: int = 4000

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """全局单例配置，避免重复读取 .env 文件。"""
    return Settings()


def get_redis_settings(redis_url: str | None = None) -> RedisSettings:
    """从 Redis URL 解析 ARQ 需要的 RedisSettings 对象。  # [Fix #5]

    将 URL 解析逻辑集中一处，供 main.py 和 worker.py 共用，
    避免重复实现。
    """
    url = redis_url or get_settings().redis_url
    parsed = urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
    )