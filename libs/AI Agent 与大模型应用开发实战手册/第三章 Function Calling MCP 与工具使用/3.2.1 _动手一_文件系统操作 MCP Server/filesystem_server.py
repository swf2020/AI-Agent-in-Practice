import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# ── 安全配置 ──────────────────────────────────────────────────────────────────
# ALLOWED_ROOT 定义 Server 可操作的根目录。
# 优先从环境变量读取，让部署时灵活配置；默认限制在当前工作目录。
ALLOWED_ROOT = Path(
    os.environ.get("MCP_ALLOWED_ROOT", Path.cwd())
).expanduser().resolve()

# 文件大小上限：避免 Claude 把一个 2GB 的日志文件整个读进上下文
MAX_FILE_SIZE_BYTES = int(os.environ.get("MCP_MAX_FILE_SIZE", 1024 * 1024))  # 默认 1MB

def _safe_path(raw: str) -> Path:
    """
    将用户传入的路径解析为绝对路径，并校验是否在白名单根目录内。
    
    使用 Path.resolve() 而非简单的字符串前缀匹配，是为了防止路径穿越攻击：
    ../../../etc/passwd 经过 resolve() 后会暴露真实绝对路径，从而被拦截。
    """
    resolved = Path(raw).expanduser().resolve()
    # is_relative_to 是 Python 3.9+ 的方法，确保路径在白名单内
    if not resolved.is_relative_to(ALLOWED_ROOT):
        raise PermissionError(
            f"路径 '{raw}' 超出允许的根目录 '{ALLOWED_ROOT}'。"
            f"请设置环境变量 MCP_ALLOWED_ROOT 扩大访问范围。"
        )
    return resolved

# ── Server 初始化 ──────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="filesystem-server",
    # instructions 是 Server 级别的系统提示，会注入给调用方的 LLM。
    # 清晰说明能力边界，有助于 Claude 做出更准确的工具选择。
    instructions=(
        f"你可以操作本机文件系统，根目录限定为：{ALLOWED_ROOT}。"
        f"单文件最大读取 {MAX_FILE_SIZE_BYTES // 1024}KB。"
        "写操作会直接修改磁盘文件，请在执行前向用户确认。"
    ),
)