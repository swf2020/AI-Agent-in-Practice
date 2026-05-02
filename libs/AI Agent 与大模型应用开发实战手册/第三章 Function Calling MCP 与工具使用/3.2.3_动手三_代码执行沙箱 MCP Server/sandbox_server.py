"""
代码执行沙箱 MCP Server
依赖：mcp[cli]>=1.3.0, fastmcp>=0.4.0
运行：python sandbox_server.py  （或通过 Claude Desktop 配置）
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

# ── 服务实例 ──────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "code-sandbox",
    instructions=(
        "Python 代码执行沙箱。支持运行代码、安装包、查看历史、重置环境。"
        "每次调用 execute_python 都是独立子进程，变量不跨次保留。"
        "如需多步共享状态，请在单次 execute_python 中写完整脚本。"
    ),
)

# ── 会话状态（进程级，重启后清空）────────────────────────────────────────────
@dataclass
class ExecutionRecord:
    """单次执行的完整记录"""
    timestamp: str
    code: str
    success: bool
    stdout: str
    stderr: str
    duration_ms: int
    error: str = ""

@dataclass
class SessionState:
    """跨 Tool 调用的共享会话状态"""
    history: list[ExecutionRecord] = field(default_factory=list)
    installed_packages: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

# 全局单例：MCP Server 是单进程服务，这里的全局变量在整个运行期间有效
_session = SessionState()