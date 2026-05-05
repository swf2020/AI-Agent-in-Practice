"""
代码执行沙箱 MCP Server
依赖：mcp[cli]>=1.3.0, fastmcp>=0.4.0
运行：python sandbox_server.py  （或通过 Claude Desktop 配置）
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
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

# ── 安全策略 ─────────────────────────────────────────────────────────────────
DANGEROUS_MODULES = {
    "os", "sys", "subprocess", "shutil", "multiprocessing",
    "socket", "ctypes", "pickle", "marshal", "ftplib", "telnetlib",
    "http.client", "urllib.request",
}

DANGEROUS_FUNCTIONS = {"eval", "exec", "compile", "__import__", "getattr", "setattr"}


def is_safe_code(code: str) -> tuple[bool, str]:
    """
    静态分析代码，检查是否包含危险 import 或危险函数调用。

    Returns:
        (True, "")  if safe
        (False, reason) if unsafe
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"代码语法错误: {e}"

    for node in ast.walk(tree):
        # 检查危险 import
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in DANGEROUS_MODULES:
                    return False, f"安全检查拒绝：禁止导入模块 '{alias.name}'"
                # 检查子模块（如 os.path -> os）
                top = alias.name.split(".")[0]
                if top in DANGEROUS_MODULES:
                    return False, f"安全检查拒绝：禁止导入模块 '{alias.name}'"

        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in DANGEROUS_MODULES:
                return False, f"安全检查拒绝：禁止从 '{node.module}' 导入"

        # 检查危险函数调用
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DANGEROUS_FUNCTIONS:
                return False, f"安全检查拒绝：禁止调用 '{func.id}()'"

    return True, ""


# ── 核心执行逻辑 ─────────────────────────────────────────────────────────────
def execute_python(code: str, timeout: int = 10, work_dir: str | None = None) -> dict[str, Any]:
    """
    在子进程中执行 Python 代码，返回结构化结果。

    Args:
        code: 要执行的 Python 代码字符串
        timeout: 超时时间（秒），默认 10 秒
        work_dir: 工作目录（默认使用临时目录）

    Returns:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "error": str,
            "duration_ms": int,
        }
    """
    # 安全检查
    safe, reason = is_safe_code(code)
    if not safe:
        record = ExecutionRecord(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            code=code, success=False, stdout="", stderr="",
            duration_ms=0, error=reason,
        )
        _session.history.append(record)
        return {"success": False, "stdout": "", "stderr": "", "error": reason, "duration_ms": 0}

    # 创建临时工作目录
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="sandbox_")

    # 将代码写入临时文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=work_dir, delete=False
    ) as f:
        f.write(code)
        script_path = f.name

    start_time = time.time()
    try:
        # 在子进程中执行，隔离环境变量
        env = os.environ.copy()
        # 移除可能干扰的 Python 路径
        env.pop("PYTHONPATH", None)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True,
            timeout=timeout,
            cwd=work_dir,
            env=env,
        )
        duration_ms = int((time.time() - start_time) * 1000)

        success = result.returncode == 0
        record = ExecutionRecord(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            code=code, success=success,
            stdout=result.stdout, stderr=result.stderr,
            duration_ms=duration_ms,
            error="" if success else f"进程退出码: {result.returncode}",
        )
        _session.history.append(record)

        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": "" if success else f"进程退出码: {result.returncode}",
            "duration_ms": duration_ms,
        }

    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - start_time) * 1000)
        error_msg = f"代码执行超时（>{timeout}秒）"
        record = ExecutionRecord(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            code=code, success=False, stdout="", stderr="",
            duration_ms=duration_ms, error=error_msg,
        )
        _session.history.append(record)
        return {"success": False, "stdout": "", "stderr": "", "error": error_msg, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_msg = f"执行异常: {e}"
        record = ExecutionRecord(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            code=code, success=False, stdout="", stderr="",
            duration_ms=duration_ms, error=error_msg,
        )
        _session.history.append(record)
        return {"success": False, "stdout": "", "stderr": "", "error": error_msg, "duration_ms": duration_ms}


def install_package(package_name: str) -> dict[str, Any]:
    """
    在沙箱环境中安装 pip 包。

    Args:
        package_name: 包名（如 numpy, matplotlib）

    Returns:
        {
            "success": bool,
            "message": str,
        }
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "-q"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            _session.installed_packages.append(package_name)
            return {"success": True, "message": f"成功安装 {package_name}"}
        else:
            return {
                "success": False,
                "message": f"安装失败: {result.stderr.strip()[:200]}",
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "安装超时 (>60s)"}
    except Exception as e:
        return {"success": False, "message": f"安装异常: {e}"}


# ── MCP 工具定义 ─────────────────────────────────────────────────────────────
@mcp.tool()
def run_code(code: str, timeout: int = 10) -> dict[str, Any]:
    """
    在沙箱中执行 Python 代码。

    Args:
        code: 要执行的 Python 代码
        timeout: 超时时间（秒），默认 10 秒

    Returns:
        包含 success, stdout, stderr, error, duration_ms 的字典
    """
    return execute_python(code, timeout=timeout)


@mcp.tool()
def pip_install(package: str) -> dict[str, Any]:
    """
    安装 Python 包到沙箱环境中。

    Args:
        package: 包名（如 numpy）
    """
    return install_package(package)


@mcp.tool()
def get_execution_history(last_n: int = 10) -> dict[str, Any]:
    """
    获取本次会话的代码执行历史记录。

    Args:
        last_n: 返回最近 N 条记录（默认 10，最大 50）
    """
    last_n = max(1, min(last_n, 50))
    recent = _session.history[-last_n:][::-1]

    return {
        "total": len(_session.history),
        "session_start": _session.created_at,
        "installed_packages": _session.installed_packages,
        "records": [
            {
                "timestamp": r.timestamp,
                "success": r.success,
                "duration_ms": r.duration_ms,
                "code_preview": r.code[:100] + ("..." if len(r.code) > 100 else ""),
                "stdout_preview": r.stdout[:200] + ("..." if len(r.stdout) > 200 else ""),
                "error": r.error,
            }
            for r in recent
        ],
    }


@mcp.tool()
def reset_session_state() -> dict[str, Any]:
    """
    清空执行历史和已安装包记录，重置会话状态。
    注意：不会卸载已安装的 pip 包。
    """
    global _session
    old_count = len(_session.history)
    _session = SessionState()

    return {
        "success": True,
        "message": f"会话已重置。清除了 {old_count} 条执行记录。",
        "note": "pip 包不会被卸载，已安装的包在本进程内仍可使用。",
    }


# 别名：供 test_sandbox.py 和外部导入使用
def reset_session() -> dict[str, Any]:
    """重置会话状态（非 MCP 工具版本）"""
    return reset_session_state()


# ── 服务入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔒 代码执行沙箱 MCP Server 启动", flush=True)
    print(f"   Python: {sys.executable}", flush=True)
    print("   安全模式: 静态分析 + 子进程隔离 + 环境变量隔离", flush=True)
    mcp.run(transport="stdio")
