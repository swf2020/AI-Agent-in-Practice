# ⚠️ 仅用于本地学习，生产环境绝对禁止
import subprocess
import sys

def local_execute_unsafe(code: str) -> str:
    """本地执行，无任何沙箱隔离——仅用于测试！"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=10
    )
    return result.stdout or result.stderr