"""E2B 沙箱代码执行工具"""

from __future__ import annotations

import os
from typing import Any

from e2b_code_interpreter import Sandbox

from tools.base import BaseTool


class E2BCodeExecutorTool(BaseTool):
    """在 E2B 安全沙箱中执行 Python 代码。
    
    适用场景：
    - 数学计算（尤其是需要精确结果时，避免 LLM 算错）
    - 数据处理与统计分析
    - 代码验证与调试
    
    ⚠️ 不适用于：需要访问本地文件系统或数据库的场景。
    """

    def __init__(self, api_key: str | None = None, timeout: int = 30) -> None:
        self._api_key = api_key or os.environ["E2B_API_KEY"]
        self._timeout = timeout  # 单次执行超时秒数，防止死循环

    @property
    def name(self) -> str:
        return "python_executor"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "在安全沙箱中执行 Python 代码。用于精确数学计算、"
                    "数据统计、算法验证等需要真实运行结果的任务。"
                    "代码可以使用 numpy、pandas、matplotlib 等常用库。"
                    "不能访问外部网络或本地文件。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "要执行的 Python 代码。"
                                "用 print() 输出结果，否则无法获取返回值。"
                            ),
                        },
                        "packages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "需要额外安装的 pip 包列表（已预装 numpy/pandas/matplotlib）",
                            "default": [],
                        },
                    },
                    "required": ["code"],
                },
            },
        }

    def run(self, code: str, packages: list[str] | None = None) -> str:
        """在 E2B 沙箱中执行代码并返回输出。
        
        每次调用都会创建一个新的沙箱实例，执行完毕自动销毁。
        这是 E2B 的设计哲学：无状态 = 无风险。
        """
        # 使用 context manager 确保沙箱资源释放
        with Sandbox(api_key=self._api_key, timeout=self._timeout) as sandbox:
            # 按需安装额外依赖（冷启动约 3-5 秒，有成本意识）
            if packages:
                install_result = sandbox.commands.run(
                    f"pip install -q {' '.join(packages)}"
                )
                if install_result.exit_code != 0:
                    return f"Package installation failed: {install_result.stderr}"

            execution = sandbox.run_code(code)

            # 收集所有输出类型
            output_parts: list[str] = []

            # 标准输出（print 的内容）
            if execution.logs.stdout:
                output_parts.append("【输出】\n" + "\n".join(execution.logs.stdout))

            # 错误信息
            if execution.logs.stderr:
                output_parts.append("【错误】\n" + "\n".join(execution.logs.stderr))

            # 如果有图表，告知 LLM（图表本身无法直接传回文本 LLM）
            if execution.results:
                for result in execution.results:
                    if hasattr(result, "png"):
                        output_parts.append("【图表】已生成图表（PNG 格式，共 1 张）")

            if not output_parts:
                return "代码执行成功，无输出（确认 print() 是否被调用）"

            return "\n\n".join(output_parts)