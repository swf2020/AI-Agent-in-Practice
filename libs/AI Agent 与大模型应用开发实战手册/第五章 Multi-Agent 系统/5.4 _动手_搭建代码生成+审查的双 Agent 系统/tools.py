"""
确定性工具层：代码执行、静态分析、安全扫描。
所有函数均在独立进程中运行，与主进程隔离，防止恶意代码污染状态。
"""

import subprocess
import tempfile
import textwrap
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ToolResult:
    """工具执行结果的统一返回格式"""
    success: bool
    output: str
    score: float  # 0.0 ~ 1.0，供终止条件判断用


def _write_temp_file(code: str, suffix: str = ".py") -> Path:
    """将代码写入临时文件，返回路径。调用方负责清理。"""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    tmp.write(code)
    tmp.flush()
    return Path(tmp.name)


def execute_code_with_tests(implementation: str, test_code: str) -> ToolResult:
    """
    在独立进程中执行实现代码 + pytest 测试用例。

    Args:
        implementation: 被测代码字符串
        test_code: pytest 格式的测试代码字符串

    Returns:
        ToolResult，success=True 表示所有测试通过
    """
    impl_path = _write_temp_file(implementation, ".py")
    # 测试文件 import 实现模块，因此需要放在同一目录
    test_path = impl_path.parent / f"test_{impl_path.name}"

    # 在测试文件顶部注入 sys.path，保证 import 能找到实现文件
    test_with_import = textwrap.dedent(f"""\
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(r"{impl_path}").parent))
        from {impl_path.stem} import *  # noqa: F401,F403
        {test_code}
    """)
    test_path.write_text(test_with_import, encoding="utf-8")

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", str(test_path), "-v", "--tb=short", "--no-header"],
            capture_output=True,
            text=True,
            timeout=30,  # 单次测试最多 30 秒，防止死循环
        )
        output = result.stdout + result.stderr

        # 解析通过率：从 pytest 最后一行 "X passed, Y failed" 提取
        passed, total = _parse_pytest_summary(output)
        score = passed / total if total > 0 else 0.0
        success = result.returncode == 0

        return ToolResult(
            success=success,
            output=f"[测试执行结果]\n{output.strip()}\n通过率: {passed}/{total} ({score:.0%})",
            score=score,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="[错误] 测试超时（>30s），可能存在死循环", score=0.0)
    finally:
        impl_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)


def _parse_pytest_summary(output: str) -> tuple[int, int]:
    """从 pytest 输出解析通过数和总数，返回 (passed, total)。"""
    import re
    # 匹配 "3 passed" / "2 passed, 1 failed" 等模式
    passed = sum(int(m) for m in re.findall(r"(\d+) passed", output))
    failed = sum(int(m) for m in re.findall(r"(\d+) failed", output))
    error = sum(int(m) for m in re.findall(r"(\d+) error", output))
    total = passed + failed + error
    return passed, max(total, 1)


def run_static_analysis(code: str) -> ToolResult:
    """
    运行 pylint + flake8 静态分析。

    pylint 给出 0-10 评分，我们要求 >= 8.0 才算通过。
    flake8 检测 PEP8 违规，零容忍 E/W 级别错误。
    """
    code_path = _write_temp_file(code)
    issues: list[str] = []
    pylint_score = 0.0

    try:
        # pylint：忽略 C0114/C0116（docstring 缺失），专注逻辑错误
        pylint_result = subprocess.run(
            ["python", "-m", "pylint", str(code_path),
             "--disable=C0114,C0116,C0115",
             "--output-format=text"],
            capture_output=True, text=True, timeout=15
        )
        pylint_output = pylint_result.stdout
        issues.append(f"[pylint]\n{pylint_output.strip()}")

        # 提取 pylint 评分："Your code has been rated at 8.50/10"
        import re
        match = re.search(r"rated at ([\d.]+)/10", pylint_output)
        pylint_score = float(match.group(1)) if match else 5.0

        # flake8：只报告 E（错误）和 W（警告），忽略 E501（行长度）
        flake8_result = subprocess.run(
            ["python", "-m", "flake8", str(code_path), "--max-line-length=100",
             "--select=E,W", "--ignore=E501"],
            capture_output=True, text=True, timeout=15
        )
        flake8_output = flake8_result.stdout.strip()
        if flake8_output:
            issues.append(f"[flake8]\n{flake8_output}")
        else:
            issues.append("[flake8] ✅ 无 PEP8 违规")

        # 综合评分：pylint 权重 0.7，flake8 通过权重 0.3
        flake8_clean = flake8_result.returncode == 0
        combined_score = (pylint_score / 10.0) * 0.7 + (1.0 if flake8_clean else 0.0) * 0.3
        success = pylint_score >= 8.0 and flake8_clean

        return ToolResult(
            success=success,
            output="\n\n".join(issues) + f"\n\n综合评分: {combined_score:.2f}/1.00",
            score=combined_score,
        )
    finally:
        code_path.unlink(missing_ok=True)


def run_security_scan(code: str) -> ToolResult:
    """
    使用 bandit 扫描常见安全漏洞。

    bandit 严重级别：LOW / MEDIUM / HIGH。
    我们对 HIGH 级别零容忍，MEDIUM 级别超过 2 个即不通过。
    """
    code_path = _write_temp_file(code)

    try:
        result = subprocess.run(
            ["python", "-m", "bandit", str(code_path), "-f", "text", "-ll"],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout + result.stderr

        import re
        high_count = len(re.findall(r"Severity: High", output, re.IGNORECASE))
        medium_count = len(re.findall(r"Severity: Medium", output, re.IGNORECASE))

        success = high_count == 0 and medium_count <= 2
        # 安全分：HIGH 每个扣 0.3，MEDIUM 每个扣 0.1
        score = max(0.0, 1.0 - high_count * 0.3 - medium_count * 0.1)

        summary = f"HIGH: {high_count} 个  MEDIUM: {medium_count} 个"
        return ToolResult(
            success=success,
            output=f"[bandit 安全扫描]\n{output.strip()}\n{summary}",
            score=score,
        )
    finally:
        code_path.unlink(missing_ok=True)