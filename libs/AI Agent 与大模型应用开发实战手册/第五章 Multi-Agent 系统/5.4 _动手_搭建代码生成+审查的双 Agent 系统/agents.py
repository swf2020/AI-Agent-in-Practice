"""
双 Agent 代码生成+审查系统。

Coder Agent：根据需求生成代码 + 测试用例。
Reviewer Agent：审查代码质量，给出修改建议。
工具层：代码执行（pytest）、静态分析（pylint/flake8）、安全扫描（bandit）。
"""

import re
import json
import litellm
from dataclasses import dataclass, field

from core_config import get_litellm_id, get_api_key, get_base_url
from tools import execute_code_with_tests, run_static_analysis, run_security_scan, ToolResult


# ── 系统 Prompt ────────────────────────────────────────────────────

CODER_SYSTEM_PROMPT = """\
你是一个资深 Python 工程师。你的任务是根据用户需求，生成高质量的 Python 代码和对应的测试用例。

**输出格式（严格遵守）**：

你的输出必须包含两个代码块，使用以下格式：

```implementation
<这里是完整的 Python 实现代码>
```

```tests
<这里是 pytest 格式的测试代码>
```

**代码规范**：
- 使用类型注解（type hints）
- 遵循 PEP 8 风格
- 纯函数优先，避免全局状态和副作用
- 对非法输入进行校验并抛出合适的异常
- 代码必须能通过所有提供的测试用例

**测试规范**：
- 使用 pytest 格式（`test_` 前缀的函数或 `assert` 语句）
- 覆盖正常路径和边界情况
- 不要 import 被测代码（框架会自动注入）"""

REVIEWER_SYSTEM_PROMPT = """\
你是一个严格的代码审查专家。请从以下维度审查代码：

1. **正确性**：逻辑是否正确？边界情况是否处理？
2. **安全性**：是否存在常见安全漏洞（如 eval/exec 滥用、输入注入）？
3. **可读性**：命名是否清晰？结构是否合理？
4. **性能**：是否有明显的性能问题？
5. **测试覆盖**：测试用例是否充分？

**输出格式（严格遵守）**：

如果代码通过审查（综合评分 >= 0.85），输出：
```review
STATUS: PASS
SCORE: <0.0~1.0 的分数>
COMMENT: <简短的通过评语>
```

如果代码不通过，输出：
```review
STATUS: FAIL
SCORE: <0.0~1.0 的分数>
COMMENT: <具体修改建议，分点列出>
```"""


# ── 代码块解析 ────────────────────────────────────────────────────

def _extract_code_blocks(message: str) -> dict[str, str]:
    """
    从 LLM 输出中提取 markdown 代码块。

    Returns:
        {"implementation": "...", "tests": "..."} 或
        {"review": "..."} 或空 dict（解析失败时）
    """
    blocks: dict[str, str] = {}
    # 匹配 ```lang\n...\n``` 模式
    pattern = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(message):
        lang = match.group(1).strip().lower()
        code = match.group(2).strip()
        blocks[lang] = code
    return blocks


# ── LLM 调用封装 ──────────────────────────────────────────────────

def _call_llm(messages: list[dict], temperature: float = 0.7) -> str:
    """
    调用 LiteLLM 完成一次对话，返回 assistant 消息内容。
    """
    response = litellm.completion(
        model=get_litellm_id(),
        api_key=get_api_key(),
        api_base=get_base_url(),
        messages=messages,
        temperature=temperature,
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ── 双 Agent 主循环 ────────────────────────────────────────────────

def run_dual_agent_loop(
    requirement: str,
    pass_threshold: float = 0.85,
    max_rounds: int = 6,
    verbose: bool = False,
) -> dict:
    """
    执行 Coder + Reviewer 双 Agent 循环。

    Args:
        requirement: 用户需求描述
        pass_threshold: 综合评分通过阈值（0.0 ~ 1.0）
        max_rounds: 最大迭代轮次
        verbose: 是否打印详细过程

    Returns:
        {
            "success": bool,
            "rounds": int,
            "final_code": str,
            "final_tests": str,
            "final_score": float,
            "history": list[dict],  # 每轮详情
        }
    """
    history: list[dict] = []
    final_code = ""
    final_tests = ""
    final_score = 0.0

    # 对话历史（Coder 侧）
    coder_messages = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {"role": "user", "content": f"需求：\n{requirement}"},
    ]

    for round_num in range(1, max_rounds + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"第 {round_num} 轮")
            print(f"{'='*60}")

        # ── Step 1: Coder 生成代码 ──
        coder_reply = _call_llm(coder_messages, temperature=0.1)
        blocks = _extract_code_blocks(coder_reply)
        implementation = blocks.get("implementation", "")
        tests = blocks.get("tests", "")

        if not implementation:
            if verbose:
                print("[Coder] 未能解析实现代码，重试...")
            coder_messages.append({"role": "assistant", "content": coder_reply})
            coder_messages.append({
                "role": "user",
                "content": "请严格按照输出格式提供 implementation 和 tests 代码块。",
            })
            continue

        final_code = implementation
        final_tests = tests

        if verbose:
            print(f"[Coder] 已生成代码（{len(implementation)} 字符）")

        # ── Step 2: 工具层评估 ──
        tool_results: list[tuple[str, ToolResult]] = []

        # 2a. 测试执行
        if tests:
            exec_result = execute_code_with_tests(implementation, tests)
            tool_results.append(("测试执行", exec_result))
            if verbose:
                print(f"[工具] 测试: {'通过' if exec_result.success else '失败'} (score={exec_result.score:.2f})")

        # 2b. 静态分析
        static_result = run_static_analysis(implementation)
        tool_results.append(("静态分析", static_result))
        if verbose:
            print(f"[工具] 静态分析: score={static_result.score:.2f}")

        # 2c. 安全扫描
        security_result = run_security_scan(implementation)
        tool_results.append(("安全扫描", security_result))
        if verbose:
            print(f"[工具] 安全扫描: score={security_result.score:.2f}")

        # ── Step 3: Reviewer 审查 ──
        tool_summary = "\n\n".join(
            f"--- {name} ---\n{r.output}" for name, r in tool_results
        )
        reviewer_prompt = (
            f"请审查以下代码。\n\n"
            f"需求：\n{requirement}\n\n"
            f"代码：\n```python\n{implementation}\n```\n\n"
            f"测试：\n```python\n{tests}\n```\n\n"
            f"工具评估结果：\n{tool_summary}"
        )

        reviewer_messages = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": reviewer_prompt},
        ]
        reviewer_reply = _call_llm(reviewer_messages, temperature=0.3)
        review_blocks = _extract_code_blocks(reviewer_reply)
        review_text = review_blocks.get("review", reviewer_reply)

        # 解析 Reviewer 判定
        review_status = "FAIL"
        review_score = 0.0
        if "STATUS: PASS" in review_text:
            review_status = "PASS"
            score_match = re.search(r"SCORE:\s*([\d.]+)", review_text)
            review_score = float(score_match.group(1)) if score_match else 0.85
        else:
            score_match = re.search(r"SCORE:\s*([\d.]+)", review_text)
            review_score = float(score_match.group(1)) if score_match else 0.5

        # 综合评分 = 工具分 60% + Reviewer 分 40%
        tool_avg = sum(r.score for _, r in tool_results) / len(tool_results)
        final_score = tool_avg * 0.6 + review_score * 0.4

        round_record = {
            "round": round_num,
            "code": implementation,
            "tests": tests,
            "tool_results": {name: r.output for name, r in tool_results},
            "reviewer_reply": review_text,
            "tool_avg": tool_avg,
            "reviewer_score": review_score,
            "final_score": final_score,
            "status": review_status,
        }
        history.append(round_record)

        if verbose:
            print(f"[Reviewer] {review_status} (reviewer_score={review_score:.2f})")
            print(f"[综合] 最终评分: {final_score:.2f}")

        # ── Step 4: 终止判断 ──
        if final_score >= pass_threshold:
            if verbose:
                print(f"\n达到阈值 {pass_threshold}，循环结束！")
            return {
                "success": True,
                "rounds": round_num,
                "final_code": final_code,
                "final_tests": final_tests,
                "final_score": final_score,
                "history": history,
            }

        # ── Step 5: 反馈给 Coder，进入下一轮 ──
        feedback = (
            f"第 {round_num} 轮评分 {final_score:.2f}，未达到阈值 {pass_threshold}。\n\n"
            f"审查意见：\n{review_text}\n\n"
            f"工具结果摘要：\n{tool_summary}\n\n"
            f"请根据以上反馈修改代码，重新提供 implementation 和 tests 代码块。"
        )
        coder_messages.append({"role": "assistant", "content": coder_reply})
        coder_messages.append({"role": "user", "content": feedback})

    # 超出最大轮数
    return {
        "success": False,
        "rounds": max_rounds,
        "final_code": final_code,
        "final_tests": final_tests,
        "final_score": final_score,
        "history": history,
    }
