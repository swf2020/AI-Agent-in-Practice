# 继续 agents.py

def _extract_code_blocks(message: str) -> dict[str, str]:
    """
    从 Coder 输出中提取 implementation 和 tests 代码块。
    
    Returns:
        {"implementation": "...", "tests": "..."} 或空 dict（解析失败时）
    """
    pattern = re.compile(r"