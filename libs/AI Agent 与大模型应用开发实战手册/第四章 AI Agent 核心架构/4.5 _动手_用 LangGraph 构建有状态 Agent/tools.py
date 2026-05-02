import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

load_dotenv()


def get_search_tool() -> TavilySearchResults:
    """返回配置好的 Tavily 搜索工具。
    
    max_results=3 是经验值：结果太多会撑爆上下文窗口，
    太少可能漏掉关键信息。可根据模型上下文大小调整。
    """
    return TavilySearchResults(max_results=3)


@tool
def calculate(expression: str) -> str:
    """安全地执行数学计算表达式。
    
    Args:
        expression: 合法的 Python 数学表达式，如 "2 ** 10 + 100"
    
    Returns:
        计算结果的字符串表示
    
    Example:
        >>> calculate("(1024 * 1024) / 1000")
        '1048.576'
    """
    # ⚠️ 生产环境应使用沙箱（E2B / Modal），这里用白名单简化演示
    allowed_names = {"__builtins__": {}}
    import math
    allowed_names.update({k: getattr(math, k) for k in dir(math) if not k.startswith("_")})
    try:
        result = eval(expression, allowed_names)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


# 工具列表，后续绑定到 LLM 和 ToolNode
TOOLS = [get_search_tool(), calculate]