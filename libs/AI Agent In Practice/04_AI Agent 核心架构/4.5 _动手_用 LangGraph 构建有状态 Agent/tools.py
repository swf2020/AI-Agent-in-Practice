import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool

# 尝试从多个位置加载 .env 文件
# 1. 当前工作目录
# 2. 项目根目录（向上查找）
env_loaded = load_dotenv()
if not env_loaded:
    project_root = Path(__file__).resolve().parent
    while project_root.parent != project_root:
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            break
        project_root = project_root.parent


def get_search_tool():
    """返回配置好的搜索工具。

    优先使用 TavilySearchResults（需 TAVILY_API_KEY）；
    若未配置则返回一个 Mock 搜索工具，避免 ImportError 阻塞导入。

    max_results=3 是经验值：结果太多会撑爆上下文窗口，
    太少可能漏掉关键信息。可根据模型上下文大小调整。
    """
    if os.getenv("TAVILY_API_KEY") and os.getenv("TAVILY_API_KEY") != "test_dummy_key":
        from langchain_tavily import TavilySearch
        return TavilySearch(max_results=3)
    else:
        return mock_search


@tool
def mock_search(query: str) -> str:
    """搜索网络获取最新信息。

    Args:
        query: 搜索查询词

    Returns:
        模拟的搜索结果（当 TAVILY_API_KEY 未配置时使用）
    """
    return f"[模拟搜索] 关于'{query}'：当前未配置 TAVILY_API_KEY，此为占位结果。请设置真实的 Tavily API Key 启用真实搜索。"


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


# [Fix #5] 改用惰性求值：每次调用时动态检查环境变量，
# 而非模块导入时一次性锁定。支持运行时设置 API Key 后即时生效。
def get_tools() -> list:
    """返回当前环境下的工具列表。
    
    每次调用时重新执行 get_search_tool()，确保环境变量变更后
    工具实例能立即反映最新配置（如 TAVILY_API_KEY）。
    """
    return [get_search_tool(), calculate]


# 保留 TOOLS 供向后兼容（模块导入时锁定环境变量状态）
# 推荐新代码使用 get_tools()
TOOLS = get_tools()