from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from state import AgentState
from tools import TOOLS
from core_config import get_litellm_id, get_chat_model_id, get_api_key, get_base_url, ACTIVE_MODEL_KEY, MODEL_REGISTRY


def create_llm(provider: str = "default"):
    """工厂函数：按提供商创建 LLM 实例并绑定工具。

    provider 可选值：
    - "default": 使用 core_config 中 ACTIVE_MODEL_KEY 对应的模型（通过 LiteLLM 路由）
    - "anthropic": 直连 Anthropic Claude
    - "openai": 直连 OpenAI GPT

    bind_tools() 是关键：它把工具的 JSON Schema 注入到每次请求，
    让模型知道有哪些工具可用，返回的 AIMessage 可能携带 tool_calls。
    """
    if provider == "anthropic":
        llm = ChatAnthropic(model=get_chat_model_id("Claude-Sonnet"), temperature=0)
    elif provider == "openai":
        llm = ChatOpenAI(model=get_chat_model_id("GPT-4o-Mini"), temperature=0)
    else:
        # 默认路径：通过 LiteLLM 调用，模型由 core_config 统一管理
        litellm_model = get_litellm_id()
        api_key = get_api_key()
        base_url = get_base_url()

        # 使用 langchain-litellm 的 ChatLiteLLM 适配层
        from langchain_litellm import ChatLiteLLM

        llm = ChatLiteLLM(
            model=litellm_model,
            api_key=api_key,
            api_base=base_url,
            temperature=0,
        )

    return llm.bind_tools(TOOLS)


SYSTEM_PROMPT = """你是一个专业的研究助手，能够搜索最新信息并进行计算。

工作原则：
1. 对于需要实时信息的问题，优先使用搜索工具
2. 对于数学计算，使用计算器工具确保准确性
3. 综合多个搜索结果后给出有依据的回答
4. 如果不确定，如实说明，不要编造信息

当前工具限制：每轮对话最多调用工具 5 次，超出后必须给出最终回答。"""


def agent_node(state: AgentState) -> dict:
    """Agent 核心节点：接收当前状态，调用 LLM，返回状态更新。

    Node 的契约：
    - 输入：完整的 AgentState
    - 输出：需要更新的字段（partial update），LangGraph 负责合并

    这个函数是纯函数：相同输入 → 相同（类型的）输出，便于测试和调试。
    """
    llm_with_tools = create_llm()

    # 注入系统提示（每次调用都加，确保模型行为一致）
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    response = llm_with_tools.invoke(messages)

    # 只返回需要更新的字段，LangGraph 的 Reducer 负责合并
    return {
        "messages": [response],
        "tool_calls_count": state["tool_calls_count"] + (
            1 if hasattr(response, "tool_calls") and response.tool_calls else 0
        ),
    }
