"""
流式 CoT 解析核心模块。
支持多种模型和推理模式：
  1. CoT Prompt 模式（通用）：通过 <think>...</think> 标签区分思考与回答
  2. DeepSeek 推理模型：使用 deepseek-reasoner 原生推理能力
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI

from core_config import get_api_key, get_base_url, get_litellm_id, ACTIVE_MODEL_KEY, MODEL_REGISTRY

load_dotenv()


class ChunkType(str, Enum):
    THINKING = "thinking"  # 推理过程（灰显）
    ANSWER = "answer"      # 最终回答（高亮）
    META = "meta"          # 元信息（如速率）


@dataclass
class StreamChunk:
    """单个流式 chunk 的标准化结构。"""
    content: str
    chunk_type: ChunkType
    timestamp: float = field(default_factory=time.perf_counter)


def get_openai_client(model_key: str | None = None):
    """根据 core_config 配置返回正确配置的 OpenAI 客户端。"""
    key = model_key or ACTIVE_MODEL_KEY
    cfg = MODEL_REGISTRY.get(key)
    if cfg is None:
        # 回退到无配置情况
        return OpenAI()
    api_key = get_api_key(key)
    base_url = get_base_url(key)
    if api_key:
        return OpenAI(api_key=api_key, base_url=base_url)
    else:
        # [Fix #6] API Key 缺失时给出明确的错误提示，帮助学习者快速定位问题
        env_var = MODEL_REGISTRY[key].get("api_key_env", "API_KEY")
        print(f"⚠️  未检测到 {key} 的 API Key")
        print(f"   请设置环境变量：export {env_var}='your-key'")
        print(f"   或创建 .env 文件并确保 load_dotenv() 已调用")
        return OpenAI()


def get_default_model(model_key: str | None = None):
    """根据 core_config 返回默认模型名称（SDK 用，去掉 provider 前缀）。"""
    key = model_key or ACTIVE_MODEL_KEY
    litellm_id = get_litellm_id(key)
    # OpenAI SDK 不需要 "provider/" 前缀，去掉它
    if "/" in litellm_id:
        return litellm_id.split("/", 1)[1]
    return litellm_id


# ─────────────────────────────────────────────
#  通用流解析器（被两种模式共享）
# ─────────────────────────────────────────────

def _parse_tagged_stream(
    stream,
    think_tag: str = "think",
    answer_tag: str = "answer",
) -> Generator[StreamChunk, None, None]:
    """
    通用标签流解析器。逐 chunk 消费原始流，识别 XML 标签边界，
    将每个 chunk 标注类型（THINKING / ANSWER）后 yield。

    标签检测采用 accumulated 缓冲区累积策略，
    防止标签被切分到两个 chunk 中导致漏检。
    """
    open_think = f"<{think_tag}>"
    close_think = f"</{think_tag}>"
    open_answer = f"<{answer_tag}>"
    close_answer = f"</{answer_tag}>"

    # 检测优先级：闭合标签优先于开放标签，避免将闭合标签误认为内容
    tag_set = [close_think, close_answer, open_answer, open_think]

    accumulated = ""
    in_thinking = False
    processed_up_to = 0

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue

        accumulated += delta

        if not in_thinking and open_think in accumulated:
            think_start = accumulated.index(open_think) + len(open_think)
            processed_up_to = think_start
            in_thinking = True

        if in_thinking and close_think in accumulated:
            think_end = accumulated.index(close_think)
            remaining_think = accumulated[processed_up_to:think_end]
            if remaining_think:
                yield StreamChunk(content=remaining_think, chunk_type=ChunkType.THINKING)
            processed_up_to = think_end + len(close_think)
            in_thinking = False

        if not in_thinking and open_answer in accumulated[processed_up_to:]:
            answer_tag_pos = accumulated.index(open_answer, processed_up_to) + len(open_answer)
            processed_up_to = answer_tag_pos

        new_content = accumulated[processed_up_to:]

        for tag in tag_set:
            if tag in new_content:
                new_content = new_content[:new_content.index(tag)]
                break

        if new_content:
            chunk_type = ChunkType.THINKING if in_thinking else ChunkType.ANSWER
            yield StreamChunk(content=new_content, chunk_type=chunk_type)
            processed_up_to += len(new_content)


# ─────────────────────────────────────────────
#  模式一：CoT Prompt（OpenAI 兼容接口）
# ─────────────────────────────────────────────

COT_SYSTEM_PROMPT = """\
你是一个严谨的推理助手。

在回答任何问题时，请严格遵循以下格式：
1. 先在 <think> 标签内写出完整的推理过程（可以有多个步骤）
2. 再在 <answer> 标签内给出最终答案（简洁清晰）

格式示例：
<think>
首先分析题目...
然后考虑...
因此得出...
</think>
<answer>
最终答案是...
</answer>

注意：两个标签都必须出现，不要省略推理过程。"""


def stream_cot_prompt(
    prompt: str,
    model: str | None = None,  # [Fix #11] 类型注解修正
    temperature: float = 0.6,
) -> Generator[StreamChunk, None, None]:
    """
    使用 CoT System Prompt 驱动流式输出，实时解析 <think> 标签。

    Args:
        prompt: 用户问题
        model:  任何 OpenAI 兼容模型名称，默认根据 ACTIVE_MODEL_KEY 自动选择
        temperature: 采样温度，推理任务建议 0.3-0.7

    Yields:
        StreamChunk，chunk_type 区分 thinking 和 answer
    """
    client = get_openai_client()
    model = model or get_default_model()

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=temperature,
    )

    # [Fix #5] 使用通用解析器，与 stream_extended_thinking 共享逻辑
    yield from _parse_tagged_stream(stream, think_tag="think", answer_tag="answer")


# ─────────────────────────────────────────────
#  模式二：DeepSeek 推理模型
# ─────────────────────────────────────────────

DEEPSEEK_THINKING_PROMPT = """\
你现在进入了深度推理模式。请按照以下严格的格式输出：

**思考阶段**：在 <thinking> 和 </thinking> 标签之间详细记录你的推理过程，包括：
- 问题分析和理解
- 关键假设和前提条件
- 逐步推理步骤
- 可能的验证方法

**回答阶段**：在 <answer> 和 </answer> 标签之间给出最终答案，确保：
- 答案简洁明了
- 直接回应用户问题
- 使用自然友好的语言

例如：
<thinking>
用户问的是一个数学问题...
首先我需要理解题目...
然后应用相关公式...
验证计算结果...
</thinking>
<answer>
最终答案是：XXX
</answer>

严格按照此格式输出，不要省略任何标签！"""


def stream_extended_thinking(
    prompt: str,
    budget_tokens: int = 8000,  # [Fix #2] budget_tokens 现在实际控制 max_tokens
    use_reasoner: bool = False,
) -> Generator[StreamChunk, None, None]:
    """
    DeepSeek 思考链推理。

    Args:
        prompt:       用户问题
        budget_tokens: 分配给思考过程的最大 token 数（越大推理越充分）
                      [Fix #2] 此参数现在会实际传递给 API 的 max_tokens
        use_reasoner:  True 时使用 deepseek-reasoner 推理模型，
                      False 时使用普通 chat 模型 + 系统提示词开启思考模式

    Yields:
        StreamChunk，chunk_type=thinking 对应思考过程，answer 对应最终回复
    """
    client = get_openai_client()

    if use_reasoner:
        # 使用 DeepSeek 推理模型（deepseek-reasoner）
        stream = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=budget_tokens,  # [Fix #2] 使用 budget_tokens 而非硬编码值
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                yield StreamChunk(
                    content=delta.reasoning_content,
                    chunk_type=ChunkType.THINKING,
                )
            if hasattr(delta, 'content') and delta.content:
                yield StreamChunk(
                    content=delta.content,
                    chunk_type=ChunkType.ANSWER,
                )
    else:
        # 使用普通 chat 模型，通过系统提示词开启思考模式
        model = get_default_model()

        stream = client.chat.completions.create(
            model=model,
            messages=[
                # [Fix #3] 添加 DEEPSEEK_THINKING_PROMPT 作为 System Prompt，
                # 引导模型按 <thinking>/<answer> 标签格式输出
                {"role": "system", "content": DEEPSEEK_THINKING_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            temperature=0.4,
            max_tokens=budget_tokens,  # [Fix #2] 使用 budget_tokens 而非硬编码值
        )

        # [Fix #5] 使用通用解析器，与 stream_cot_prompt 共享逻辑
        yield from _parse_tagged_stream(stream, think_tag="thinking", answer_tag="answer")
