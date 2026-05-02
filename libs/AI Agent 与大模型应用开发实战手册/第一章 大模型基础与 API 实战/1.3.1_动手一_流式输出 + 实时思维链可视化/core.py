"""
流式 CoT 解析核心模块。
支持多种模型和推理模式：
  1. CoT Prompt 模式（通用）：通过 <think>...</think> 标签区分思考与回答
  2. DeepSeek 推理模型：使用 deepseek-reasoner 原生推理能力
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 模型配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")


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


def get_openai_client():
    """根据环境变量返回正确配置的 OpenAI 客户端。"""
    if DEEPSEEK_API_KEY:
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    elif QWEN_API_KEY:
        return OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    else:
        return OpenAI()


def get_default_model():
    """根据环境变量返回默认模型名称。"""
    if DEEPSEEK_API_KEY:
        return "deepseek-chat"
    elif QWEN_API_KEY:
        return "qwen-plus"
    else:
        return "gpt-4o"


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
    model: str = None,
    temperature: float = 0.6,
) -> Generator[StreamChunk, None, None]:
    """
    使用 CoT System Prompt 驱动流式输出，实时解析 <think> 标签。

    Args:
        prompt: 用户问题
        model:  任何 OpenAI 兼容模型名称，默认根据环境变量自动选择
        temperature: 采样温度，推理任务建议 0.3-0.7

    Yields:
        StreamChunk，chunk_type 区分 thinking 和 answer
    """
    client = get_openai_client()
    model = model or get_default_model()

    accumulated = ""
    in_thinking = False
    processed_up_to = 0

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=temperature,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue

        accumulated += delta

        if not in_thinking and "<think>" in accumulated:
            think_start = accumulated.index("<think>") + len("<think>")
            processed_up_to = think_start
            in_thinking = True

        if in_thinking and "</think>" in accumulated:
            think_end = accumulated.index("</think>")
            remaining_think = accumulated[processed_up_to:think_end]
            if remaining_think:
                yield StreamChunk(content=remaining_think, chunk_type=ChunkType.THINKING)
            processed_up_to = think_end + len("</think>")
            in_thinking = False

        if not in_thinking and "<answer>" in accumulated[processed_up_to:]:
            answer_tag_pos = accumulated.index("<answer>", processed_up_to) + len("<answer>")
            processed_up_to = answer_tag_pos

        new_content = accumulated[processed_up_to:]

        for tag in ["</think>", "</answer>", "<answer>", "<think>"]:
            if tag in new_content:
                new_content = new_content[:new_content.index(tag)]
                break

        if new_content:
            chunk_type = ChunkType.THINKING if in_thinking else ChunkType.ANSWER
            yield StreamChunk(content=new_content, chunk_type=chunk_type)
            processed_up_to += len(new_content)


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
    budget_tokens: int = 8000,
    use_reasoner: bool = False,
) -> Generator[StreamChunk, None, None]:
    """
    DeepSeek 思考链推理。

    Args:
        prompt:       用户问题
        budget_tokens: 分配给思考过程的最大 token 数（越大推理越充分）
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
            max_tokens=16000,
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

        accumulated = ""
        in_thinking = False
        processed_up_to = 0

        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=True,
            temperature=0.4,
            max_tokens=16000,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue

            accumulated += delta

            if not in_thinking and "<thinking>" in accumulated:
                think_start = accumulated.index("<thinking>") + len("<thinking>")
                processed_up_to = think_start
                in_thinking = True

            if in_thinking and "</thinking>" in accumulated:
                think_end = accumulated.index("</thinking>")
                remaining_think = accumulated[processed_up_to:think_end]
                if remaining_think:
                    yield StreamChunk(content=remaining_think, chunk_type=ChunkType.THINKING)
                processed_up_to = think_end + len("</thinking>")
                in_thinking = False

            if not in_thinking and "<answer>" in accumulated[processed_up_to:]:
                answer_tag_pos = accumulated.index("<answer>", processed_up_to) + len("<answer>")
                processed_up_to = answer_tag_pos

            new_content = accumulated[processed_up_to:]

            for tag in ["</thinking>", "</answer>", "<answer>", "<thinking>"]:
                if tag in new_content:
                    new_content = new_content[:new_content.index(tag)]
                    break

            if new_content:
                chunk_type = ChunkType.THINKING if in_thinking else ChunkType.ANSWER
                yield StreamChunk(content=new_content, chunk_type=chunk_type)
                processed_up_to += len(new_content)
