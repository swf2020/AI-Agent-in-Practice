"""
流式 CoT 解析核心模块。
支持两种模式：
  1. CoT Prompt 模式（OpenAI/任意模型）：通过 <think>...</think> 标签区分思考与回答
  2. Claude Extended Thinking 模式：通过原生 thinking_delta / text_delta 事件区分
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

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
    model: str = "gpt-4o",
    temperature: float = 0.6,
) -> Generator[StreamChunk, None, None]:
    """
    使用 CoT System Prompt 驱动流式输出，实时解析 <think> 标签。

    Args:
        prompt: 用户问题
        model:  任何 OpenAI 兼容模型名称
        temperature: 采样温度，推理任务建议 0.3-0.7

    Yields:
        StreamChunk，chunk_type 区分 thinking 和 answer
    """
    client = OpenAI()

    # 缓冲区：累积已接收的全部文本，用于标签边界检测
    accumulated = ""
    # 状态机：当前是否在 <think> 块内
    in_thinking = False
    # 已处理到的缓冲区位置，避免重复 yield
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

        # ── 标签检测与状态切换 ──────────────────────────────
        # 检测 <think> 开始标签
        if not in_thinking and "<think>" in accumulated:
            think_start = accumulated.index("<think>") + len("<think>")
            # 跳过标签本身，从标签后的内容开始
            processed_up_to = think_start
            in_thinking = True

        # 检测 </think> 结束标签
        if in_thinking and "</think>" in accumulated:
            think_end = accumulated.index("</think>")
            # yield 标签前的剩余 thinking 内容
            remaining_think = accumulated[processed_up_to:think_end]
            if remaining_think:
                yield StreamChunk(content=remaining_think, chunk_type=ChunkType.THINKING)
            processed_up_to = think_end + len("</think>")
            in_thinking = False

        # 检测 <answer> 开始标签
        if not in_thinking and "<answer>" in accumulated[processed_up_to:]:
            answer_tag_pos = accumulated.index("<answer>", processed_up_to) + len("<answer>")
            processed_up_to = answer_tag_pos

        # ── yield 当前有效内容 ──────────────────────────────
        # 只 yield 尚未处理的、且不包含标签的内容
        new_content = accumulated[processed_up_to:]

        # 过滤掉结束标签
        for tag in ["</think>", "</answer>", "<answer>", "<think>"]:
            if tag in new_content:
                new_content = new_content[:new_content.index(tag)]
                break

        if new_content:
            chunk_type = ChunkType.THINKING if in_thinking else ChunkType.ANSWER
            yield StreamChunk(content=new_content, chunk_type=chunk_type)
            processed_up_to += len(new_content)


# ─────────────────────────────────────────────
#  模式二：Claude Extended Thinking（原生推理流）
# ─────────────────────────────────────────────

def stream_extended_thinking(
    prompt: str,
    budget_tokens: int = 5000,
) -> Generator[StreamChunk, None, None]:
    """
    使用 Claude claude-3-7-sonnet 的 Extended Thinking，
    原生区分模型"内部思考"与"最终回复"。

    Args:
        prompt:       用户问题
        budget_tokens: 分配给思考过程的最大 token 数（越大推理越充分，但越慢越贵）

    Yields:
        StreamChunk，chunk_type=thinking 对应模型原始思考，answer 对应最终回复

    ⚠️ 注意：Extended Thinking 目前要求 temperature=1（固定值），
       且计费按 thinking token + output token 合计计算。
    """
    client = Anthropic()

    # Extended Thinking 需要通过 betas 参数启用交错思考流
    with client.messages.stream(
        model="claude-3-7-sonnet-20250219",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": budget_tokens,
        },
        messages=[{"role": "user", "content": prompt}],
        betas=["interleaved-thinking-2025-05-14"],
    ) as stream:
        # Extended Thinking 的流事件类型比 OpenAI 更丰富
        # 需要监听 raw_stream_event 来区分 block 类型
        for event in stream:
            event_type = type(event).__name__

            # thinking block 中的增量内容
            if event_type == "RawContentBlockDeltaEvent":
                delta = event.delta
                if hasattr(delta, "thinking"):
                    # ThinkingDelta：模型原始思考内容
                    if delta.thinking:
                        yield StreamChunk(
                            content=delta.thinking,
                            chunk_type=ChunkType.THINKING,
                        )
                elif hasattr(delta, "text"):
                    # TextDelta：最终回复内容
                    if delta.text:
                        yield StreamChunk(
                            content=delta.text,
                            chunk_type=ChunkType.ANSWER,
                        )