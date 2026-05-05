from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from openai import OpenAI

from core_config import get_litellm_id as _get_litellm_id

JUDGE_SYSTEM_PROMPT = """你是一位严格的客服质量评审专家。
你将收到一个用户问题、一个标准答案和一个待评估答案。
请从以下三个维度对待评估答案打分（每项 1-5 分，整数）：

1. **准确性**（Accuracy）：答案是否正确、无错误信息
2. **完整性**（Completeness）：是否回答了问题的全部要点
3. **简洁性**（Conciseness）：是否避免了无关废话，直击用户需求

请严格按以下 JSON 格式输出，不要添加任何其他内容：
{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "conciseness": <1-5>,
  "reasoning": "<一句话说明最主要的扣分原因，满分则说明优点>"
}"""


@dataclass
class JudgeScore:
    accuracy: float
    completeness: float
    conciseness: float
    total: float          # 三项均值
    reasoning: str


def llm_judge(
    instruction: str,
    reference: str,
    prediction: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",   # 4o-mini 成本约为 4o 的 1/15，评估任务够用
    max_retries: int = 3,
) -> JudgeScore:
    """
    调用 LLM 对单条输出打分。
    
    模型选型：
    - 精度要求高：gpt-4o（约 $0.005/次评估）
    - 成本敏感：gpt-4o-mini（约 $0.0003/次，50条测试集总成本 < $0.02）
    - 离线场景：可替换为本地 Qwen2.5-72B-Instruct 作为 Judge
    """
    user_msg = f"""【用户问题】
{instruction}

【标准答案】
{reference}

【待评估答案】
{prediction}"""

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,          # Judge 必须用 temperature=0，保证评分稳定
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            scores = [data["accuracy"], data["completeness"], data["conciseness"]]
            return JudgeScore(
                accuracy=data["accuracy"],
                completeness=data["completeness"],
                conciseness=data["conciseness"],
                total=sum(scores) / 3,
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"LLM Judge 解析失败: {e}") from e
            time.sleep(2 ** attempt)    # 指数退避

    raise RuntimeError("LLM Judge 超过最大重试次数")


def batch_judge(
    samples: list,
    predictions: list[str],
    client: OpenAI,
    model: str | None = None,
) -> list[JudgeScore]:
    """批量评估，带速率限制保护。"""
    from tqdm import tqdm

    if model is None:
        model = _get_litellm_id("GPT-4o-mini")

    results = []
    for sample, pred in tqdm(zip(samples, predictions), total=len(samples), desc="LLM Judge"):
        score = llm_judge(
            instruction=sample.instruction,
            reference=sample.reference,
            prediction=pred,
            client=client,
            model=model,
        )
        results.append(score)
        time.sleep(0.1)    # 避免触发 API 速率限制（tier-1 约 500 RPM）
    return results