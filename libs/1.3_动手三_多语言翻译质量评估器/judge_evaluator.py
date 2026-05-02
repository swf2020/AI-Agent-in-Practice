"""
核心评估引擎：单条评估 + 批量异步评估 + 多次重复运行
"""
import asyncio
import json
import time
from typing import Any

from litellm import acompletion
from tenacity import retry, stop_after_attempt, wait_exponential

from judge.prompts import PROMPT_VERSIONS


@dataclass
class JudgeResult:
    item_id: str
    translator: str           # gpt4o / deepseek / google
    prompt_version: str       # v1_simple / v2_with_criteria / ...
    faithfulness: int
    fluency: int
    terminology: int
    overall: float
    key_issues: str
    reasoning: str
    run_index: int            # 第几次重复（用于一致性测试）
    latency_ms: float


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
async def judge_single(
    item_id: str,
    source: str,
    translation: str,
    reference: str,
    translator: str,
    prompt_version: str,
    judge_model: str = "gpt-4o",
    run_index: int = 0,
) -> JudgeResult:
    """
    对单条翻译进行 LLM-as-Judge 评估

    关键设计决策：
    1. temperature=0：裁判必须确定性，否则一致性测试毫无意义
    2. response_format=json_object：强制 JSON 输出，避免解析失败
    3. tenacity 重试：避免单次 API 抖动导致整批失败
    """
    prompt_template = PROMPT_VERSIONS[prompt_version]

    # V1/V2 没有 reference 参数，需要判断
    if "{reference}" in prompt_template:
        prompt = prompt_template.format(
            source=source, translation=translation, reference=reference
        )
    else:
        prompt = prompt_template.format(source=source, translation=translation)

    t0 = time.perf_counter()
    resp = await acompletion(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,                              # 裁判必须 temperature=0
        response_format={"type": "json_object"},   # 强制 JSON 模式
        max_tokens=300,                             # 评分输出不长，限制 token 节省成本
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = resp.choices[0].message.content
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败，原始输出：{raw[:200]}") from e

    return JudgeResult(
        item_id=item_id,
        translator=translator,
        prompt_version=prompt_version,
        faithfulness=int(data.get("faithfulness", 0)),
        fluency=int(data.get("fluency", 0)),
        terminology=int(data.get("terminology", 0)),
        overall=float(data.get("overall", 0.0)),
        key_issues=data.get("key_issues", ""),
        reasoning=data.get("reasoning", ""),
        run_index=run_index,
        latency_ms=latency_ms,
    )


async def judge_batch(
    items: list[dict],           # [{id, source, translation, reference, translator}]
    prompt_version: str,
    judge_model: str = "gpt-4o",
    runs: int = 1,
    concurrency: int = 10,       # 并发数，避免触发速率限制
) -> list[JudgeResult]:
    """
    批量异步评估

    Args:
        items: 待评估列表
        prompt_version: 使用哪个版本的裁判 Prompt
        runs: 重复评估次数（1=正常评估，5=一致性测试）
        concurrency: 最大并发请求数

    性能参考：
        30条 × 1次 × gpt-4o ≈ 15秒，约 $0.05
        30条 × 5次（一致性测试） ≈ 60秒，约 $0.25
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[JudgeResult] = []

    async def bounded_judge(item: dict, run_index: int) -> JudgeResult:
        async with semaphore:
            return await judge_single(
                item_id=item["id"],
                source=item["source"],
                translation=item["translation"],
                reference=item["reference"],
                translator=item["translator"],
                prompt_version=prompt_version,
                judge_model=judge_model,
                run_index=run_index,
            )

    tasks = [
        bounded_judge(item, run_index)
        for run_index in range(runs)
        for item in items
    ]

    completed = await asyncio.gather(*tasks, return_exceptions=True)
    for r in completed:
        if isinstance(r, Exception):
            print(f"⚠️  评估失败：{r}")
        else:
            results.append(r)

    return results