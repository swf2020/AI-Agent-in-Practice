"""
对外暴露的标准评估接口
使用方式：
    from judge import evaluate_translation, evaluate_batch
"""
import asyncio
from judge.evaluator import judge_single, judge_batch, JudgeResult
from judge.prompts import PROMPT_VERSIONS


async def evaluate_translation(
    source: str,
    translation: str,
    reference: str,
    prompt_version: str = "v3_with_reference",
    judge_model: str = "gpt-4o",
) -> dict:
    """
    评估单条翻译质量（异步）

    Args:
        source: 原文
        translation: 待评译文
        reference: 参考译文（版本 v3/v4 需要）
        prompt_version: 裁判 Prompt 版本，推荐 v3_with_reference
        judge_model: 裁判模型，推荐 gpt-4o

    Returns:
        {faithfulness, fluency, terminology, overall, key_issues, reasoning}

    Example:
        result = await evaluate_translation(
            source="The model achieved state-of-the-art performance.",
            translation="该模型达到了最先进的性能。",
            reference="该模型取得了最先进的性能。",
        )
        print(result["overall"])  # 4.7
    """
    result = await judge_single(
        item_id="single_eval",
        source=source,
        translation=translation,
        reference=reference,
        translator="unknown",
        prompt_version=prompt_version,
        judge_model=judge_model,
    )
    return {
        "faithfulness": result.faithfulness,
        "fluency": result.fluency,
        "terminology": result.terminology,
        "overall": result.overall,
        "key_issues": result.key_issues,
        "reasoning": result.reasoning,
        "latency_ms": result.latency_ms,
    }


async def evaluate_batch(
    items: list[dict[str, str]],
    prompt_version: str = "v3_with_reference",
    judge_model: str = "gpt-4o",
    concurrency: int = 10,
) -> list[dict]:
    """
    批量评估翻译质量（异步，自动并发）

    Args:
        items: [{"id": "...", "source": "...", "translation": "...", 
                  "reference": "...", "translator": "..."}]
        concurrency: 并发数，GPT-4o 建议 10，DeepSeek 可到 20

    性能参考：100条翻译，concurrency=10，约 30-45 秒完成
    """
    results = await judge_batch(
        items=items,
        prompt_version=prompt_version,
        judge_model=judge_model,
        runs=1,
        concurrency=concurrency,
    )
    return [
        {
            "id": r.item_id,
            "translator": r.translator,
            "faithfulness": r.faithfulness,
            "fluency": r.fluency,
            "terminology": r.terminology,
            "overall": r.overall,
            "key_issues": r.key_issues,
        }
        for r in results
    ]