"""
三项对抗性测试：位置偏差 / 冗长偏好 / 一致性
"""
import asyncio
from dataclasses import dataclass

from judge.evaluator import judge_single, JudgeResult


# ── 测试一：位置偏差 ─────────────────────────────────────────
async def test_position_bias(
    source: str,
    reference: str,
    good_translation: str,    # 人工确认质量较高的译文
    bad_translation: str,     # 人工确认质量较差的译文
    prompt_version: str,
    judge_model: str = "gpt-4o",
) -> dict:
    """
    构造双译文对比 Prompt，交换顺序后观察评分变化
    
    检验逻辑：如果裁判客观，交换 A/B 顺序后，好译文的分数应该始终更高
    如果出现"顺序影响评分"，说明裁判存在位置偏差
    """
    # 顺序一：好在前，差在后
    result_order1_good = await judge_single(
        item_id="bias_test_good_first",
        source=source,
        translation=good_translation,
        reference=reference,
        translator="good",
        prompt_version=prompt_version,
        judge_model=judge_model,
    )
    result_order1_bad = await judge_single(
        item_id="bias_test_bad_first",
        source=source,
        translation=bad_translation,
        reference=reference,
        translator="bad",
        prompt_version=prompt_version,
        judge_model=judge_model,
    )

    # 顺序二：差在前，好在后
    # 这里我们复用评估函数，但实际上 pairwise 测试需要同时呈现两个译文
    # 简化版：分别评分，检验相对排名是否稳定
    result_order2_bad = await judge_single(
        item_id="bias_test_bad_second",
        source=source,
        translation=bad_translation,
        reference=reference,
        translator="bad_reversed",
        prompt_version=prompt_version,
        judge_model=judge_model,
    )
    result_order2_good = await judge_single(
        item_id="bias_test_good_second",
        source=source,
        translation=good_translation,
        reference=reference,
        translator="good_reversed",
        prompt_version=prompt_version,
        judge_model=judge_model,
    )

    good_scores = [result_order1_good.overall, result_order2_good.overall]
    bad_scores = [result_order1_bad.overall, result_order2_bad.overall]
    score_delta = abs(good_scores[0] - good_scores[1])

    return {
        "good_scores": good_scores,
        "bad_scores": bad_scores,
        "good_score_variance": score_delta,
        "ranking_consistent": all(g > b for g, b in zip(good_scores, bad_scores)),
        "verdict": "✅ 位置偏差可控" if score_delta < 0.5 else "❌ 存在明显位置偏差",
    }


# ── 测试二：冗长偏好 ─────────────────────────────────────────
async def test_verbosity_bias(
    source: str,
    reference: str,
    good_translation: str,
    bad_translation: str,
    prompt_version: str,
    judge_model: str = "gpt-4o",
) -> dict:
    """
    将差翻译填充到与好翻译相同长度，观察评分是否上升
    
    填充策略：在差翻译末尾添加无关但听起来正式的补充说明
    如果填充后分数上升 > 0.5，说明裁判存在冗长偏好
    """
    # 计算目标长度差
    good_len = len(good_translation)
    bad_len = len(bad_translation)
    padding_needed = max(0, good_len - bad_len)

    # 填充无意义但格式正式的内容（模拟冗长但非信息的译文）
    padding = "（本译文已经过专业审校，确保术语使用符合行业标准规范。）"
    padded_bad = bad_translation + padding[:padding_needed] if padding_needed > 0 else bad_translation

    # 评估原始差译文
    result_bad_original = await judge_single(
        item_id="verbosity_bad_original",
        source=source, translation=bad_translation, reference=reference,
        translator="bad_original", prompt_version=prompt_version, judge_model=judge_model,
    )

    # 评估填充后的差译文
    result_bad_padded = await judge_single(
        item_id="verbosity_bad_padded",
        source=source, translation=padded_bad, reference=reference,
        translator="bad_padded", prompt_version=prompt_version, judge_model=judge_model,
    )

    score_change = result_bad_padded.overall - result_bad_original.overall

    return {
        "original_score": result_bad_original.overall,
        "padded_score": result_bad_padded.overall,
        "score_change": round(score_change, 2),
        "verdict": "✅ 无明显冗长偏好" if abs(score_change) < 0.5 else f"❌ 填充后分数变化 {score_change:+.1f}，存在冗长偏好",
    }


# ── 测试三：一致性（评分稳定性）──────────────────────────────
async def test_consistency(
    items: list[dict],
    prompt_version: str,
    judge_model: str = "gpt-4o",
    runs: int = 5,
) -> dict:
    """
    同一输入重复评估5次，计算标准差
    标准差 < 0.3 认为一致性合格
    """
    from judge.evaluator import judge_batch
    import numpy as np

    results = await judge_batch(
        items=items,
        prompt_version=prompt_version,
        judge_model=judge_model,
        runs=runs,
    )

    # 按 item_id 分组，计算每条数据5次评分的标准差
    from collections import defaultdict
    scores_by_item: dict[str, list[float]] = defaultdict(list)
    for r in results:
        scores_by_item[r.item_id].append(r.overall)

    stds = [np.std(scores) for scores in scores_by_item.values()]
    mean_std = float(np.mean(stds))
    max_std = float(np.max(stds))

    return {
        "mean_std": round(mean_std, 3),
        "max_std": round(max_std, 3),
        "per_item_std": {k: round(float(np.std(v)), 3) for k, v in scores_by_item.items()},
        "verdict": "✅ 一致性合格" if mean_std < 0.3 else f"❌ 平均标准差 {mean_std:.3f}，一致性不足",
    }