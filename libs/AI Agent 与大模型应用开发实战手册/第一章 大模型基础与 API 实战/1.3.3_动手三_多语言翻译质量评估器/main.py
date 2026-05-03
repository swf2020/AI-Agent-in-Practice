# main.py —— 端到端冒烟测试，可直接复制运行

import asyncio
import json
import pandas as pd
from dataclasses import asdict
from dotenv import load_dotenv

load_dotenv()

from data.test_set import TEST_SET
from judge import evaluate_translation, evaluate_batch
from judge.evaluator import judge_batch
from judge.adversarial import test_position_bias, test_consistency
from analysis.metrics import (
    correlation_with_human,
    consistency_score,
    plot_heatmap,
    generate_summary_report,
)


async def main():
    print("=" * 60)
    print("多语言翻译质量评估器 —— 端到端验证")
    print("=" * 60)

    # ── 1. 单条评估冒烟测试 ──────────────────────────────────
    print("\n[1/4] 单条评估测试...")
    result = await evaluate_translation(
        item_id="test_001",
        source="The transformer architecture revolutionized natural language processing.",
        translation="Transformer架构彻底改变了自然语言处理领域。",
        reference="Transformer架构彻底革新了自然语言处理领域。",
        translator="test",
        prompt_version="v4_with_reference",
    )
    print(f"  评分结果：{json.dumps(asdict(result), ensure_ascii=False, indent=2)}")

    # ── 2. 位置偏差测试 ──────────────────────────────────────
    print("\n[2/4] 位置偏差测试...")
    bias_result = await test_position_bias(
        source=TEST_SET[0].source,
        reference=TEST_SET[0].reference,
        good_translation="Transformer架构通过用自注意力机制取代循环神经网络，彻底革新了自然语言处理领域。",
        bad_translation="这个transformer东西让NLP变得不一样了，用了新的机制。",
        prompt_version="v4_with_reference",
    )
    print(f"  位置偏差测试结果：{json.dumps(bias_result, ensure_ascii=False, indent=2)}")

    # ── 3. 一致性测试（取前5条数据，重复3次）────────────────
    print("\n[3/4] 一致性测试（5条 × 3次重复）...")
    sample_items = [
        {
            "id": item.id,
            "source": item.source,
            "translation": "Transformer架构通过用自注意力机制取代循环神经网络，彻底革新了自然语言处理领域。",
            "reference": item.reference,
            "translator": "sample",
        }
        for item in TEST_SET[:5]
    ]
    consistency_results = await judge_batch(
        items=sample_items,
        prompt_version="v4_with_reference",
        runs=3,
        concurrency=5,
    )

    # 重组为 (runs × items) 格式
    from collections import defaultdict
    scores_by_item: dict[str, list[float]] = defaultdict(list)
    for r in consistency_results:
        scores_by_item[r.item_id].append(r.overall)

    overall_scores_matrix = list(scores_by_item.values())  # n_items × runs
    transposed = list(map(list, zip(*overall_scores_matrix)))  # runs × n_items
    c_score = consistency_score(transposed)
    print(f"  一致性分数：{json.dumps(c_score, ensure_ascii=False, indent=2)}")
    print(f"  {c_score['verdict'] if 'verdict' in c_score else ('✅ 合格' if c_score['mean_std'] < 0.3 else '❌ 需改进')}")

    # ── 4. 相关系数（用测试集人工评分对齐）────────────────────
    print("\n[4/4] 与人工评分相关系数计算...")
    # 使用测试集中已有的人工评分做演示
    human_overall = [
        (item.human_scores["faithfulness"] + item.human_scores["fluency"] + item.human_scores["terminology"]) / 3
        for item in TEST_SET[:5]
    ]
    llm_overall = [r.overall for r in consistency_results if r.run_index == 0][:5]

    if len(llm_overall) == len(human_overall):
        corr = correlation_with_human(llm_overall, human_overall)
        print(f"  相关系数：")
        print(f"    Spearman 相关系数: {corr['spearman_r']}")
        print(f"    P 值: {corr['p_value']}")
        print(f"    显著性: {'显著' if corr['significant'] else '不显著'}")
        print(f"    解释: {corr['interpretation']}")
    else:
        print(f"  ⚠️  数据对齐失败，LLM={len(llm_overall)}，Human={len(human_overall)}")

    print("\n" + "=" * 60)
    print("✅ 冒烟测试通过！可以运行完整评估了。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())