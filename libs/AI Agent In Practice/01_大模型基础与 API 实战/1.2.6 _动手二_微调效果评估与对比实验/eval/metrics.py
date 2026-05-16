from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch                                  # [Fix #9] 顶部统一导入，与 inference.py 一致
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

# --------------------------------------------------------------------------- #
# ROUGE 评估
# --------------------------------------------------------------------------- #

@dataclass
class RougeResult:
    rouge1: float
    rouge2: float
    rougeL: float


def compute_rouge(
    predictions: list[str],
    references: list[str],
    lang: str = "zh",
) -> RougeResult:
    """
    计算 ROUGE 分数。
    
    中文注意事项：
    - rouge-score 库默认按空格分词，中文需要逐字处理
    - 解决方案：将字符用空格隔开，让库按"字"级别统计 n-gram
    
    Args:
        predictions: 模型输出列表
        references: 标准答案列表
        lang: "zh" 表示中文（逐字切分），"en" 表示英文（空格切分）
    """
    def tokenize_zh(text: str) -> str:
        """中文逐字加空格，让 ROUGE 按字级别计算。"""
        return " ".join(list(text.replace(" ", "")))

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,           # 中文不需要 stemming
    )

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        if lang == "zh":
            pred_tok = tokenize_zh(pred)
            ref_tok = tokenize_zh(ref)
        else:
            pred_tok, ref_tok = pred, ref

        result = scorer.score(ref_tok, pred_tok)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)

    return RougeResult(
        rouge1=float(np.mean(scores["rouge1"])),
        rouge2=float(np.mean(scores["rouge2"])),
        rougeL=float(np.mean(scores["rougeL"])),
    )


# --------------------------------------------------------------------------- #
# BERTScore 评估
# --------------------------------------------------------------------------- #

@dataclass
class BertScoreResult:
    precision: float
    recall: float
    f1: float


def compute_bert_score(
    predictions: list[str],
    references: list[str],
    model_type: str = "bert-base-chinese",
) -> BertScoreResult:
    """
    计算 BERTScore。
    
    选型说明：
    - 中文首选 bert-base-chinese，精度与速度平衡最佳
    - 如果追求更高精度，可换 hfl/chinese-roberta-wwm-ext（慢约 2x）
    - 英文场景首选 microsoft/deberta-xlarge-mnli（bert_score 官方推荐）
    
    ⚠️ 首次运行会下载约 400MB 的 BERT 模型，请确保网络畅通
    """
    P, R, F1 = bert_score_fn(
        cands=predictions,
        refs=references,
        model_type=model_type,
        lang="zh",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",  # [Fix #9] 使用顶部 import 的 torch
    )
    return BertScoreResult(
        precision=float(P.mean()),
        recall=float(R.mean()),
        f1=float(F1.mean()),
    )