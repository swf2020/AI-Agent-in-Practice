"""[Fix #2] 共享评估流水线 — 供 main.py 和 run_eval.py 复用"""
from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from eval.inference import EvalSample, InferenceConfig, ModelInferencer
from eval.metrics import compute_bert_score, compute_rouge
from eval.llm_judge import batch_judge
from eval.ablation import build_example_suite
from eval.overfitting import load_trainer_state, plot_loss_curves
from core_config import get_api_key, get_litellm_id


def run_evaluation_pipeline(
    test_samples: list[EvalSample],
    base_model: str,
    lora_path: str | None = None,
    checkpoint_dir: str | None = None,
    judge_model_key: str = "GPT-4o-mini",
) -> dict:
    """
    运行完整的评估流水线，返回各方案评测结果字典。

    Args:
        test_samples: 测试样本列表（EvalSample）
        base_model: 基座模型路径（如 "Qwen/Qwen2.5-7B-Instruct"）
        lora_path: LoRA adapter 路径（None 则跳过微调方案）
        checkpoint_dir: checkpoint 目录（用于过拟合诊断，None 则跳过）
        judge_model_key: LLM Judge 使用的模型 key

    Returns:
        {
            "<方案名>": {
                "predictions": [...],
                "rouge": RougeResult,
                "bert_score": BertScoreResult,
                "judge_score": float,
            },
            ...
        }
    """
    results: dict = {}

    # ----------------------------------------------------------------------- #
    # Step 1：三方推理
    # ----------------------------------------------------------------------- #
    print("=" * 60)
    print("Step 1: 三方推理")

    configs = {
        "base": InferenceConfig(
            model_mode="base",
            base_model_path=base_model,
            system_prompt="你是一个助手。",
        ),
        "prompt_eng": InferenceConfig(
            model_mode="prompt_eng",
            base_model_path=base_model,
            system_prompt=(
                "你是一位专业的电商客服，负责解答用户关于订单、物流、退款的问题。"
                "回答要简洁（不超过80字）、友好、具体，不要使用模糊表述。"
            ),
        ),
    }
    if lora_path:
        configs["finetuned"] = InferenceConfig(
            model_mode="finetuned",
            base_model_path=base_model,
            lora_adapter_path=lora_path,
            system_prompt="你是一个专业的客服助手，请简洁、准确地回答用户问题。",
        )

    predictions: dict[str, list[str]] = {}
    for name, cfg in configs.items():
        inferencer = ModelInferencer(cfg)
        predictions[name] = inferencer.batch_generate(test_samples)
        del inferencer  # 释放显存，避免 OOM

    results = {name: {"predictions": preds} for name, preds in predictions.items()}

    # ----------------------------------------------------------------------- #
    # Step 2：自动评估
    # ----------------------------------------------------------------------- #
    print("\nStep 2: 自动评估")
    references = [s.reference for s in test_samples]

    for name, preds in predictions.items():
        rouge = compute_rouge(preds, references, lang="zh")
        bert = compute_bert_score(preds, references)
        results[name]["rouge"] = rouge
        results[name]["bert_score"] = bert
        print(f"\n[{name}]")
        print(f"  ROUGE-L:         {rouge.rougeL:.4f}")
        print(f"  BERTScore-F1:    {bert.f1:.4f}")

    # ----------------------------------------------------------------------- #
    # Step 3：LLM Judge
    # ----------------------------------------------------------------------- #
    print("\nStep 3: LLM Judge")

    # [Fix #8] 显式检查 API Key，避免静默失败
    api_key = get_api_key(judge_model_key)
    if api_key is None:
        print(f"⚠️  未设置 {judge_model_key} 的 API Key，跳过 LLM Judge 评估")
        print("   请运行: export OPENAI_API_KEY='***' 或在 .env 中配置")
    else:
        client = OpenAI(api_key=api_key)
        for name, preds in predictions.items():
            scores = batch_judge(test_samples, preds, client, model=get_litellm_id(judge_model_key))
            avg_total = sum(s.total for s in scores) / len(scores)
            results[name]["judge_score"] = avg_total
            print(f"[{name}] LLM Judge 均分: {avg_total:.2f}/5.00")

    # ----------------------------------------------------------------------- #
    # Step 4：消融实验可视化
    # ----------------------------------------------------------------------- #
    print("\nStep 4: 消融实验可视化")
    suite = build_example_suite()
    suite.plot("rank", "ablation_rank.png")
    suite.plot("epoch", "ablation_epoch.png")
    suite.plot("data_size", "ablation_data_size.png")

    # ----------------------------------------------------------------------- #
    # Step 5：过拟合诊断
    # ----------------------------------------------------------------------- #
    print("\nStep 5: 过拟合诊断")
    if checkpoint_dir and Path(checkpoint_dir).exists():
        curve = load_trainer_state(checkpoint_dir)
        plot_loss_curves(curve, checkpoint_dir, "loss_curves.png")
    else:
        if checkpoint_dir:
            print(f"⚠️  找不到 checkpoint 目录 {checkpoint_dir}，跳过过拟合诊断")

    print("\n✅ 评估流水线运行完成")
    return results
