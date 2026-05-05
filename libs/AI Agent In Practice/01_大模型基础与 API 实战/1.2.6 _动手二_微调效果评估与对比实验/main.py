"""主入口：微调效果评估与对比实验"""
import os
import sys
import warnings

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# 抑制 HuggingFace transformers 库的冗余 warning（保留其他库的 warning）
warnings.filterwarnings("ignore", module="transformers")


def run_full_eval():
    """运行完整的评估流水线。"""
    from run_eval import TEST_SAMPLES, BASE_MODEL, LORA_PATH, CHECKPOINT_DIR
    from eval.inference import InferenceConfig, ModelInferencer
    from eval.metrics import compute_bert_score, compute_rouge
    from eval.llm_judge import batch_judge
    from eval.ablation import build_example_suite
    from eval.overfitting import load_trainer_state, plot_loss_curves
    from openai import OpenAI
    from core_config import get_api_key, get_litellm_id
    from pathlib import Path

    print("=" * 60)
    print("Step 1: 三方推理")

    configs = {
        "base": InferenceConfig(
            model_mode="base",
            base_model_path=BASE_MODEL,
            system_prompt="你是一个助手。",
        ),
        "prompt_eng": InferenceConfig(
            model_mode="prompt_eng",
            base_model_path=BASE_MODEL,
            system_prompt=(
                "你是一位专业的电商客服，负责解答用户关于订单、物流、退款的问题。"
                "回答要简洁（不超过80字）、友好、具体，不要使用模糊表述。"
            ),
        ),
        "finetuned": InferenceConfig(
            model_mode="finetuned",
            base_model_path=BASE_MODEL,
            lora_adapter_path=LORA_PATH,
            system_prompt="你是一个专业的客服助手，请简洁、准确地回答用户问题。",
        ),
    }

    predictions = {}
    for name, cfg in configs.items():
        inferencer = ModelInferencer(cfg)
        predictions[name] = inferencer.batch_generate(TEST_SAMPLES)
        del inferencer

    print("\nStep 2: 自动评估")
    references = [s.reference for s in TEST_SAMPLES]
    for name, preds in predictions.items():
        rouge = compute_rouge(preds, references, lang="zh")
        bert = compute_bert_score(preds, references)
        print(f"\n[{name}]")
        print(f"  ROUGE-L:         {rouge.rougeL:.4f}")
        print(f"  BERTScore-F1:    {bert.f1:.4f}")

    print("\nStep 3: LLM Judge")
    client = OpenAI(api_key=get_api_key("GPT-4o-mini"))
    for name, preds in predictions.items():
        scores = batch_judge(TEST_SAMPLES, preds, client, model=get_litellm_id("GPT-4o-mini"))
        avg_total = sum(s.total for s in scores) / len(scores)
        print(f"[{name}] LLM Judge 均分: {avg_total:.2f}/5.00")

    print("\nStep 4: 消融实验可视化")
    suite = build_example_suite()
    suite.plot("rank", "ablation_rank.png")
    suite.plot("epoch", "ablation_epoch.png")
    suite.plot("data_size", "ablation_data_size.png")

    print("\nStep 5: 过拟合诊断")
    if Path(CHECKPOINT_DIR).exists():
        curve = load_trainer_state(CHECKPOINT_DIR)
        plot_loss_curves(curve, CHECKPOINT_DIR, "loss_curves.png")
    else:
        print(f"找不到 checkpoint 目录 {CHECKPOINT_DIR}，跳过过拟合诊断")

    print("\n评估流水线运行完成")


if __name__ == "__main__":
    run_full_eval()
