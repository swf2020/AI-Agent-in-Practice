# run_eval.py — 端到端冒烟测试，直接运行验证整个评估流水线
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from eval.inference import EvalSample, InferenceConfig, ModelInferencer
from eval.metrics import compute_bert_score, compute_rouge
from eval.llm_judge import batch_judge
from eval.ablation import AblationSuite, build_example_suite
from eval.overfitting import load_trainer_state, plot_loss_curves

load_dotenv()

# --------------------------------------------------------------------------- #
# 配置区（按实际路径修改）
# --------------------------------------------------------------------------- #
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "./outputs/qwen2.5-7b-customer-service/final"
CHECKPOINT_DIR = "./outputs/qwen2.5-7b-customer-service/checkpoint-last"

# 测试集（实际使用时替换为从文件加载的完整 50 条）
TEST_SAMPLES = [
    EvalSample(0, "我的快递三天没动静了，怎么办？", "您好，请您提供快递单号，我们立即为您查询物流状态并协调处理。"),
    EvalSample(1, "想申请退款，需要什么材料？", "退款需提供订单号、退款原因及商品照片（如有质量问题）。"),
    EvalSample(2, "账号被锁定了如何解锁？", "请通过注册手机号接收验证码完成身份验证，或联系人工客服协助处理。"),
]

# --------------------------------------------------------------------------- #
# Step 1：三方推理
# --------------------------------------------------------------------------- #
print("=" * 60)
print("Step 1: 三方推理")

configs = {
    "base": InferenceConfig(
        model_mode="base",
        base_model_path=BASE_MODEL,
        system_prompt="你是一个助手。",   # 最简 system prompt，体现基座原始能力
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

predictions: dict[str, list[str]] = {}
for name, cfg in configs.items():
    inferencer = ModelInferencer(cfg)
    predictions[name] = inferencer.batch_generate(TEST_SAMPLES)
    del inferencer   # 释放显存，避免 OOM

# --------------------------------------------------------------------------- #
# Step 2：自动评估
# --------------------------------------------------------------------------- #
print("\nStep 2: 自动评估")
references = [s.reference for s in TEST_SAMPLES]

for name, preds in predictions.items():
    rouge = compute_rouge(preds, references, lang="zh")
    bert = compute_bert_score(preds, references)
    print(f"\n[{name}]")
    print(f"  ROUGE-L:         {rouge.rougeL:.4f}")
    print(f"  BERTScore-F1:    {bert.f1:.4f}")

# --------------------------------------------------------------------------- #
# Step 3：LLM Judge
# --------------------------------------------------------------------------- #
print("\nStep 3: LLM Judge")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

for name, preds in predictions.items():
    scores = batch_judge(TEST_SAMPLES, preds, client, model="gpt-4o-mini")
    avg_total = sum(s.total for s in scores) / len(scores)
    print(f"[{name}] LLM Judge 均分: {avg_total:.2f}/5.00")

# --------------------------------------------------------------------------- #
# Step 4：消融实验可视化
# --------------------------------------------------------------------------- #
print("\nStep 4: 消融实验可视化")
suite = build_example_suite()   # 替换为你实际的训练结果
suite.plot("rank", "ablation_rank.png")
suite.plot("epoch", "ablation_epoch.png")
suite.plot("data_size", "ablation_data_size.png")

# --------------------------------------------------------------------------- #
# Step 5：过拟合诊断
# --------------------------------------------------------------------------- #
print("\nStep 5: 过拟合诊断")
if Path(CHECKPOINT_DIR).exists():
    curve = load_trainer_state(CHECKPOINT_DIR)
    plot_loss_curves(curve, CHECKPOINT_DIR, "loss_curves.png")
else:
    print(f"⚠️ 找不到 checkpoint 目录 {CHECKPOINT_DIR}，跳过过拟合诊断")

print("\n✅ 评估流水线运行完成")