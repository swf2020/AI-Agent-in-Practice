"""
Unsloth 高效微调加速实战 — 主入口
端到端冒烟测试：验证 Unsloth 微调全流程可跑通
在 Colab T4 环境约需 15-20 分钟
模型配置通过 core_config.py 统一管理，修改 ACTIVE_MODEL_KEY 即可切换。
"""

# ─── 安装（仅 Colab 需要，本地已安装跳过）───
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps trl==0.13.0 peft==0.14.0 accelerate==1.2.1

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch

# 从统一配置获取模型 ID
from core_config import get_unsloth_id, ACTIVE_MODEL_KEY

print("=" * 60)
print("Unsloth 微调冒烟测试")
print(f"当前模型：{ACTIVE_MODEL_KEY} -> {get_unsloth_id()}")
print("=" * 60)

# Step 1: 加载模型
print("\n[1/4] 加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=get_unsloth_id(),
    max_seq_length=1024,  # 冒烟测试用较小值，加快速度
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=8, target_modules=["q_proj", "v_proj"],
    lora_alpha=8, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)
print(f"  显存占用：{torch.cuda.memory_allocated()/1e9:.2f} GB")

# Step 2: 准备数据集
print("\n[2/4] 准备数据集...")
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:200]", trust_remote_code=True)

def format_fn(examples):
    texts = []
    for conv in examples["conversations"]:
        messages = [
            {"role": "user" if t["from"] == "human" else "assistant",
             "content": t["value"]}
            for t in conv
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return {"text": texts}

dataset = dataset.map(format_fn, batched=True, remove_columns=dataset.column_names)
print(f"  数据集大小：{len(dataset)} 条")

# Step 3: 训练（仅跑 20 步验证流程可通）
print("\n[3/4] 启动训练（20 steps 冒烟）...")
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=True,
    args=TrainingArguments(
        output_dir="./smoke_test_output",
        max_steps=20,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        logging_steps=5,
        report_to="none",
        seed=42,
    ),
)
stats = trainer.train()
print(f"  最终 Loss：{stats.metrics['train_loss']:.4f}")
print(f"  训练速度：{stats.metrics['train_samples_per_second']:.1f} samples/s")

# Step 4: 推理验证
print("\n[4/4] 推理验证...")
FastLanguageModel.for_inference(model)
test_input = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain what is machine learning in one sentence."}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True,
                         pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print(f"\n模型输出：{response}")
print("\n" + "=" * 60)
print("✅ 冒烟测试通过！Unsloth 微调全流程正常")
print(f"峰值显存：{torch.cuda.max_memory_reserved()/1e9:.2f} GB")
print("=" * 60)
