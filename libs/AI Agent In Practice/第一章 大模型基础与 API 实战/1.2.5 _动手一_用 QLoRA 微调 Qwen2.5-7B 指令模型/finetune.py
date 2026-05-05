"""
QLoRA 微调 Qwen2.5-7B 指令模型 — 训练流程

使用 4-bit 量化加载基座模型，注入 LoRA adapter，在客服数据集上微调。
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from core_config import FINETUNE_BASE_MODEL


# ── 超参数配置 ─────────────────────────────────────
OUTPUT_DIR = "./outputs/qwen2.5-7b-customer-service"
DATASET_NAME = "customer_service_qa"  # 替换为实际数据集路径
MAX_SEQ_LENGTH = 512


def get_quantization_config() -> BitsAndBytesConfig:
    """4-bit 量化配置（QLoRA）。"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    """LoRA adapter 配置。"""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def format_sample(sample: dict) -> dict:
    """将数据集样本格式化为 ChatML 指令格式。"""
    messages = [
        {"role": "system", "content": "你是一位专业的电商客服助手，负责解答用户关于订单、物流、退换货等问题。"},
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_BASE_MODEL, trust_remote_code=True)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def run_finetuning() -> None:
    """执行完整的 QLoRA 微调流程。"""
    print(f"加载基座模型: {FINETUNE_BASE_MODEL}")

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        FINETUNE_BASE_MODEL, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. 加载量化基座模型
    bnb_config = get_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNE_BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. 注入 LoRA adapter
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 加载并格式化数据集
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    # 5. 训练
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    print("开始训练...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"训练完成，模型保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_finetuning()
