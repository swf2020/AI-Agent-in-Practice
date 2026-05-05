"""
端到端冒烟测试：验证微调后模型效果
文件：smoke_test.py
依赖：pip install ollama（或直接用 transformers 加载推理）
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def test_with_transformers(adapter_path: str = "./outputs/qwen2.5-7b-customer-service/final_adapter") -> None:
    """直接加载 adapter 推理（无需合并，适合快速验证）。"""
    from peft import PeftModel
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    test_cases = [
        "我的订单还没发货，已经等了5天了",
        "收到的商品和图片不一样，要怎么处理？",
        "如何修改收货地址？",
    ]
    
    for query in test_cases:
        messages = [
            {"role": "system", "content": "你是一位专业的电商客服助手，负责解答用户关于订单、物流、退换货等问题。"},
            {"role": "user",   "content": query},
        ]
        
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(f"用户：{query}")
        print(f"客服：{response}")
        print("-" * 60)


if __name__ == "__main__":
    test_with_transformers()