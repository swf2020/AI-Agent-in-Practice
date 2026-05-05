from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class InferenceConfig:
    """推理配置，三种方案共用同一接口。"""
    model_mode: Literal["base", "prompt_eng", "finetuned"]
    base_model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_adapter_path: str | None = None          # 仅 finetuned 需要
    system_prompt: str = "你是一个专业的客服助手，请简洁、准确地回答用户问题。"
    max_new_tokens: int = 256
    temperature: float = 0.1                       # 评估时用低温，减少随机性干扰
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvalSample:
    """单条测试样本。"""
    idx: int
    instruction: str
    reference: str                                 # 标准答案


# --------------------------------------------------------------------------- #
# 推理器
# --------------------------------------------------------------------------- #

class ModelInferencer:
    """
    统一封装三种方案的推理逻辑。
    
    设计决策：
    - 模型只加载一次，avoid OOM（每次实例化都加载会在多方案对比时耗尽显存）
    - bfloat16 加载：Qwen2.5 原生支持 bf16，精度损失可忽略，显存减半
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        print(f"[{config.model_mode}] 加载模型: {config.base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_path,
            trust_remote_code=True,
        )

        # 基座模型统一 bfloat16 加载
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if config.model_mode == "finetuned":
            # ⚠️ 必须先加载基座再套 LoRA，不能直接加载 adapter checkpoint
            if not config.lora_adapter_path:
                raise ValueError("finetuned 模式需要提供 lora_adapter_path")
            self.model = PeftModel.from_pretrained(
                base_model,
                config.lora_adapter_path,
                torch_dtype=torch.bfloat16,
            )
            # merge 权重后推理速度与基座相同，无额外开销
            self.model = self.model.merge_and_unload()
        else:
            self.model = base_model

        self.model.eval()

        self.gen_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def _build_prompt(self, instruction: str) -> str:
        """
        Prompt Engineering 方案比基座多一个精心设计的 system prompt，
        两者的区别仅在 system 内容，保证对比的公平性。
        """
        sys = self.config.system_prompt
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": instruction},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate(self, instruction: str) -> str:
        """单条推理，返回纯文本输出（不含 prompt 部分）。"""
        prompt = self._build_prompt(instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            generation_config=self.gen_config,
        )

        # 只取新生成的 token，去掉输入部分
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def batch_generate(self, samples: list[EvalSample]) -> list[str]:
        """批量推理，带进度条。"""
        from tqdm import tqdm
        return [self.generate(s.instruction) for s in tqdm(samples, desc=self.config.model_mode)]