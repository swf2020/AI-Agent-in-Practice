# run_eval.py — 端到端冒烟测试，直接运行验证整个评估流水线  [Fix #2] 复用 pipeline.py
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from eval.inference import EvalSample
from eval.pipeline import run_evaluation_pipeline

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
# 运行评估流水线
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_evaluation_pipeline(
        test_samples=TEST_SAMPLES,
        base_model=BASE_MODEL,
        lora_path=LORA_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
    )
