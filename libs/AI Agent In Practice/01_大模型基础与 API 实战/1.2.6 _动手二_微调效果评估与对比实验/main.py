"""主入口：微调效果评估与对比实验 — [Fix #2] 复用 eval/pipeline.py 共享流水线"""
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
    # [Fix #2] 配置与流水线执行解耦：run_eval.py 定义配置，pipeline.py 执行逻辑
    from run_eval import TEST_SAMPLES, BASE_MODEL, LORA_PATH, CHECKPOINT_DIR
    from eval.pipeline import run_evaluation_pipeline

    run_evaluation_pipeline(
        test_samples=TEST_SAMPLES,
        base_model=BASE_MODEL,
        lora_path=LORA_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
    )


if __name__ == "__main__":
    run_full_eval()
