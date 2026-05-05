"""
QLoRA 微调 Qwen2.5-7B 指令模型 — 主入口

用法：
    python main.py                    # 运行冒烟测试（mock 模式）
    python main.py --test             # 运行冒烟测试
    python main.py --finetune         # 执行微调训练（需要 GPU 和数据集）
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from core_config import FINETUNE_BASE_MODEL, get_model_list


def run_smoke_test(adapter_path: str | None = None) -> None:
    """运行微调后模型的冒烟测试（加载 adapter 推理）。"""
    from smoke_test import test_with_transformers
    test_with_transformers(adapter_path=adapter_path or "./outputs/qwen2.5-7b-customer-service/final_adapter")


def run_finetune() -> None:
    """执行 QLoRA 微调训练流程。"""
    from finetune import run_finetuning
    run_finetuning()


def main():
    parser = argparse.ArgumentParser(description="QLoRA 微调 Qwen2.5-7B 指令模型")
    parser.add_argument("--test", action="store_true", help="运行冒烟测试")
    parser.add_argument("--finetune", action="store_true", help="执行微调训练")
    parser.add_argument("--adapter-path", default=None, help="微调 adapter 路径（用于测试）")
    parser.add_argument("--list-models", action="store_true", help="列出已注册模型")
    args = parser.parse_args()

    if args.list_models:
        print("已注册模型:", get_model_list())
        print(f"微调基座模型: {FINETUNE_BASE_MODEL}")
        return

    if args.finetune:
        print(f"开始微调，基座模型: {FINETUNE_BASE_MODEL}")
        run_finetune()
        return

    # 默认运行冒烟测试
    print("运行冒烟测试...")
    run_smoke_test(adapter_path=args.adapter_path)


if __name__ == "__main__":
    main()
