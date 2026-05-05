"""
TradingAgents 实战项目 — 主入口

统一调度四个实验：
  1. 美股分析（NVDA / TSLA）
  2. A股适配（AKShare）
  3. Checkpoint 断点续跑
  4. 多模型对比

用法：
    python main.py 1          # 运行实验一
    python main.py 2          # 运行实验二
    python main.py 3          # 运行实验三
    python main.py 4          # 运行实验四
    python main.py test       # 运行冒烟测试
"""
import sys


def run_experiment_1():
    """实验一：分析美股 NVDA，解读 TradingAgents 五级评级输出"""
    from experiment_1_basic_analysis import run_analysis
    from experiment_1_parse_output import display_decision

    result = run_analysis(
        ticker="NVDA",
        analysis_date="2025-01-10",
        risk_level="neutral",
    )
    display_decision(result)
    return result


def run_experiment_2():
    """实验二：用 AKShare 适配器分析 A 股（贵州茅台）"""
    from experiment_2_astock import analyze_astock

    result = analyze_astock("600519", "2025-01-10")
    return result


def run_experiment_3():
    """实验三：开启 Checkpoint 实现断点续跑"""
    from experiment_3_checkpoint import analyze_with_checkpoint, list_checkpoints

    result = analyze_with_checkpoint("NVDA", "2025-01-10")
    print(f"Thread ID: {result['thread_id']}")
    list_checkpoints("NVDA")
    return result


def run_experiment_4():
    """实验四：多模型对比（GPT-4o vs DeepSeek-V3）"""
    from experiment_4_model_comparison import compare_models, display_comparison

    results = compare_models("NVDA", "2025-01-10", models=["gpt4o", "deepseek"])
    display_comparison(results)
    return results


def run_smoke_test():
    """运行冒烟测试"""
    from smoke_test import test_astock_adapter, test_checkpoint_save, test_model_config

    test_astock_adapter()
    test_checkpoint_save()
    test_model_config()
    print("所有非 LLM 测试通过")


if __name__ == "__main__":
    experiments = {
        "1": ("美股分析（实验一）", run_experiment_1),
        "2": ("A股适配（实验二）", run_experiment_2),
        "3": ("Checkpoint 续跑（实验三）", run_experiment_3),
        "4": ("多模型对比（实验四）", run_experiment_4),
        "test": ("冒烟测试", run_smoke_test),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in experiments:
        print("用法: python main.py [1|2|3|4|test]")
        for key, (desc, _) in experiments.items():
            print(f"  {key} — {desc}")
        sys.exit(1)

    label, func = experiments[sys.argv[1]]
    print(f"正在运行: {label}")
    func()
