from __future__ import annotations

import json                              # [Fix #6] 标准库置顶
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class LossCurve:
    steps: list[int]
    train_loss: list[float]
    val_loss: list[float]


def load_trainer_state(checkpoint_dir: str | Path) -> LossCurve:
    """
    从 HuggingFace Trainer 的 trainer_state.json 提取 Loss 数据。
    
    trainer_state.json 自动生成于 checkpoint 目录，无需额外配置。
    """
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"找不到 trainer_state.json：{state_path}")

    with open(state_path) as f:
        state = json.load(f)

    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    for entry in state.get("log_history", []):
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            val_steps.append(entry["step"])
            val_losses.append(entry["eval_loss"])

    # 对齐步骤（取交集）
    common_steps = sorted(set(train_steps) & set(val_steps))
    train_map = dict(zip(train_steps, train_losses))
    val_map = dict(zip(val_steps, val_losses))

    return LossCurve(
        steps=common_steps,
        train_loss=[train_map[s] for s in common_steps],
        val_loss=[val_map[s] for s in common_steps],
    )


def detect_divergence_point(curve: LossCurve, patience: int = 3) -> int | None:
    """
    检测 train/val Loss 分叉点（过拟合起始步骤）。
    
    算法：滑动窗口检测 val_loss 连续上升而 train_loss 持续下降的起点。
    
    Args:
        curve: Loss 曲线数据
        patience: 连续多少步 val_loss 上升才确认分叉（默认 3，避免噪声误报）
    
    Returns:
        分叉起始 step，None 表示未检测到过拟合
    """
    rising_count = 0
    diverge_start = None

    for i in range(1, len(curve.steps)):
        # [Fix #3] 同时检查 val_loss 上升 AND train_loss 下降，避免欠拟合误判
        # [Fix #5] 去掉未使用的 gaps 变量，直接比对原始 loss 值
        val_rising = curve.val_loss[i] > curve.val_loss[i - 1]
        train_falling = curve.train_loss[i] < curve.train_loss[i - 1]
        if val_rising and train_falling:
            rising_count += 1
            if rising_count == patience and diverge_start is None:
                diverge_start = curve.steps[i - patience + 1]
        else:
            rising_count = 0

    return diverge_start


def plot_loss_curves(
    curve: LossCurve,
    checkpoint_dir: str,
    output_path: str = "loss_curves.png",
) -> None:
    """绘制 Loss 曲线并标注分叉点。"""
    diverge_step = detect_divergence_point(curve)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve.steps, curve.train_loss, label="Train Loss", color="steelblue", linewidth=2)
    ax.plot(curve.steps, curve.val_loss, label="Val Loss", color="darkorange", linewidth=2)

    if diverge_step is not None:
        ax.axvline(
            x=diverge_step,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"过拟合起点 (step={diverge_step})",
        )
        ax.annotate(
            f"⚠️ 过拟合起点\nstep={diverge_step}",
            xy=(diverge_step, min(curve.train_loss + curve.val_loss)),
            xytext=(15, 20),
            textcoords="offset points",
            color="red",
            fontsize=10,
        )
        print(f"[诊断] 检测到过拟合起点：step={diverge_step}")
        print(f"[建议] 将 max_steps 设为 {diverge_step}，或启用早停回调（patience=3）")
    else:
        print("[诊断] 未检测到明显过拟合，当前训练步数合理")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"训练过程 Loss 曲线 — {Path(checkpoint_dir).name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Loss 曲线已保存：{output_path}")
    plt.show()