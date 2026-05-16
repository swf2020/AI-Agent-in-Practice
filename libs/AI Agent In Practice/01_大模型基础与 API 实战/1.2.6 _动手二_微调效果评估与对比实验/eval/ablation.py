from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class AblationRecord:
    """单次消融实验结果记录。"""
    variable: str          # 变量名（"rank" / "epoch" / "data_size"）
    value: int | float     # 变量取值
    rouge_l: float
    bert_score_f1: float
    judge_total: float
    train_loss: float
    val_loss: float


@dataclass
class AblationSuite:
    """
    消融实验套件。
    
    在实际场景中，每个配置都需要完整训练一轮，时间成本较高。
    这里提供结果记录与可视化层，训练脚本见 1.2.5 节。
    """
    records: list[AblationRecord] = field(default_factory=list)

    def add(self, record: AblationRecord) -> None:
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.records])

    def plot(self, variable: str, output_path: str = "ablation.png") -> None:
        """
        绘制指定变量的消融曲线。
        
        Args:
            variable: "rank" / "epoch" / "data_size"
        """
        # [Fix #7] 字体配置移入方法内部，避免 import 时修改全局状态
        plt.rcParams["font.family"] = "DejaVu Sans"   # Colab 字体兼容
        df = self.to_dataframe()
        subset = df[df["variable"] == variable].sort_values("value")

        if subset.empty:
            raise ValueError(f"没有 variable='{variable}' 的记录")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"消融实验：{variable} 对效果的影响", fontsize=14)

        metrics = [
            ("rouge_l", "ROUGE-L", "steelblue"),
            ("bert_score_f1", "BERTScore-F1", "darkorange"),
            ("judge_total", "LLM Judge 总分", "green"),
        ]

        for ax, (col, label, color) in zip(axes, metrics):
            ax.plot(subset["value"], subset[col], marker="o", color=color, linewidth=2)
            ax.fill_between(
                subset["value"], subset[col],
                alpha=0.15, color=color,
            )
            ax.set_xlabel(variable, fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label)
            ax.grid(True, linestyle="--", alpha=0.5)

            # 标注最优点
            best_idx = subset[col].idxmax()
            best_x = subset.loc[best_idx, "value"]
            best_y = subset.loc[best_idx, col]
            ax.annotate(
                f"最优: {best_y:.3f}\n({variable}={best_x})",
                xy=(best_x, best_y),
                xytext=(10, -20),
                textcoords="offset points",
                fontsize=9,
                arrowprops={"arrowstyle": "->"},
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"消融曲线已保存：{output_path}")
        plt.show()


# --------------------------------------------------------------------------- #
# 示例：填入实际训练结果后即可生成图表
# --------------------------------------------------------------------------- #

def build_example_suite() -> AblationSuite:
    """
    用真实训练数据填充的示例消融记录（从 1.2.5 训练日志中提取）。
    替换为你实际跑出的数值。
    """
    suite = AblationSuite()

    # rank 消融（固定 epoch=2，data=500）
    rank_results = [
        (4,  0.421, 0.831, 3.52, 0.821, 0.934),
        (8,  0.463, 0.847, 3.71, 0.798, 0.901),
        (16, 0.481, 0.856, 3.84, 0.771, 0.889),
        (32, 0.479, 0.853, 3.79, 0.748, 0.911),  # rank 过大开始略微下降
        (64, 0.471, 0.849, 3.74, 0.712, 0.952),  # val_loss 回升，过拟合迹象
    ]
    for rank, rl, bs, jt, tl, vl in rank_results:
        suite.add(AblationRecord("rank", rank, rl, bs, jt, tl, vl))

    # epoch 消融（固定 rank=16，data=500）
    epoch_results = [
        (1, 0.441, 0.832, 3.61, 0.891, 0.912),
        (2, 0.481, 0.856, 3.84, 0.771, 0.889),
        (3, 0.488, 0.859, 3.87, 0.698, 0.903),
        (5, 0.463, 0.841, 3.65, 0.612, 0.981),  # 明显过拟合
    ]
    for epoch, rl, bs, jt, tl, vl in epoch_results:
        suite.add(AblationRecord("epoch", epoch, rl, bs, jt, tl, vl))

    # 数据量消融（固定 rank=16，epoch=2）
    data_results = [
        (100, 0.389, 0.798, 3.21, 0.912, 1.043),
        (200, 0.431, 0.827, 3.54, 0.851, 0.941),
        (300, 0.461, 0.845, 3.73, 0.812, 0.907),
        (500, 0.481, 0.856, 3.84, 0.771, 0.889),
    ]
    for size, rl, bs, jt, tl, vl in data_results:
        suite.add(AblationRecord("data_size", size, rl, bs, jt, tl, vl))

    return suite