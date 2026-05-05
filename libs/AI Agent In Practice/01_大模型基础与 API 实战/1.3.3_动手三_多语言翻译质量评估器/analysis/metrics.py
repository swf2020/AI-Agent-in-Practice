"""
量化分析：相关系数计算 + 热力图生成
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def correlation_with_human(
    llm_scores: list[float],
    human_scores: list[float],
) -> dict:
    """
    计算 LLM 评分与人工评分的 Spearman 相关系数
    
    选用 Spearman 而非 Pearson 的原因：
    - 1-5 的评分是序数数据，不满足 Pearson 的区间尺度假设
    - Spearman 只关心排名顺序，对异常值更鲁棒
    """
    if len(llm_scores) != len(human_scores):
        raise ValueError("LLM 评分与人工评分数量不一致")

    coef, pvalue = spearmanr(llm_scores, human_scores)
    return {
        "spearman_r": round(float(coef), 3),
        "p_value": round(float(pvalue), 4),
        "significant": pvalue < 0.05,
        "interpretation": _interpret_correlation(coef),
    }


def _interpret_correlation(r: float) -> str:
    if r >= 0.7:
        return "强相关（裁判可信）"
    elif r >= 0.5:
        return "中等相关（裁判可接受）"
    elif r >= 0.3:
        return "弱相关（裁判需改进）"
    else:
        return "无相关（裁判不可用）"


def consistency_score(multi_run_scores: list[list[float]]) -> dict:
    """
    计算多次评分的标准差

    Args:
        multi_run_scores: shape = (runs, n_items)，每行是一次评估的所有分数
    """
    arr = np.array(multi_run_scores)  # shape: (runs, n_items)
    stds = np.std(arr, axis=0)        # 每条数据的标准差
    return {
        "mean_std": round(float(stds.mean()), 3),
        "max_std": round(float(stds.max()), 3),
        "per_item_std": stds.tolist(),
    }


def plot_heatmap(
    results_df: pd.DataFrame,
    output_path: str = "heatmap.png",
) -> None:
    """
    绘制热力图：裁判版本 × 翻译来源 × 综合评分

    Args:
        results_df: 包含 prompt_version / translator / overall 列的 DataFrame
    """
    pivot = results_df.pivot_table(
        values="overall",
        index="prompt_version",
        columns="translator",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=1,
        vmax=5,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("裁判版本 × 翻译来源 × 平均综合评分", fontsize=14, pad=15)
    ax.set_xlabel("翻译来源", fontsize=12)
    ax.set_ylabel("裁判 Prompt 版本", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"热力图已保存至 {output_path}")


def generate_summary_report(
    results_df: pd.DataFrame,
    human_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    生成汇总报告：按裁判版本计算相关系数和一致性指标

    Returns:
        DataFrame，每行是一个裁判版本的汇总指标
    """
    report_rows = []

    for version in results_df["prompt_version"].unique():
        subset = results_df[results_df["prompt_version"] == version]

        # 对齐人工评分
        merged = subset.merge(human_scores_df, on="item_id", suffixes=("_llm", "_human"))
        if merged.empty:
            continue

        corr = correlation_with_human(
            merged["overall_llm"].tolist(),
            merged["overall_human"].tolist(),
        )

        # 一致性：取该版本所有 item 在多次 run 中的标准差
        stds = (
            subset.groupby("item_id")["overall"]
            .std(ddof=0)
            .fillna(0)
        )

        report_rows.append({
            "prompt_version": version,
            "spearman_r": corr["spearman_r"],
            "p_value": corr["p_value"],
            "significant": corr["significant"],
            "mean_std": round(stds.mean(), 3),
            "max_std": round(stds.max(), 3),
            "interpretation": corr["interpretation"],
        })

    return pd.DataFrame(report_rows).sort_values("spearman_r", ascending=False)