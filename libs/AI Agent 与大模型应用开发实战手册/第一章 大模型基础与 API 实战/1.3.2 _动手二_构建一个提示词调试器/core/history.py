"""
实验历史持久化

格式选择：JSONL（JSON Lines）
  - 追加写入：每次 save_run 只 append 一行，不重写整个文件
  - 可读性强：每行独立，可用任何文本工具查看
  - 容错性好：某行损坏不影响其他行的读取
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.caller import CallResult

HISTORY_FILE = Path("history.jsonl")


def save_run(
    system_prompt: str,
    user_prompt: str,
    selected_models: list[str],
    temperature: float,
    max_tokens: int,
    results: list[CallResult],
    scores: dict[str, int] | None = None,
    notes: str = "",
) -> str:
    """
    将一次实验追加写入 history.jsonl。

    Returns:
        本次实验的唯一 ID（时间戳格式）
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "selected_models": selected_models,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "results": [
            {
                "model": r.model,
                "output": r.output,
                "latency": r.latency,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
                "estimated_cost": r.estimated_cost,
                "error": r.error,
                # 评分：若 scores 为 None 或该模型未评分，记为 -1（待评分）
                "score": (scores or {}).get(r.model, -1),
            }
            for r in results
        ],
        "notes": notes,
    }

    # 追加写入，确保文件编码为 UTF-8（处理中文 Prompt）
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return run_id


def load_history() -> pd.DataFrame:
    """
    读取 history.jsonl 并转换为 DataFrame，供 Gradio 表格展示。

    Returns:
        DataFrame，每行对应一次模型调用（非实验），按时间倒序排列
    """
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=[
            "run_id", "timestamp", "模型", "耗时(s)", "Tokens",
            "费用($)", "评分", "User Prompt 预览", "备注"
        ])

    rows = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # 跳过损坏行，不让单行错误中断整个加载
                continue

            for result in record["results"]:
                rows.append({
                    "run_id": record["run_id"],
                    "timestamp": record["timestamp"][:19].replace("T", " "),
                    "模型": result["model"],
                    "耗时(s)": result["latency"],
                    "Tokens": result["total_tokens"],
                    "费用($)": result["estimated_cost"],
                    "评分": result["score"] if result["score"] != -1 else "—",
                    "User Prompt 预览": record["params"]["user_prompt"][:40] + "...",
                    "备注": record.get("notes", ""),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def get_run_by_id(run_id: str) -> dict | None:
    """通过 run_id 查找完整实验记录，用于历史回填功能"""
    if not HISTORY_FILE.exists():
        return None
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record["run_id"] == run_id:
                    return record
            except json.JSONDecodeError:
                continue
    return None


def export_comparison_report(run_ids: list[str]) -> str:
    """
    将多条历史记录生成 Markdown 对比报告。

    Args:
        run_ids: 要对比的实验 ID 列表

    Returns:
        Markdown 格式的对比报告字符串
    """
    records = [r for rid in run_ids if (r := get_run_by_id(rid))]
    if not records:
        return "未找到指定实验记录"

    lines = ["# Prompt 实验对比报告\n"]
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"对比实验数：{len(records)}\n\n---\n")

    for rec in records:
        p = rec["params"]
        lines.append(f"## 实验 `{rec['run_id']}`\n")
        lines.append(f"**System Prompt**：{p['system_prompt'][:100]}...\n\n")
        lines.append(f"**User Prompt**：{p['user_prompt']}\n\n")
        lines.append(f"**参数**：Temperature={p['temperature']}, Max Tokens={p['max_tokens']}\n\n")

        for r in rec["results"]:
            score_str = f"⭐ {r['score']}/5" if r["score"] != -1 else "未评分"
            lines.append(f"### {r['model']} — {score_str}\n")
            if r["error"]:
                lines.append(f"> {r['error']}\n\n")
            else:
                lines.append(
                    f"⏱ {r['latency']}s | 🪙 {r['total_tokens']} tokens | "
                    f"💰 ${r['estimated_cost']}\n\n"
                )
                lines.append(f"{r['output']}\n\n")
        lines.append("---\n")

    return "\n".join(lines)