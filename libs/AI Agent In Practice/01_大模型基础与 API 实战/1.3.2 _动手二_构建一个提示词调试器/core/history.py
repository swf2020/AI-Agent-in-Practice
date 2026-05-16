"""
实验历史持久化

格式选择：JSONL（JSON Lines）
  - 追加写入：每次 save_run 只 append 一行，不重写整个文件
  - 可读性强：每行独立，可用任何文本工具查看
  - 容错性好：某行损坏不影响其他行的读取
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.caller import CallResult

HISTORY_FILE = Path("history.jsonl")

_history_cache = None
_cache_timestamp = None


def _invalidate_cache():
    global _history_cache, _cache_timestamp
    _history_cache = None
    _cache_timestamp = None


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
                "score": (scores or {}).get(r.model, -1),
            }
            for r in results
        ],
        "notes": notes,
    }

    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _invalidate_cache()

    return run_id


def load_history(use_cache: bool = True) -> pd.DataFrame:
    """
    读取 history.jsonl 并转换为 DataFrame，供 Gradio 表格展示。
    使用内存缓存避免重复读取文件。

    Args:
        use_cache: 是否使用缓存，False 时强制重新读取文件

    Returns:
        DataFrame，每行对应一次模型调用（非实验），按时间倒序排列
    """
    global _history_cache, _cache_timestamp

    columns = [
        "run_id", "timestamp", "模型", "耗时(s)", "Tokens",
        "费用($)", "评分", "User Prompt 预览", "备注"
    ]

    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=columns)

    current_mtime = HISTORY_FILE.stat().st_mtime if HISTORY_FILE.exists() else 0

    if use_cache and _history_cache is not None and _cache_timestamp == current_mtime:
        return _history_cache

    # [Fix #4] 先收集 score_update 记录，后续合并到实验行
    score_map: dict[tuple[str, str], int] = {}
    rows: list[dict] = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 收集评分更新记录（先跳过，稍后合并）
            if record.get("type") == "score_update":
                rid = record.get("target_run_id", "")
                for model, score in record.get("scores", {}).items():
                    score_map[(rid, model)] = score
                continue

            if "results" not in record:
                continue

            for result in record["results"]:
                rows.append({
                    "run_id": record["run_id"],
                    "timestamp": record["timestamp"][:19].replace("T", " "),
                    "模型": result["model"],
                    "耗时(s)": result["latency"],
                    "Tokens": result["total_tokens"],
                    "费用($)": result["estimated_cost"],
                    # [Fix #4] 优先使用 score_update 中的评分，否则使用原始评分
                    _score = score_map.get((record["run_id"], result["model"]), result["score"])
                    "评分": _score if _score != -1 else "—",
                    "User Prompt 预览": record["params"]["user_prompt"][:40] + "...",
                    "备注": record.get("notes", ""),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=columns)

    _history_cache = df
    _cache_timestamp = current_mtime

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
