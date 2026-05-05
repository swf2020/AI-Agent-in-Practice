"""
Prompt 调试器主程序

运行：python app.py
然后打开浏览器访问 http://localhost:7860
支持 DeepSeek 和 Qwen 模型。
"""
import asyncio
import json
from datetime import datetime

import gradio as gr
import pandas as pd

import core  # 触发 dotenv 加载
from core.caller import call_all, CallResult, MODEL_REGISTRY
from core.history import (
    export_comparison_report,
    get_run_by_id,
    load_history,
    save_run,
)

ALL_MODELS = list(MODEL_REGISTRY.keys())


def format_result_markdown(r: CallResult) -> str:
    """将 CallResult 格式化为 Markdown，供 Gradio Markdown 组件展示"""
    if r.error:
        return f"## ❌ {r.model}\n\n{r.error}"

    return (
        f"## ✅ {r.model}\n\n"
        f"⏱ **{r.latency}s** | "
        f"🪙 **{r.total_tokens}** tokens "
        f"({r.input_tokens} in / {r.output_tokens} out) | "
        f"💰 **${r.estimated_cost}**\n\n"
        f"---\n\n"
        f"{r.output}"
    )


def run_experiment(
    system_prompt: str,
    user_prompt: str,
    selected_models: list[str],
    temperature: float,
    max_tokens: int,
) -> tuple:
    """
    Gradio 事件处理函数：触发并发调用，返回各模型结果。

    Returns:
        2 个 Markdown 内容 + 状态信息，对应界面上 2 个输出列
    """
    if not selected_models:
        return ("⚠️ 请至少选择一个模型", "", "未选择模型")
    if not user_prompt.strip():
        return ("⚠️ User Prompt 不能为空", "", "Prompt 为空")

    # 在同步函数中运行异步代码
    # 注意：Gradio 5.x 在内部线程中运行事件处理器，asyncio.run() 是安全的
    results: list[CallResult] = asyncio.run(
        call_all(selected_models, system_prompt, user_prompt, temperature, int(max_tokens))
    )

    # 将结果映射到固定的 2 个输出槽
    model_to_result = {r.model: r for r in results}
    outputs = []
    for model in ALL_MODELS:  # 固定顺序：DeepSeek-V3, Qwen-Max
        if model in model_to_result:
            outputs.append(format_result_markdown(model_to_result[model]))
        else:
            outputs.append("")  # 未选择则留空

    # 保存历史（评分在 UI 中单独触发，此处先存 -1）
    run_id = save_run(
        system_prompt, user_prompt, selected_models,
        temperature, int(max_tokens), results
    )

    status = (
        f"✅ 实验完成 [{run_id}] — "
        f"共 {len(results)} 个模型，"
        f"总费用约 ${sum(r.estimated_cost for r in results):.6f}"
    )

    return tuple(outputs) + (status,)


def save_scores_and_notes(
    run_id_input: str,
    score_deepseek: int,
    score_qwen: int,
    notes: str,
) -> str:
    """将用户评分写回 history.jsonl（通过重写对应行实现）"""
    record = {
        "type": "score_update",
        "target_run_id": run_id_input.strip(),
        "timestamp": datetime.utcnow().isoformat(),
        "scores": {
            "DeepSeek-V3": score_deepseek,
            "Qwen-Max": score_qwen,
        },
        "notes": notes,
    }
    with open("history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return f"✅ 评分已保存到实验 {run_id_input.strip()}"


def refresh_history() -> pd.DataFrame:
    """刷新历史记录表格（强制重新读取文件）"""
    return load_history(use_cache=False)


def fill_from_history(evt: gr.SelectData, df: pd.DataFrame):
    """
    点击历史记录表格某行时，回填 Prompt 和参数到输入区。

    Gradio SelectData 包含 index（行号）和 value（单元格值）。
    我们通过行号找到 run_id，再从文件读取完整记录。
    """
    if evt.index is None or df.empty:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    row_idx = evt.index[0]
    if row_idx >= len(df):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    run_id = df.iloc[row_idx]["run_id"]
    record = get_run_by_id(run_id)
    if not record:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    p = record["params"]
    return (
        gr.update(value=p["system_prompt"]),
        gr.update(value=p["user_prompt"]),
        gr.update(value=p["selected_models"]),
        gr.update(value=p["temperature"]),
        gr.update(value=p["max_tokens"]),
    )


# ─────────────── Gradio Blocks UI ───────────────
with gr.Blocks(
    title="🔬 Prompt 调试器",
) as demo:
    gr.Markdown("# 🔬 Prompt 调试器\n> 改变一个变量，观察输出变化，记录结论\n\n**支持的模型：DeepSeek-V3, Qwen-Max**")

    # ── 输入区 ──
    with gr.Row():
        with gr.Column(scale=2):
            system_box = gr.Textbox(
                label="System Prompt",
                placeholder="你是一个专业的代码审查员...",
                lines=4,
                value="You are a helpful assistant. Be concise and precise.",
            )
        with gr.Column(scale=2):
            user_box = gr.Textbox(
                label="User Prompt",
                placeholder="在这里输入你的问题或指令...",
                lines=4,
            )

    # ── 参数控制区 ──
    with gr.Row():
        with gr.Column(scale=2):
            model_check = gr.CheckboxGroup(
                choices=ALL_MODELS,
                value=["DeepSeek-V3"],
                label="选择模型（可多选，并发调用）",
            )
        with gr.Column(scale=1):
            temp_slider = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                label="Temperature（越高越随机）",
            )
        with gr.Column(scale=1):
            token_slider = gr.Slider(
                minimum=100, maximum=4000, value=1000, step=100,
                label="Max Tokens（输出上限）",
            )

    run_btn = gr.Button("🚀 运行实验", variant="primary", size="lg")
    status_box = gr.Textbox(label="实验状态", interactive=False)

    # ── 输出区（2 列固定对应 2 个模型）──
    gr.Markdown("## 📊 模型输出对比")
    with gr.Row(equal_height=False):
        out_deepseek = gr.Markdown(elem_classes=["output-col"])
        out_qwen = gr.Markdown(elem_classes=["output-col"])

    # ── 评分区 ──
    with gr.Accordion("📝 手动评分（可选）", open=False):
        gr.Markdown("打分后点击保存，评分会关联到本次实验 run_id")
        run_id_input = gr.Textbox(
            label="Run ID（从实验状态栏复制）", placeholder="20241201_143022_123456"
        )
        with gr.Row():
            score_deepseek = gr.Slider(1, 5, value=3, step=1, label="DeepSeek-V3 评分")
            score_qwen = gr.Slider(1, 5, value=3, step=1, label="Qwen-Max 评分")
        notes_box = gr.Textbox(label="备注", placeholder="DeepSeek 格式更规范，但少了一个边界条件...")
        save_score_btn = gr.Button("💾 保存评分")
        score_status = gr.Textbox(label="保存状态", interactive=False)

    # ── 历史记录区 ──
    with gr.Accordion("📚 历史记录", open=False):
        refresh_btn = gr.Button("🔄 刷新历史")
        history_df = gr.DataFrame(
            value=load_history(),
            label="实验历史（点击行可回填参数）",
            interactive=False,
            wrap=True,
        )
        gr.Markdown("*点击表格中任意行，Prompt 和参数会自动回填到输入区*")

    # ── 报告导出区 ──
    with gr.Accordion("📄 导出对比报告", open=False):
        export_ids = gr.Textbox(
            label="输入 Run ID（多个用逗号分隔）",
            placeholder="20241201_143022_123456, 20241201_150033_654321",
        )
        export_btn = gr.Button("📥 生成 Markdown 报告")
        report_output = gr.Markdown()

    # ─── 事件绑定 ───
    run_btn.click(
        fn=run_experiment,
        inputs=[system_box, user_box, model_check, temp_slider, token_slider],
        outputs=[out_deepseek, out_qwen, status_box],
    )

    save_score_btn.click(
        fn=save_scores_and_notes,
        inputs=[run_id_input, score_deepseek, score_qwen, notes_box],
        outputs=[score_status],
    )

    refresh_btn.click(fn=refresh_history, outputs=[history_df])

    history_df.select(
        fn=fill_from_history,
        inputs=[history_df],
        outputs=[system_box, user_box, model_check, temp_slider, token_slider],
    )

    export_btn.click(
        fn=lambda ids: export_comparison_report(
            [x.strip() for x in ids.split(",") if x.strip()]
        ),
        inputs=[export_ids],
        outputs=[report_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        theme=gr.themes.Soft(),
        css=".output-col { min-height: 300px; }",
        share=False,
        show_error=True,
    )
