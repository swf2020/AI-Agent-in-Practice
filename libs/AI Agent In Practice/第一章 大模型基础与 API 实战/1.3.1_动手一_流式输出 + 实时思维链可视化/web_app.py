"""
Streamlit 网页版思维链可视化。
运行方式：streamlit run web_app.py
支持 DeepSeek、Qwen 或 OpenAI 模型，包括 DeepSeek 推理模型。
"""

import time

import streamlit as st

from core import (
    ChunkType,
    get_default_model,
    stream_cot_prompt,
    stream_extended_thinking,
)

st.set_page_config(
    page_title="实时思维链可视化",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 实时思维链可视化")
st.caption("让模型「思考」变得可见 · Streaming + CoT 实战")

with st.sidebar:
    st.header("⚙️ 配置")

    mode = st.radio(
        "推理模式",
        options=["CoT Prompt（通用）", "DeepSeek 推理模型"],
        help=(
            "CoT Prompt：通过 System Prompt 引导模型输出带标签的推理过程\n\n"
            "DeepSeek 推理模型：使用 DeepSeek 原生推理能力，思考过程不可被 Prompt 干预"
        ),
    )

    if mode == "CoT Prompt（通用）":
        model_options = {
            "DeepSeek Chat": "deepseek-chat",
            "DeepSeek Reasoner (推理更强)": "deepseek-reasoner",
            "Qwen (通义千问)": "qwen-plus",
            "GPT-4o": "gpt-4o",
        }
        selected_model = st.selectbox(
            "模型",
            options=list(model_options.keys()),
            index=0,
        )
        model = model_options[selected_model]
        budget_tokens = None
    else:
        model = "deepseek-reasoner"
        budget_tokens = st.slider(
            "思考预算（tokens）",
            min_value=2000,
            max_value=64000,
            value=8000,
            step=1000,
            help="分配给模型思考过程的最大 token 数。越大推理越充分，但延迟和成本同步增加。",
        )

    st.divider()
    st.subheader("📊 实时指标")
    metric_ttft = st.empty()
    metric_tps = st.empty()
    metric_tokens = st.empty()

EXAMPLE_PROMPTS = {
    "🧮 数学推理": "小明有72块糖，要平均分给9个朋友。后来又来了3个朋友，重新分配后每人能分到几块？",
    "🔍 逻辑题": "一个盒子里有红球和蓝球共20个。红球比蓝球多4个。红球和蓝球各有几个？",
    "🚂 应用题": "一列火车从A城出发，以90km/h速度行驶。同时另一列从B城以60km/h相向而行。AB距450km，几小时后相遇？",
    "📝 自定义": "",
}

selected = st.selectbox("选择示例题目", list(EXAMPLE_PROMPTS.keys()))
default_prompt = EXAMPLE_PROMPTS[selected]

prompt = st.text_area(
    "输入你的问题",
    value=default_prompt,
    height=100,
    placeholder="输入任意推理题目...",
)

col_btn, col_stop = st.columns([1, 5])
with col_btn:
    start_btn = st.button("▶ 开始推理", type="primary", use_container_width=True)

if start_btn and prompt.strip():
    col_think, col_answer = st.columns(2, gap="medium")

    with col_think:
        st.subheader("🧠 思考过程")
        think_placeholder = st.empty()

    with col_answer:
        st.subheader("✅ 最终回答")
        answer_placeholder = st.empty()

    think_text = ""
    answer_text = ""
    token_count = 0
    start_time = time.perf_counter()
    ttft: float | None = None

    if mode == "CoT Prompt（通用）":
        stream_gen = stream_cot_prompt(prompt, model=model)
    else:
        stream_gen = stream_extended_thinking(prompt, budget_tokens=budget_tokens)

    for chunk in stream_gen:
        now = time.perf_counter()

        if ttft is None:
            ttft = now - start_time
            metric_ttft.metric("⚡ TTFT", f"{ttft:.2f}s")

        token_count += 1
        elapsed = now - start_time
        tps = token_count / elapsed if elapsed > 0 else 0

        if chunk.chunk_type == ChunkType.THINKING:
            think_text += chunk.content
            think_placeholder.code(think_text, language=None)
        else:
            answer_text += chunk.content
            answer_placeholder.markdown(answer_text)

        if token_count % 10 == 0:
            metric_tps.metric("📈 Token/s", f"{tps:.1f}")
            metric_tokens.metric("🔢 Token 数", token_count)

    total_time = time.perf_counter() - start_time
    metric_tps.metric("📈 Token/s", f"{token_count / total_time:.1f}")
    metric_tokens.metric("🔢 Token 数", token_count)

    st.success(f"✅ 推理完成 · 总耗时 {total_time:.2f}s · {token_count} tokens")

elif start_btn and not prompt.strip():
    st.warning("请先输入问题")
