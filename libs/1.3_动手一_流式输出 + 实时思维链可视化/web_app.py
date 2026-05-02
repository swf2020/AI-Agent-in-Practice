"""
Streamlit 网页版思维链可视化。
运行方式：streamlit run web_app.py
Colab 运行方式见文件末尾说明。
"""

import time
import threading
from typing import Generator

import streamlit as st

from core import ChunkType, stream_cot_prompt, stream_extended_thinking

# ── 页面配置 ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="实时思维链可视化",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 实时思维链可视化")
st.caption("让模型「思考」变得可见 · Streaming + CoT 实战")

# ── 侧边栏配置 ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 配置")

    mode = st.radio(
        "推理模式",
        options=["CoT Prompt（OpenAI）", "Extended Thinking（Claude）"],
        help=(
            "CoT Prompt：通过 System Prompt 引导模型输出带标签的推理过程\n\n"
            "Extended Thinking：Claude 原生推理模式，思考内容不可被 Prompt 干预"
        ),
    )

    if mode == "CoT Prompt（OpenAI）":
        model = st.selectbox(
            "模型",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            index=0,
        )
        budget_tokens = None
    else:
        model = "claude-3-7-sonnet-20250219"
        budget_tokens = st.slider(
            "Thinking Budget（tokens）",
            min_value=1000,
            max_value=10000,
            value=4000,
            step=1000,
            help="分配给模型思考过程的最大 token 数。越大推理越充分，但延迟和成本同步增加。",
        )

    st.divider()
    st.subheader("📊 实时指标")
    metric_ttft = st.empty()
    metric_tps = st.empty()
    metric_tokens = st.empty()

# ── 主区域 ───────────────────────────────────────────────────────────
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

# ── 推理执行 ─────────────────────────────────────────────────────────
if start_btn and prompt.strip():
    # 创建双栏布局
    col_think, col_answer = st.columns(2, gap="medium")

    with col_think:
        st.subheader("🧠 思考过程")
        # 用 code 块展示思考内容，更容易区分"草稿"和"答案"的视觉感
        think_placeholder = st.empty()

    with col_answer:
        st.subheader("✅ 最终回答")
        answer_placeholder = st.empty()

    # 初始化统计变量
    think_text = ""
    answer_text = ""
    token_count = 0
    start_time = time.perf_counter()
    ttft: float | None = None

    # 选择流生成器
    if mode == "CoT Prompt（OpenAI）":
        stream_gen = stream_cot_prompt(prompt, model=model)
    else:
        stream_gen = stream_extended_thinking(prompt, budget_tokens=budget_tokens)

    # 消费流
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
            # 用 code 块展示思考，视觉上像"草稿纸"
            think_placeholder.code(think_text, language=None)
        else:
            answer_text += chunk.content
            answer_placeholder.markdown(answer_text)

        # 更新侧边栏指标（每 10 个 token 更新一次，避免频繁 re-render）
        if token_count % 10 == 0:
            metric_tps.metric("📈 Token/s", f"{tps:.1f}")
            metric_tokens.metric("🔢 Token 数", token_count)

    # 最终更新指标
    total_time = time.perf_counter() - start_time
    metric_tps.metric("📈 Token/s", f"{token_count / total_time:.1f}")
    metric_tokens.metric("🔢 Token 数", token_count)

    st.success(f"✅ 推理完成 · 总耗时 {total_time:.2f}s · {token_count} tokens")

elif start_btn and not prompt.strip():
    st.warning("请先输入问题")

# ── Colab 运行说明 ────────────────────────────────────────────────────
"""
在 Colab 中运行 Streamlit 的方式：

!pip install streamlit pyngrok -q

# 将 web_app.py 保存到文件
with open('web_app.py', 'w') as f:
    f.write(WEB_APP_CODE)  # 把本文件内容写入

import subprocess, threading
from pyngrok import ngrok

# 启动 Streamlit 进程
proc = subprocess.Popen(['streamlit', 'run', 'web_app.py', '--server.port=8501'])

# 建立 ngrok 隧道
public_url = ngrok.connect(8501)
print(f"访问地址：{public_url}")
"""