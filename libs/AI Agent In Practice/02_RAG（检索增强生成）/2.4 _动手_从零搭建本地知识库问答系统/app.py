"""
Chainlit Web 界面：启动命令 → chainlit run app.py
"""
from __future__ import annotations

import chainlit as cl

from step4_query import RAGPipeline

# 全局单例，避免每次对话重新加载模型（模型加载耗时约 2-5s）
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(top_k=5, score_threshold=0.4)
    return _pipeline


@cl.on_chat_start
async def on_start() -> None:
    """对话开始时的欢迎信息"""
    await cl.Message(
        content="👋 你好！我是本地知识库问答助手。\n"
                "请先确保已运行 `python step3_index.py` 完成文档索引。\n"
                "有什么想了解的？"
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """处理用户消息"""
    # [Fix #8] 添加异常处理，避免 Chainlit 界面直接报错
    try:
        pipeline = get_pipeline()
        question = message.content.strip()

        # 显示思考中状态
        async with cl.Step(name="🔍 检索知识库") as step:
            result = pipeline.ask(question)
            step.output = f"找到 {len(result.sources)} 条相关内容"

        # 构建引用来源的展示文本
        source_text = ""
        if result.sources:
            source_lines = [
                f"- [{c.score:.2f}] `{c.source.split('/')[-1]}`"
                for c in result.sources
            ]
            source_text = "\n\n**参考来源：**\n" + "\n".join(source_lines)

        await cl.Message(content=result.answer + source_text).send()
    except Exception as e:
        error_msg = f"抱歉，处理您的问题时出现错误：{type(e).__name__}: {e}"
        await cl.Message(content=error_msg).send()