# 最小验证示例，单文件可直接运行
# 支持 DeepSeek、Qwen 或 OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

from core import get_openai_client, get_default_model

client = get_openai_client()
model = get_default_model()

chunks = list(
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "1+1=？只回答数字"}],
        stream=True,
    )
)
answer = "".join(c.choices[0].delta.content or "" for c in chunks)
assert answer.strip() == "2", f"预期 '2'，得到 '{answer}'"
print(f"✅ 基础流式调用正常，回答：{answer.strip()!r}")

from core import ChunkType, stream_cot_prompt
chunks = list(stream_cot_prompt("1+1=？请先写出思考过程再给出答案"))
types = {c.chunk_type for c in chunks}
assert ChunkType.THINKING in types, "未检测到 thinking 块，检查 System Prompt"
assert ChunkType.ANSWER in types, "未检测到 answer 块，检查 XML 标签解析"
think_len = sum(len(c.content) for c in chunks if c.chunk_type == ChunkType.THINKING)
answer_len = sum(len(c.content) for c in chunks if c.chunk_type == ChunkType.ANSWER)
print(f"✅ CoT 解析正常：thinking={think_len} chars，answer={answer_len} chars")

print("\n🎉 所有验证通过，可以运行 terminal_app.py 或 web_app.py")
