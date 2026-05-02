# 最小验证示例，单文件可直接运行
# 运行前确保 .env 中有 OPENAI_API_KEY

import os
from dotenv import load_dotenv
load_dotenv()

# 验证 1：基础流式调用
from openai import OpenAI
client = OpenAI()
chunks = list(
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "1+1=？只回答数字"}],
        stream=True,
    )
)
answer = "".join(c.choices[0].delta.content or "" for c in chunks)
assert answer.strip() == "2", f"预期 '2'，得到 '{answer}'"
print(f"✅ 基础流式调用正常，回答：{answer.strip()!r}")

# 验证 2：CoT 解析器
from core import ChunkType, stream_cot_prompt
chunks = list(stream_cot_prompt("1+1=？请先写出思考过程再给出答案"))
types = {c.chunk_type for c in chunks}
assert ChunkType.THINKING in types, "未检测到 thinking 块，检查 System Prompt"
assert ChunkType.ANSWER in types, "未检测到 answer 块，检查 XML 标签解析"
think_len = sum(len(c.content) for c in chunks if c.chunk_type == ChunkType.THINKING)
answer_len = sum(len(c.content) for c in chunks if c.chunk_type == ChunkType.ANSWER)
print(f"✅ CoT 解析正常：thinking={think_len} chars，answer={answer_len} chars")

print("\n🎉 所有验证通过，可以运行 terminal_app.py 或 web_app.py")