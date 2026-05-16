#!/usr/bin/env python3
"""CoT 流式输出快速演示（仅打印，不做断言）。
如需完整验证（含断言），请运行 smoke_test.py。"""

from core import stream_cot_prompt, ChunkType

print('🔍 测试 CoT 流式输出...')
prompt = '一个篮子里有5个苹果，你拿走了3个，你有几个苹果？'

thinking = []
answer = []

for chunk in stream_cot_prompt(prompt):
    if chunk.chunk_type == ChunkType.THINKING:
        thinking.append(chunk.content)
    else:
        answer.append(chunk.content)
    print(chunk.content, end='', flush=True)

print()
print('\n✅ 测试完成！')
print(f'思考内容长度: {len("".join(thinking))} 字符')
print(f'回答内容长度: {len("".join(answer))} 字符')
