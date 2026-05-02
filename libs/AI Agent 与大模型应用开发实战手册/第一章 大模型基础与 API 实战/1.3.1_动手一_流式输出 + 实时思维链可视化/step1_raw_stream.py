# 演示原始 chunk 结构，独立可运行

import time
from core import get_openai_client, get_default_model

client = get_openai_client()
model = get_default_model()


def inspect_stream(prompt: str) -> None:
    """
    打印每个 chunk 的原始结构，用于理解 SDK 返回格式。
    生产代码不需要这么做，这里纯粹是学习用途。
    """
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    start_time = time.perf_counter()
    first_token_time: float | None = None
    token_count = 0
    full_text = ""

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.6,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content

        if delta is None:
            continue

        now = time.perf_counter()

        if first_token_time is None:
            first_token_time = now
            ttft = first_token_time - start_time
            print(f"\n⚡ TTFT: {ttft:.3f}s\n")

        token_count += 1
        full_text += delta
        print(delta, end="", flush=True)

    total_time = time.perf_counter() - start_time
    generation_time = total_time - (first_token_time - start_time) if first_token_time else total_time

    print(f"\n\n{'='*60}")
    print(f"📊 统计")
    print(f"   总耗时:     {total_time:.2f}s")
    print(f"   TTFT:       {ttft:.3f}s")
    print(f"   生成耗时:   {generation_time:.2f}s")
    print(f"   估算 tokens:{token_count}")
    print(f"   Token/s:    {token_count / generation_time:.1f}")


if __name__ == "__main__":
    inspect_stream("1+1等于几？用一句话回答。")
