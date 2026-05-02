"""
调用三个翻译来源生成中文译文：GPT-4o / DeepSeek-V3 / Google Translate
"""
import asyncio
import os
from dotenv import load_dotenv
from litellm import acompletion
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

TRANSLATE_PROMPT = """请将以下英文句子翻译成中文。
要求：
- 直接输出译文，不要任何解释或前缀
- 保持原文的语气和风格
- 专业术语请使用中文领域标准译法

原文：{source}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def translate_with_llm(
    source: str,
    model: str,
) -> str:
    """调用 LLM 翻译单条文本，失败自动重试3次"""
    resp = await acompletion(
        model=model,
        messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(source=source)}],
        temperature=0.3,  # 翻译任务用低 temperature 保持稳定，但不能为0（否则过于死板）
    )
    return resp.choices[0].message.content.strip()


async def generate_all_translations(
    test_set: list,
    models: dict[str, str],
) -> dict[str, dict[str, str]]:
    """
    并发生成所有翻译
    
    Returns:
        {item_id: {"gpt4o": "译文", "deepseek": "译文", "google": "译文"}}
    """
    results: dict[str, dict[str, str]] = {item.id: {} for item in test_set}

    for model_name, model_id in models.items():
        print(f"正在用 {model_name} 翻译 {len(test_set)} 条...")
        tasks = [translate_with_llm(item.source, model_id) for item in test_set]
        translations = await asyncio.gather(*tasks, return_exceptions=True)

        for item, translation in zip(test_set, translations):
            if isinstance(translation, Exception):
                print(f"  ⚠️  {model_name} 翻译 {item.id} 失败：{translation}")
                results[item.id][model_name] = "[翻译失败]"
            else:
                results[item.id][model_name] = translation

    return results


# 注：Google Translate 需要单独的 SDK，这里用 deep-translator 库
# uv pip install deep-translator==1.11.4
async def translate_with_google(source: str) -> str:
    """Google Translate 免费版（无需 API Key）"""
    from deep_translator import GoogleTranslator
    # 同步调用包在 executor 里，避免阻塞事件循环
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: GoogleTranslator(source="en", target="zh-CN").translate(source),
    )
    return result