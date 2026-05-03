"""judge 模块 - 翻译质量评估核心功能"""
from .prompts import (
    PROMPT_VERSIONS,
)
from .translator import translate_with_llm, generate_all_translations, translate_with_google
from .evaluator import judge_single, judge_batch
from .adversarial import test_position_bias, test_consistency

# 向后兼容的别名
evaluate_translation = judge_single
evaluate_batch = judge_batch

__all__ = [
    "PROMPT_VERSIONS",
    "translate_with_llm",
    "generate_all_translations",
    "translate_with_google",
    "judge_single",
    "judge_batch",
    "evaluate_translation",
    "evaluate_batch",
    "test_position_bias",
    "test_consistency",
]