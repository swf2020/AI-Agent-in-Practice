"""judge 模块 - 翻译质量评估核心功能"""
from .prompts import (
    build_evaluation_prompt,
    PROMPT_VERSIONS,
    DEFAULT_PROMPT_VERSION,
)
from .translator import translate_text
from .evaluator import evaluate_translation, evaluate_batch, judge_batch
from .adversarial import test_position_bias, test_consistency

__all__ = [
    "build_evaluation_prompt",
    "PROMPT_VERSIONS",
    "DEFAULT_PROMPT_VERSION",
    "translate_text",
    "evaluate_translation",
    "evaluate_batch",
    "judge_batch",
    "test_position_bias",
    "test_consistency",
]