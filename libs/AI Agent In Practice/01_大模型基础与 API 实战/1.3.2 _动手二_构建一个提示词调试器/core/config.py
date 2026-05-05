"""
向后兼容模块 — 实际配置已迁移到项目根目录的 core_config.py。
此模块重新导出所有符号，确保 core/caller.py 等原有导入无需修改。
"""
from core_config import (
    MODEL_REGISTRY,
    ACTIVE_MODEL_KEY,
    estimate_cost,
    get_litellm_id,
    get_api_key,
    get_base_url,
    get_model_list,
    get_active_config,
)

__all__ = [
    "MODEL_REGISTRY",
    "ACTIVE_MODEL_KEY",
    "estimate_cost",
    "get_litellm_id",
    "get_api_key",
    "get_base_url",
    "get_model_list",
    "get_active_config",
]
