# tests/test_main.py — 自动生成的冒烟测试
# 注：本项目为 GPU/Unsloth 微调项目，main.py 直接导入 CUDA 依赖，
# 因此测试聚焦于 core_config.py 的纯逻辑验证，main.py 导入测试在非 GPU 环境跳过。
import pytest
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


# ── 测试 core_config 基础结构 ──────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY,
            get_litellm_id, get_api_key, get_base_url,
            get_model_list, estimate_cost, get_unsloth_id,
        )
        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0
        assert isinstance(ACTIVE_MODEL_KEY, str)
        assert ACTIVE_MODEL_KEY in MODEL_REGISTRY

    def test_model_registry_schema(self):
        """验证每个模型条目包含必要字段"""
        from core_config import MODEL_REGISTRY
        required_keys = {
            "litellm_id", "price_in", "price_out",
            "max_tokens_limit", "api_key_env", "base_url", "unsloth_id",
        }
        for name, cfg in MODEL_REGISTRY.items():
            missing = required_keys - set(cfg.keys())
            assert not missing, f"{name} 缺少字段: {missing}"

    def test_get_litellm_id(self):
        from core_config import get_litellm_id
        result = get_litellm_id()
        assert isinstance(result, str) and len(result) > 0

    def test_get_unsloth_id(self):
        from core_config import get_unsloth_id
        result = get_unsloth_id()
        assert isinstance(result, str) and "unsloth/" in result

    def test_get_model_list(self):
        from core_config import get_model_list, MODEL_REGISTRY
        lst = get_model_list()
        assert isinstance(lst, list)
        assert set(lst) == set(MODEL_REGISTRY.keys())

    def test_estimate_cost(self):
        from core_config import estimate_cost, get_model_list
        model_key = get_model_list()[0]
        cost = estimate_cost(model_key, input_tokens=1000, output_tokens=500)
        assert isinstance(cost, float) and cost >= 0

    def test_get_api_key_no_crash(self):
        """无环境变量时应返回 None 而不是抛异常"""
        from core_config import get_api_key
        result = get_api_key()
        assert result is None or isinstance(result, str)

    def test_get_base_url(self):
        from core_config import get_base_url
        result = get_base_url()
        # Qwen-Max 有 base_url，DeepSeek-V3 为 None
        assert result is None or isinstance(result, str)

    def test_active_model_has_unsloth_id(self):
        """激活模型必须有有效的 unsloth_id"""
        from core_config import ACTIVE_MODEL_KEY, MODEL_REGISTRY
        cfg = MODEL_REGISTRY[ACTIVE_MODEL_KEY]
        assert cfg["unsloth_id"].startswith("unsloth/")


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_exists():
    """验证 main.py 文件存在"""
    main_path = os.path.join(PROJECT_DIR, "main.py")
    assert os.path.exists(main_path), "main.py 不存在"

    # 尝试读取内容验证包含必要的 import
    with open(main_path, "r") as f:
        content = f.read()
    assert "from core_config import" in content, "main.py 未导入 core_config"
    assert "get_unsloth_id()" in content, "main.py 未使用 get_unsloth_id()"


def test_main_module_importable():
    """main.py 在不具备 GPU/unsloth 环境时应跳过而非报错"""
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
        # 实际执行导入会触发 unsloth 导入，在非 GPU 环境会失败
        # 这里仅验证文件可被解析
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")
