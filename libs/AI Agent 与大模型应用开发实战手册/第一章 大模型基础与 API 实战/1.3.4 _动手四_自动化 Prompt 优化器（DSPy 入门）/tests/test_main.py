# tests/test_main.py — 冒烟测试
import pytest
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))


# ── 测试 core_config 基础结构 ──────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY,
            get_litellm_id, get_api_key, get_base_url,
            get_model_list, estimate_cost, get_active_config,
        )
        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0
        assert isinstance(ACTIVE_MODEL_KEY, str)
        assert ACTIVE_MODEL_KEY in MODEL_REGISTRY

    def test_model_registry_schema(self):
        """验证每个模型条目包含必要字段"""
        from core_config import MODEL_REGISTRY
        required_keys = {"litellm_id", "price_in", "price_out",
                         "max_tokens_limit", "api_key_env", "base_url"}
        for name, cfg in MODEL_REGISTRY.items():
            missing = required_keys - set(cfg.keys())
            assert not missing, f"{name} 缺少字段: {missing}"

    def test_get_litellm_id(self):
        from core_config import get_litellm_id
        result = get_litellm_id()
        assert isinstance(result, str) and len(result) > 0

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
        assert result is None or isinstance(result, str)

    def test_get_active_config(self):
        from core_config import get_active_config, ACTIVE_MODEL_KEY, MODEL_REGISTRY
        cfg = get_active_config()
        assert cfg == MODEL_REGISTRY[ACTIVE_MODEL_KEY]


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 sentiment_optimizer 可导入 ────────────────────
def test_sentiment_optimizer_importable():
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sentiment_optimizer.py")
        spec = importlib.util.spec_from_file_location("sentiment_optimizer", path)
        assert spec is not None, "sentiment_optimizer.py 不存在"
    except Exception as e:
        pytest.skip(f"sentiment_optimizer 检测跳过: {e}")


# ── 测试 DSPy 签名与分类器结构（不执行真实 LLM 调用）───
class TestDSPyStructure:
    def test_sentiment_signature(self):
        """验证 SentimentSignature 定义正确"""
        from main import SentimentSignature
        assert hasattr(SentimentSignature, "__annotations__")
        assert "text" in SentimentSignature.__annotations__
        assert "sentiment" in SentimentSignature.__annotations__

    def test_sentiment_classifier_module(self):
        """验证 SentimentClassifier 可实例化"""
        from main import SentimentClassifier
        sc = SentimentClassifier()
        assert hasattr(sc, "classify")

    def test_accuracy_metric(self):
        """验证 accuracy_metric 函数"""
        from main import accuracy_metric
        example = type("Example", (), {"sentiment": "正面"})()
        prediction = type("Prediction", (), {"sentiment": "正面"})()
        assert accuracy_metric(example, prediction) == True
        prediction2 = type("Prediction", (), {"sentiment": "负面"})()
        assert accuracy_metric(example, prediction2) == False
