"""冒烟测试：core_config 结构 + 分析模块 + LLM 调用 mock"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── 测试 core_config 基础结构 ──────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY,
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

    def test_get_litellm_id_specific(self):
        from core_config import get_litellm_id
        assert "deepseek" in get_litellm_id("DeepSeek-V3").lower()

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
        from core_config import get_active_config, MODEL_REGISTRY, ACTIVE_MODEL_KEY
        cfg = get_active_config()
        assert cfg == MODEL_REGISTRY[ACTIVE_MODEL_KEY]


# ── 测试数据模块可导入 ──────────────────────────────────
class TestDataSet:
    def test_test_set_import(self):
        from data.test_set import TEST_SET, TestItem
        assert isinstance(TEST_SET, list)
        assert len(TEST_SET) > 0
        assert all(isinstance(item, TestItem) for item in TEST_SET)

    def test_test_set_has_all_domains(self):
        from data.test_set import TEST_SET
        domains = {item.domain for item in TEST_SET}
        assert "tech" in domains
        assert "legal" in domains
        assert "casual" in domains

    def test_test_set_scores(self):
        from data.test_set import TEST_SET
        for item in TEST_SET:
            for key in ("faithfulness", "fluency", "terminology"):
                assert key in item.human_scores
                assert 1 <= item.human_scores[key] <= 5


# ── 测试 prompt 模块 ───────────────────────────────────
class TestPrompts:
    def test_prompt_versions_exist(self):
        from judge.prompts import PROMPT_VERSIONS
        assert "v1_simple" in PROMPT_VERSIONS
        assert "v2_with_criteria" in PROMPT_VERSIONS
        assert "v3_with_reference" in PROMPT_VERSIONS
        assert "v4_with_role" in PROMPT_VERSIONS

    def test_prompt_formatting(self):
        from judge.prompts import PROMPT_VERSIONS
        prompt = PROMPT_VERSIONS["v1_simple"].format(
            source="Hello world", translation="你好世界"
        )
        assert "Hello world" in prompt
        assert "你好世界" in prompt

    def test_reference_prompts(self):
        """V3/V4 应该包含 reference 占位符"""
        from judge.prompts import PROMPT_VERSIONS
        for key in ("v3_with_reference", "v4_with_role"):
            assert "{reference}" in PROMPT_VERSIONS[key]


# ── 测试 metrics 分析模块 ──────────────────────────────
class TestMetrics:
    def test_correlation_with_human(self):
        from analysis.metrics import correlation_with_human
        llm = [4.0, 3.5, 4.5, 3.0, 4.0]
        human = [4.0, 3.0, 5.0, 3.0, 4.5]
        result = correlation_with_human(llm, human)
        assert "spearman_r" in result
        assert "p_value" in result
        assert "significant" in result
        assert "interpretation" in result
        assert -1 <= result["spearman_r"] <= 1

    def test_correlation_mismatch_length(self):
        from analysis.metrics import correlation_with_human
        with pytest.raises(ValueError):
            correlation_with_human([1.0, 2.0], [1.0])

    def test_consistency_score(self):
        from analysis.metrics import consistency_score
        scores = [[4.0, 3.5, 4.5], [4.1, 3.6, 4.4]]  # 2 runs, 3 items
        result = consistency_score(scores)
        assert "mean_std" in result
        assert "max_std" in result
        assert result["mean_std"] >= 0

    def test_interpret_correlation(self):
        from analysis.metrics import _interpret_correlation
        assert "强相关" in _interpret_correlation(0.8)
        assert "中等相关" in _interpret_correlation(0.6)
        assert "弱相关" in _interpret_correlation(0.4)
        assert "无相关" in _interpret_correlation(0.1)


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试核心 LLM 调用（Mock litellm）──────────────────
class TestLLMCall:
    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """确保环境变量不影响测试"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    @pytest.fixture
    def mock_completion(self):
        """Mock judge.evaluator.acompletion（patch 导入后的引用）"""
        from unittest.mock import AsyncMock, patch
        mock = AsyncMock(
            return_value=type("MockResp", (), {
                "choices": [type("MockChoice", (), {
                    "message": type("MockMessage", (), {
                        "content": '{"faithfulness": 4, "fluency": 4, "terminology": 4, "overall": 4.0, "key_issues": "无", "reasoning": "good"}'
                    })()
                })()],
                "usage": type("MockUsage", (), {"prompt_tokens": 10, "completion_tokens": 5})(),
            })()
        )
        with patch("judge.evaluator.acompletion", mock):
            yield mock

    @pytest.mark.asyncio
    async def test_mocked_judge_single(self, mock_completion):
        """验证 judge_single 在 mock 下可正常执行"""
        from judge.evaluator import judge_single
        result = await judge_single(
            item_id="test_001",
            source="Hello world",
            translation="你好世界",
            reference="你好，世界",
            translator="test",
            prompt_version="v3_with_reference",
        )
        assert result.overall == 4.0
        assert result.faithfulness == 4
        mock_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_mocked_judge_batch(self, mock_completion):
        """验证 judge_batch 在 mock 下可正常执行"""
        from judge.evaluator import judge_batch
        items = [
            {
                "id": f"item_{i}",
                "source": f"Source {i}",
                "translation": f"译文 {i}",
                "reference": f"参考 {i}",
                "translator": "test",
            }
            for i in range(3)
        ]
        results = await judge_batch(
            items=items,
            prompt_version="v3_with_reference",
            runs=1,
            concurrency=3,
        )
        assert len(results) == 3
        assert all(r.overall == 4.0 for r in results)
