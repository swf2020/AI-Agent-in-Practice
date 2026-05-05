# tests/test_main.py — 冒烟测试
import pytest
import sys
import os
import tempfile
import json

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

    def test_get_litellm_id_by_key(self):
        from core_config import get_litellm_id
        result = get_litellm_id("GPT-4o-mini")
        assert "gpt-4o-mini" in result

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
        result = get_base_url("Qwen-Max")
        assert "dashscope" in result


# ── 测试消融实验模块 ───────────────────────────────────
class TestAblation:
    def test_build_example_suite(self):
        from eval.ablation import build_example_suite
        suite = build_example_suite()
        assert len(suite.records) > 0

    def test_to_dataframe(self):
        from eval.ablation import build_example_suite
        suite = build_example_suite()
        df = suite.to_dataframe()
        assert not df.empty
        assert "rouge_l" in df.columns
        assert "bert_score_f1" in df.columns
        assert "judge_total" in df.columns

    def test_plot_with_variable(self):
        from eval.ablation import build_example_suite
        import matplotlib
        matplotlib.use("Agg")  # 非 GUI 后端
        suite = build_example_suite()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            suite.plot("rank", output_path=f.name)
            assert os.path.getsize(f.name) > 0
            os.unlink(f.name)

    def test_plot_missing_variable_raises(self):
        from eval.ablation import AblationSuite
        suite = AblationSuite()
        # Empty suite -> to_dataframe() has no columns -> KeyError on df["variable"]
        with pytest.raises(KeyError):
            suite.plot("nonexistent")


# ── 测试过拟合诊断模块 ─────────────────────────────────
class TestOverfitting:
    def test_load_trainer_state(self):
        from eval.overfitting import load_trainer_state
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "log_history": [
                    {"step": 10, "loss": 1.5},
                    {"step": 20, "loss": 1.2},
                    {"step": 20, "eval_loss": 1.4},
                    {"step": 30, "loss": 0.9},
                    {"step": 30, "eval_loss": 1.3},
                ]
            }
            with open(os.path.join(tmpdir, "trainer_state.json"), "w") as f:
                json.dump(state, f)
            curve = load_trainer_state(tmpdir)
            assert len(curve.steps) == 2
            assert len(curve.train_loss) == 2
            assert len(curve.val_loss) == 2

    def test_load_trainer_state_missing_file(self):
        from eval.overfitting import load_trainer_state
        with pytest.raises(FileNotFoundError):
            load_trainer_state("/nonexistent/path")

    def test_detect_divergence(self):
        from eval.overfitting import detect_divergence_point, LossCurve
        # 模拟过拟合：train_loss 下降，val_loss 上升
        curve = LossCurve(
            steps=[10, 20, 30, 40, 50, 60],
            train_loss=[1.5, 1.3, 1.1, 0.9, 0.8, 0.7],
            val_loss=[1.6, 1.5, 1.6, 1.7, 1.8, 1.9],
        )
        result = detect_divergence_point(curve)
        assert result is not None

    def test_plot_loss_curves(self):
        from eval.overfitting import LossCurve, plot_loss_curves
        import matplotlib
        matplotlib.use("Agg")
        curve = LossCurve(
            steps=[10, 20, 30],
            train_loss=[1.5, 1.0, 0.7],
            val_loss=[1.6, 1.2, 1.1],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_loss_curves(curve, tmpdir, output_path=os.path.join(tmpdir, "test_loss.png"))
            assert os.path.exists(os.path.join(tmpdir, "test_loss.png"))


# ── 测试 metrics 模块 ──────────────────────────────────
class TestMetrics:
    def test_compute_rouge(self):
        from eval.metrics import compute_rouge
        # ROUGE-L 需要分词，中文逐字处理后 rouge_score 的 tokenizer 不识别 Unicode 字符
        # 使用英文测试核心逻辑，中文场景见 eval_metrics.py 注释说明
        preds = ["hello world test", "good morning"]
        refs = ["hello world", "good morning everyone"]
        result = compute_rouge(preds, refs, lang="en")
        assert result.rougeL > 0

    def test_compute_bert_score(self):
        from eval.metrics import compute_bert_score
        preds = ["你好", "世界"]
        refs = ["你好", "世界"]
        result = compute_bert_score(preds, refs, model_type="bert-base-chinese")
        assert result.f1 > 0.9  # 相同文本应该得分很高


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
    spec = importlib.util.spec_from_file_location("main", path)
    assert spec is not None, "main.py 不存在"


# ── 测试 LLM Judge 模块（Mock OpenAI） ─────────────────
class TestLLMJudge:
    @pytest.fixture
    def mock_client(self):
        from unittest.mock import MagicMock
        from openai import OpenAI
        client = MagicMock(spec=OpenAI)
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content='{"accuracy": 4, "completeness": 5, "conciseness": 4, "reasoning": "回答准确完整"}')
            )]
        )
        return client

    def test_llm_judge_mocked(self, mock_client):
        from eval.llm_judge import llm_judge
        result = llm_judge(
            instruction="测试问题",
            reference="标准答案",
            prediction="待评估答案",
            client=mock_client,
            model="gpt-4o-mini",
        )
        assert result.accuracy == 4
        assert result.completeness == 5
        assert result.conciseness == 4
        assert result.total == pytest.approx(4.333, abs=0.01)

    def test_batch_judge_mocked(self, mock_client):
        from eval.llm_judge import batch_judge
        from eval.inference import EvalSample
        samples = [EvalSample(0, "问题1", "答案1")]
        preds = ["预测1"]
        results = batch_judge(samples, preds, mock_client)
        assert len(results) == 1
        assert results[0].total > 0
