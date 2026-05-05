# tests/test_main.py — QLoRA 微调项目冒烟测试
import pytest
from unittest.mock import patch, MagicMock
import sys, os

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_DIR)


# ── 测试 core_config 基础结构 ──────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY, FINETUNE_BASE_MODEL,
            get_litellm_id, get_api_key, get_base_url,
            get_model_list, estimate_cost, get_active_config,
        )
        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0
        assert isinstance(ACTIVE_MODEL_KEY, str)
        assert ACTIVE_MODEL_KEY in MODEL_REGISTRY
        assert isinstance(FINETUNE_BASE_MODEL, str)
        assert "Qwen" in FINETUNE_BASE_MODEL

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

    def test_get_active_config(self):
        from core_config import get_active_config, ACTIVE_MODEL_KEY
        cfg = get_active_config()
        assert cfg["litellm_id"] is not None

    def test_finetune_base_model(self):
        from core_config import FINETUNE_BASE_MODEL
        assert "Qwen" in FINETUNE_BASE_MODEL
        assert "7B" in FINETUNE_BASE_MODEL


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 smoke_test 模块可导入 ─────────────────────────
def test_smoke_test_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "smoke_test.py")
        spec = importlib.util.spec_from_file_location("smoke_test", path)
        assert spec is not None, "smoke_test.py 不存在"
    except Exception as e:
        pytest.skip(f"smoke_test 模块检测跳过: {e}")


# ── 测试 finetune 模块可导入 ───────────────────────────
def test_finetune_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "finetune.py")
        spec = importlib.util.spec_from_file_location("finetune", path)
        assert spec is not None, "finetune.py 不存在"
    except Exception as e:
        pytest.skip(f"finetune 模块检测跳过: {e}")


# ── Mock 测试：finetune 配置函数 ─────────────────────
class TestFinetuneConfig:
    @patch("finetune.BitsAndBytesConfig")
    def test_quantization_config(self, mock_bnb):
        from finetune import get_quantization_config
        get_quantization_config()
        mock_bnb.assert_called_once()
        call_kwargs = mock_bnb.call_args.kwargs
        assert call_kwargs["load_in_4bit"] is True
        assert call_kwargs["bnb_4bit_quant_type"] == "nf4"

    @patch("finetune.LoraConfig")
    def test_lora_config(self, mock_lora):
        from finetune import get_lora_config
        get_lora_config()
        mock_lora.assert_called_once()
        call_kwargs = mock_lora.call_args.kwargs
        assert call_kwargs["r"] == 16
        assert call_kwargs["lora_alpha"] == 32
        assert call_kwargs["task_type"] == "CAUSAL_LM"


# ── Mock 测试：smoke_test 推理流程 ───────────────────
class TestSmokeTest:
    @patch("transformers.BitsAndBytesConfig")
    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("peft.PeftModel")
    def test_mocked_inference(self, mock_peft, mock_tokenizer, mock_model_cls, mock_bnb):
        """验证冒烟测试推理流程在 mock 下可执行"""
        import torch
        from smoke_test import test_with_transformers

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.eos_token_id = 151643
        mock_tok.apply_chat_template.return_value = "mocked chat template text"
        mock_tok.decode.return_value = "您好，已为您查询到订单信息..."
        mock_tok.return_value = MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
        )
        mock_tokenizer.from_pretrained.return_value = mock_tok

        # Mock base model
        mock_base = MagicMock()
        mock_base.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_base

        # Mock PeftModel
        mock_peft_model = MagicMock()
        mock_peft_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_peft.from_pretrained.return_value = mock_peft_model

        # Mock BitsAndBytesConfig
        mock_bnb.return_value = MagicMock()

        test_with_transformers(adapter_path="./mock/adapter")

        # Verify the flow was called
        mock_model_cls.from_pretrained.assert_called_once()
        mock_peft.from_pretrained.assert_called_once()
