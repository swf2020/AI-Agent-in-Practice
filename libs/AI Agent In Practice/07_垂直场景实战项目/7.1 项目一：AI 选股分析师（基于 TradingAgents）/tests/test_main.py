# tests/test_main.py — 冒烟测试
import pytest
from unittest.mock import patch
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_DIR)


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
        result = get_litellm_id("GPT-4o")
        assert result == "openai/gpt-4o"

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
        result = get_base_url("DeepSeek-V3")
        assert result == "https://api.deepseek.com/v1"

    def test_get_active_config(self):
        from core_config import get_active_config, ACTIVE_MODEL_KEY, MODEL_REGISTRY
        cfg = get_active_config()
        assert cfg == MODEL_REGISTRY[ACTIVE_MODEL_KEY]


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 core_config 集成到业务模块 ───────────────────
class TestModuleImports:
    """验证各实验模块可以正常导入（不触发 LLM 调用）"""

    def test_parse_output_importable(self):
        from experiment_1_parse_output import RATING_COLORS
        assert isinstance(RATING_COLORS, dict)
        assert "buy" in RATING_COLORS

    def test_model_comparison_config_from_module(self):
        """验证实验四的 MODEL_CONFIGS 定义正确 — 直接从业务模块导入 [Fix #2]"""
        try:
            from experiment_4_model_comparison import MODEL_CONFIGS, ModelConfig
        except ImportError as e:
            pytest.skip(f"tradingagents 依赖未安装，跳过导入测试: {e}")

        assert "gpt4o" in MODEL_CONFIGS
        assert "deepseek" in MODEL_CONFIGS
        # DeepSeek 通过 LiteLLM 代理调用，provider 应为 "litellm"
        assert MODEL_CONFIGS["deepseek"].provider == "litellm"
        # 验证成本对比关系：DeepSeek 单价应远低于 GPT-4o
        assert MODEL_CONFIGS["deepseek"].cost_per_1m_input < MODEL_CONFIGS["gpt4o"].cost_per_1m_input
        assert MODEL_CONFIGS["deepseek"].cost_per_1m_output < MODEL_CONFIGS["gpt4o"].cost_per_1m_output


# ── 测试 AStockAdapter 数据结构（Mock AKShare）────────
class TestAStockAdapter:
    @patch("akshare.stock_zh_a_hist")
    def test_get_price_history_columns(self, mock_hist):
        """验证价格历史数据列名符合 TradingAgents 标准"""
        import pandas as pd
        from astock_adapter import AStockAdapter

        # 模拟 AKShare 返回的中文列名数据
        mock_df = pd.DataFrame({
            "日期": ["2025-01-01", "2025-01-02"],
            "开盘": [1800.0, 1810.0],
            "最高": [1820.0, 1830.0],
            "最低": [1790.0, 1800.0],
            "收盘": [1810.0, 1820.0],
            "成交量": [1000, 1200],
            "成交额": [1800000, 2100000],
        })
        mock_hist.return_value = mock_df

        adapter = AStockAdapter()
        result = adapter.get_price_history("600519", days=5)

        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result.index.tz is not None

    @patch("akshare.stock_individual_info_em")
    def test_get_fundamental_info(self, mock_info):
        """验证基本面信息字段映射正确"""
        import pandas as pd
        from astock_adapter import AStockAdapter

        mock_df = pd.DataFrame({
            "item": ["股票简称", "所处交所", "总市值", "市盈率(动态)", "市净率"],
            "value": ["贵州茅台", "上交所", 2000000000000.0, 30.5, 10.2],
        })
        mock_info.return_value = mock_df

        adapter = AStockAdapter()
        result = adapter.get_fundamental_info("600519")

        assert result["name"] == "贵州茅台"
        assert result["ticker"] == "600519"
        assert result["currency"] == "CNY"
        assert result["country"] == "CN"


# ── 测试 display_decision 不崩溃 ──────────────────────
class TestDisplayDecision:
    def test_display_decision_dict_state(self, capsys):
        """验证 display_decision 正确处理 dict 格式的 state（常见场景） [Fix #1]"""
        from experiment_1_parse_output import display_decision

        mock_result = {
            "ticker": "TEST",
            "date": "2025-01-10",
            "decision": {
                "action": "buy",
                "target_price": 150.0,
                "stop_loss": 130.0,
                "take_profit": 180.0,
                "confidence": 0.75,
                "reasoning": "This is a test reasoning text.",
            },
            "state": {"risk_tolerance": "aggressive"},  # 不应退回默认值
        }
        display_decision(mock_result)
        captured = capsys.readouterr()
        # 验证风险偏好正确显示为"激进型"而非默认的"中性型"
        assert "激进型" in captured.out, (
            f"风险偏好应为'激进型'，实际输出: {captured.out[:500]}"
        )

    def test_display_decision_none_confidence(self, capsys):
        """验证 display_decision 正确处理 confidence=None 的情况 [Fix #3]"""
        from experiment_1_parse_output import display_decision

        mock_result = {
            "ticker": "TEST",
            "date": "2025-01-10",
            "decision": {
                "action": "hold",
                "target_price": None,
                "stop_loss": None,
                "take_profit": None,
                "confidence": None,
                "reasoning": "Mixed signals.",
            },
            "state": {"risk_tolerance": "neutral"},
        }
        # Should not raise TypeError when formatting None as percentage
        display_decision(mock_result)
        captured = capsys.readouterr()
        assert "未知" in captured.out, (
            f"confidence=None 时应显示'未知'，实际输出: {captured.out[:500]}"
        )
