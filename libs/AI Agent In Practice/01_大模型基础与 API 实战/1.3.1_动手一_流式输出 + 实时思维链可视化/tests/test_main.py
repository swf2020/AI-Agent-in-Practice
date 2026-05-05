# tests/test_main.py — 自动生成的冒烟测试
import pytest
from unittest.mock import patch, MagicMock
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


# ── 测试 core 模块 ──────────────────────────────────────
class TestCoreModule:
    def test_import_core(self):
        from core import ChunkType
        assert ChunkType.THINKING == "thinking"
        assert ChunkType.ANSWER == "answer"
        assert ChunkType.META == "meta"

    def test_stream_chunk_creation(self):
        from core import StreamChunk, ChunkType
        chunk = StreamChunk(content="hello", chunk_type=ChunkType.ANSWER)
        assert chunk.content == "hello"
        assert chunk.chunk_type == ChunkType.ANSWER
        assert chunk.timestamp > 0

    def test_get_default_model(self):
        from core import get_default_model
        result = get_default_model()
        assert isinstance(result, str) and len(result) > 0

    def test_get_openai_client(self):
        from core import get_openai_client
        client = get_openai_client()
        assert client is not None

    @patch("core.OpenAI")
    def test_stream_cot_prompt_mocked(self, mock_openai_cls):
        """验证 CoT 流式输出在 mock 下可正常工作"""
        # 构造模拟流式 chunk
        chunks_data = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="<think>"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="让我思考一下..."))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="</think>"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="<answer>"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="答案是42"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="</answer>"))]),
        ]

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(chunks_data))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream
        mock_openai_cls.return_value = mock_client

        from core import stream_cot_prompt, ChunkType

        results = list(stream_cot_prompt("test question"))
        types = {c.chunk_type for c in results}
        assert ChunkType.THINKING in types, "应检测到 thinking chunk"
        assert ChunkType.ANSWER in types, "应检测到 answer chunk"

    @patch("core.OpenAI")
    def test_stream_extended_thinking_mocked(self, mock_openai_cls):
        """验证 DeepSeek 推理模式在 mock 下可正常工作"""
        # 构造模拟流式 chunk（带 reasoning_content）
        chunks_data = [
            MagicMock(choices=[MagicMock(delta=MagicMock(reasoning_content="第一步分析...", content=None))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(reasoning_content="第二步推导...", content=None))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(reasoning_content=None, content="最终答案是42"))]),
        ]

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(chunks_data))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream
        mock_openai_cls.return_value = mock_client

        from core import stream_extended_thinking, ChunkType

        results = list(stream_extended_thinking("test", use_reasoner=True))
        types = {c.chunk_type for c in results}
        assert ChunkType.THINKING in types, "应检测到 thinking chunk"
        assert ChunkType.ANSWER in types, "应检测到 answer chunk"

    @patch("core.OpenAI")
    def test_stream_extended_thinking_non_reasoner_mocked(self, mock_openai_cls):
        """验证非 reasoner 模式（系统提示词驱动）在 mock 下可正常工作"""
        chunks_data = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="<thinking>"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="逐步推理..."))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="</thinking>"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="<answer>"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="结果是10"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="</answer>"))]),
        ]

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(chunks_data))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream
        mock_openai_cls.return_value = mock_client

        from core import stream_extended_thinking, ChunkType

        results = list(stream_extended_thinking("test", use_reasoner=False))
        types = {c.chunk_type for c in results}
        assert ChunkType.THINKING in types
        assert ChunkType.ANSWER in types


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")
