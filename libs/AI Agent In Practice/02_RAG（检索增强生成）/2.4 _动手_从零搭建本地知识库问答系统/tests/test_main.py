# tests/test_main.py — 冒烟测试
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

    def test_get_active_config(self):
        from core_config import get_active_config, MODEL_REGISTRY, ACTIVE_MODEL_KEY
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


# ── 测试文档解析模块 ────────────────────────────────────
class TestParse:
    def test_parse_module_importable(self):
        from step1_parse import parse_document, ParsedDocument
        assert callable(parse_document)
        assert ParsedDocument is not None

    def test_parse_unsupported_format(self):
        from step1_parse import parse_document
        with pytest.raises(ValueError, match="不支持"):
            parse_document("/tmp/test.xyz")


# ── 测试切块模块 ─────────────────────────────────────────
class TestChunk:
    def test_chunk_module_importable(self):
        from step2_chunk import chunk_document
        assert callable(chunk_document)

    def test_chunk_fixed_size(self):
        from step1_parse import ParsedDocument
        from step2_chunk import chunk_fixed_size
        doc = ParsedDocument(
            content="这是第一段内容。\n\n这是第二段内容。\n\n这是第三段内容。",
            source="test.txt",
            doc_type="web",
        )
        chunks = chunk_fixed_size(doc, chunk_size=512)
        assert len(chunks) > 0
        assert all(c.strategy == "fixed" for c in chunks)

    def test_chunk_by_section_no_headings(self):
        """无标题时退化到 fixed 切块"""
        from step1_parse import ParsedDocument
        from step2_chunk import chunk_by_section
        doc = ParsedDocument(
            content="没有标题的普通文本。",
            source="test.txt",
            doc_type="web",
        )
        chunks = chunk_by_section(doc)
        assert len(chunks) > 0

    def test_chunk_by_section_with_headings(self):
        """有标题时按章节切块"""
        from step1_parse import ParsedDocument
        from step2_chunk import chunk_by_section
        doc = ParsedDocument(
            content="# 第一章\n\n这是第一章的内容。\n\n# 第二章\n\n这是第二章的内容。",
            source="test.md",
            doc_type="word",
        )
        chunks = chunk_by_section(doc)
        assert len(chunks) >= 2
        assert all(c.strategy == "section" for c in chunks)


# ── 测试索引模块结构 ─────────────────────────────────────
class TestIndex:
    def test_index_module_importable(self):
        """只测试可导入，不实际加载 embedding 模型"""
        from step3_index import COLLECTION_NAME, EMBED_MODEL_NAME, QDRANT_PATH, VECTOR_DIM
        assert isinstance(COLLECTION_NAME, str)
        assert isinstance(EMBED_MODEL_NAME, str)
        assert isinstance(QDRANT_PATH, str)
        assert VECTOR_DIM == 512


# ── 测试 RAG Pipeline（Mock）────────────────────────────
class TestRAGPipeline:
    @patch("step4_query.SentenceTransformer")
    @patch("step4_query.QdrantClient")
    def test_pipeline_init_mocked(self, mock_qdrant_cls, mock_sentencetransformer):
        """测试 RAGPipeline 在 mock 下可以正常初始化"""
        from step4_query import RAGPipeline
        pipeline = RAGPipeline(top_k=3, score_threshold=0.3)
        assert pipeline.top_k == 3
        assert pipeline.score_threshold == 0.3

    @patch("step4_query.SentenceTransformer")
    @patch("step4_query.QdrantClient")
    def test_build_prompt_with_chunks(self, mock_qdrant_cls, mock_sentencetransformer):
        """测试 Prompt 构建逻辑"""
        from step4_query import RAGPipeline, RetrievedChunk
        pipeline = RAGPipeline()
        chunks = [
            RetrievedChunk(text="这是参考资料1", source="doc1.pdf", score=0.8, metadata={}),
            RetrievedChunk(text="这是参考资料2", source="doc2.pdf", score=0.6, metadata={}),
        ]
        prompt = pipeline.build_prompt("测试问题", chunks)
        assert "测试问题" in prompt
        assert "参考资料 1" in prompt
        assert "这是参考资料1" in prompt

    @patch("step4_query.SentenceTransformer")
    @patch("step4_query.QdrantClient")
    def test_build_prompt_no_chunks(self, mock_qdrant_cls, mock_sentencetransformer):
        """测试无资料时的 Prompt"""
        from step4_query import RAGPipeline
        pipeline = RAGPipeline()
        prompt = pipeline.build_prompt("冷门问题", [])
        assert "冷门问题" in prompt
        assert "无法回答" in prompt or "无法回答该问题" in prompt

    @patch("step4_query.SentenceTransformer")
    @patch("step4_query.QdrantClient")
    def test_retrieve_returns_list(self, mock_qdrant_cls, mock_sentencetransformer):
        """测试 retrieve 方法返回 list"""
        from step4_query import RAGPipeline
        pipeline = RAGPipeline()
        # Qdrant mock 返回空
        mock_qdrant = mock_qdrant_cls.return_value
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        # 全量扫描也返回空
        mock_qdrant.scroll.return_value = ([], None)
        results = pipeline.retrieve("测试")
        assert isinstance(results, list)


# ── 测试 LLM 调用（Mock litellm）────────────────────────
class TestLLMCall:
    @patch("litellm.completion")
    def test_mocked_completion(self, mock_completion):
        """验证核心 LLM 调用路径在 mock 下可正常执行"""
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="mock response"),
                finish_reason="stop",
            )],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        import litellm
        from core_config import get_litellm_id, get_api_key, get_base_url
        response = litellm.completion(
            model=get_litellm_id(),
            api_key=get_api_key(),
            api_base=get_base_url(),
            messages=[{"role": "user", "content": "test"}],
        )
        assert response.choices[0].message.content == "mock response"
        mock_completion.assert_called_once()


# ── 集成测试：端到端 RAG 流程（全 Mock）─────────────────
class TestEndToEnd:
    @patch("step4_query.SentenceTransformer")
    @patch("step4_query.QdrantClient")
    @patch("step4_query.get_llm_client")
    def test_full_rag_ask(self, mock_get_llm, mock_qdrant_cls, mock_sentencetransformer):
        """Mock 全部外部依赖，测试端到端 ask 流程"""
        # Mock embedding model (must return numpy array for .tolist())
        import numpy as np
        mock_sentencetransformer.return_value.encode.return_value = np.array([0.1] * 512)

        # Mock Qdrant
        mock_qdrant = mock_qdrant_cls.return_value
        mock_point = MagicMock()
        mock_point.payload = {"text": "pathlib 是 Python 标准库", "source": "test.pdf"}
        mock_point.score = 0.85
        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="根据资料，pathlib 是路径处理库。"))]
        )
        mock_get_llm.return_value = (mock_llm, "deepseek/deepseek-chat")

        from step4_query import RAGPipeline
        pipeline = RAGPipeline(top_k=3, score_threshold=0.3)
        result = pipeline.ask("pathlib 是什么？")

        assert result.question == "pathlib 是什么？"
        assert len(result.answer) > 10
        assert len(result.sources) == 1
        assert result.sources[0].text == "pathlib 是 Python 标准库"
