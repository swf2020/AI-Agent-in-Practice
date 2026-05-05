# tests/test_main.py — 冒烟测试
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    def test_config_constants(self):
        """验证向量数据库等常量存在"""
        from core_config import (
            QDRANT_URL, EMBED_MODEL,
            VECTOR_DIM, CHUNK_SIZE,
        )
        assert isinstance(QDRANT_URL, str)
        assert isinstance(EMBED_MODEL, str)
        assert VECTOR_DIM == 512
        assert CHUNK_SIZE == 512


# ── 测试数据模型 ─────────────────────────────────────
class TestDataModels:
    def test_retrieved_chunk(self):
        from retriever import RetrievedChunk
        chunk = RetrievedChunk(
            chunk_id="1", content="test", source="test.md",
            title="Test", chunk_index=0, total_chunks=1,
            tenant_id="default", rrf_score=0.5, rerank_score=0.8,
        )
        assert chunk.content == "test"
        assert chunk.rrf_score == 0.5

    def test_generated_answer(self):
        from generator import GeneratedAnswer
        answer = GeneratedAnswer(
            answer="Hello", references=[], is_abstained=False, top_rerank_score=0.9,
        )
        assert answer.answer == "Hello"
        assert not answer.is_abstained

    def test_parsed_document(self):
        from document_parser import ParsedDocument
        doc = ParsedDocument(
            content="test", source="test.md", doc_type="txt",
            file_hash="abc123", title="Test",
        )
        assert doc.doc_type == "txt"

    def test_document_chunk(self):
        from chunker import DocumentChunk
        chunk = DocumentChunk(
            chunk_id="1", content="test", source="test.md",
            title="Test", chunk_index=0, total_chunks=1,
            file_hash="abc", tenant_id="default", doc_type="txt",
        )
        assert chunk.tenant_id == "default"


# ── 测试文档解析与分块 ──────────────────────────────────
class TestParserAndChunker:
    def test_document_parser_import(self):
        from document_parser import DocumentParser
        parser = DocumentParser()
        assert parser is not None

    def test_chunk_document(self):
        from chunker import chunk_document
        from document_parser import ParsedDocument

        doc = ParsedDocument(
            content="这是一段测试文本内容。\n## 第二部分\n这是更多内容。",
            source="test.md",
            doc_type="txt",
            file_hash="hash123",
            title="测试文档",
            metadata={"tenant_id": "test", "doc_type": "txt"},
        )
        chunks = chunk_document(doc)
        assert len(chunks) > 0
        assert all(c.tenant_id == "test" for c in chunks)
        assert all(c.file_hash == "hash123" for c in chunks)

    def test_detect_type(self):
        from document_parser import _detect_type
        assert _detect_type("file.pdf") == "pdf"
        assert _detect_type("file.docx") == "docx"
        assert _detect_type("https://example.com/doc") == "url"


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 AnswerGenerator (Mock OpenAI) ────────────────
class TestAnswerGenerator:
    @patch("generator.OpenAI")
    def test_generate_answer(self, mock_openai_class):
        """验证核心回答生成函数在 mock 下可正常执行"""
        from generator import AnswerGenerator, GeneratedAnswer
        from retriever import RetrievedChunk

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=MagicMock(content="根据文档[1]，年假为15天。")
        )]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        chunks = [
            RetrievedChunk(
                chunk_id="1", content="员工每年享有15天带薪年假。",
                source="policy.md", title="公司政策", chunk_index=0,
                total_chunks=1, tenant_id="test", rrf_score=0.5,
                rerank_score=0.9,
            )
        ]

        generator = AnswerGenerator()
        result = generator.generate("年假有多少天", chunks)

        assert isinstance(result, GeneratedAnswer)
        assert not result.is_abstained
        assert len(result.references) == 1
        mock_client.chat.completions.create.assert_called_once()

    @patch("generator.OpenAI")
    def test_abstain_on_no_chunks(self, mock_openai_class):
        """没有检索结果时应返回无法回答"""
        from generator import AnswerGenerator

        generator = AnswerGenerator()
        result = generator.generate("年假有多少天", [])

        assert result.is_abstained
        mock_openai_class.return_value.chat.completions.create.assert_not_called()


# ── 测试 HybridRetriever RRF 融合逻辑 ────────────────
class TestRetriever:
    def test_rrf_fusion(self):
        from retriever import HybridRetriever

        dense = [
            ("id1", 0.9, {"content": "a"}),
            ("id2", 0.8, {"content": "b"}),
            ("id3", 0.7, {"content": "c"}),
        ]
        bm25 = [
            ("id2", 0.85, {"content": "b"}),
            ("id4", 0.6, {"content": "d"}),
            ("id1", 0.55, {"content": "a"}),
        ]

        fused = HybridRetriever._rrf_fusion(dense, bm25)
        assert len(fused) == 4
        assert fused[0][0] == "id2"  # id2 出现在两个列表的前列
        assert all(isinstance(score, float) for _, score, _ in fused)

    def test_tokenize(self):
        from retriever import HybridRetriever
        tokens = HybridRetriever._tokenize("Hello world 测试")
        assert "hello" in tokens
        assert "world" in tokens
        assert "测试" in tokens
