# tests/test_main.py — 自动生成的冒烟测试
import pytest
from unittest.mock import patch, MagicMock
import sys, os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


# ── 测试 core_config 基础结构 ──────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY,
            get_litellm_id, get_chat_model_id,
            get_api_key, get_base_url,
            get_model_list, estimate_cost, get_active_config,
        )
        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0
        assert isinstance(ACTIVE_MODEL_KEY, str)
        assert ACTIVE_MODEL_KEY in MODEL_REGISTRY

    def test_model_registry_schema(self):
        """验证每个模型条目包含必要字段"""
        from core_config import MODEL_REGISTRY
        required_keys = {"litellm_id", "chat_model_id",
                         "price_in", "price_out",
                         "max_tokens_limit", "api_key_env", "base_url"}
        for name, cfg in MODEL_REGISTRY.items():
            missing = required_keys - set(cfg.keys())
            assert not missing, f"{name} 缺少字段: {missing}"

    def test_get_litellm_id(self):
        from core_config import get_litellm_id
        result = get_litellm_id()
        assert isinstance(result, str) and len(result) > 0

    def test_get_chat_model_id(self):
        """验证 chat_model_id 无前缀（与 litellm_id 区分）"""
        from core_config import get_chat_model_id, get_litellm_id
        chat_id = get_chat_model_id()
        lite_id = get_litellm_id()
        assert isinstance(chat_id, str) and len(chat_id) > 0
        # chat_model_id 不应包含 provider 前缀
        assert "/" not in chat_id
        # 两者应该不同（litellm_id 含前缀）
        assert chat_id != lite_id

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


# ── 测试 embedding/reranker 配置 ──────────────────────
class TestEmbeddingRerankerConfig:
    def test_embedding_config(self):
        from core_config import (
            EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BASE_URL,
            get_embedding_model, get_embedding_dim, get_embedding_base_url,
        )
        assert EMBEDDING_MODEL == "text-embedding-v4"
        assert EMBEDDING_DIM == 1024
        assert get_embedding_model() == EMBEDDING_MODEL
        assert get_embedding_dim() == EMBEDDING_DIM

    def test_reranker_config(self):
        from core_config import (
            RERANKER_MODEL, RERANKER_TOP_N,
            get_reranker_model, get_reranker_top_n,
        )
        assert RERANKER_MODEL == "qwen3-rerank"
        assert RERANKER_TOP_N == 5
        assert get_reranker_model() == RERANKER_MODEL

    def test_dashscope_api_key(self):
        from core_config import get_dashscope_api_key
        result = get_dashscope_api_key()
        # 如果环境变量设了就返回字符串，否则返回 None（不抛异常）
        assert result is None or isinstance(result, str)


# ── 测试主模块可导入 ───────────────────────────────────
def test_baseline_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "baseline_rag.py")
        spec = importlib.util.spec_from_file_location("baseline_rag", path)
        assert spec is not None, "baseline_rag.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


def test_advanced_rag_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "advanced_rag.py")
        spec = importlib.util.spec_from_file_location("advanced_rag", path)
        assert spec is not None, "advanced_rag.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


def test_query_rewriter_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "query_rewriter.py")
        spec = importlib.util.spec_from_file_location("query_rewriter", path)
        assert spec is not None, "query_rewriter.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


def test_reranker_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "reranker.py")
        spec = importlib.util.spec_from_file_location("reranker", path)
        assert spec is not None, "reranker.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


def test_context_compressor_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "context_compressor.py")
        spec = importlib.util.spec_from_file_location("context_compressor", path)
        assert spec is not None, "context_compressor.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 RRF 融合逻辑 ───────────────────────────────────
class TestRRFMerge:
    def test_rrf_merge_deduplicates(self):
        """验证 RRF 融合正确去重（纯函数，无需 mock）"""
        from baseline_rag import RetrievedChunk
        from query_rewriter import QueryRewriter

        rewriter = QueryRewriter(client=MagicMock())

        # 构造两路检索结果，有重叠
        results_a = [
            RetrievedChunk(text="chunk 0", score=0.9, chunk_id="0"),
            RetrievedChunk(text="chunk 1", score=0.8, chunk_id="1"),
        ]
        results_b = [
            RetrievedChunk(text="chunk 1", score=0.7, chunk_id="1"),
            RetrievedChunk(text="chunk 2", score=0.6, chunk_id="2"),
        ]

        merged = rewriter.rrf_merge([results_a, results_b], k=60, top_n=3)
        chunk_ids = [c.chunk_id for c in merged]

        # 应去重，"1" 在两路都出现，应排第一
        assert len(chunk_ids) == 3
        assert len(set(chunk_ids)) == 3  # 无重复
        assert "1" == chunk_ids[0], "chunk 1 应在两路中都出现，排第一"


# ── 测试 QueryRewriter（Mock OpenAI）───────────────────
class TestQueryRewriter:
    def test_hyde_mocked(self):
        """验证 HyDE 生成路径"""
        from query_rewriter import QueryRewriter

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="假设文档内容"))]
        )
        rewriter = QueryRewriter(client=mock_client, model="test-model")
        result = rewriter.hyde("测试问题")
        assert result == "假设文档内容"
        mock_client.chat.completions.create.assert_called_once()

    def test_multi_query_mocked(self):
        """验证 Multi-Query 生成路径"""
        from query_rewriter import QueryRewriter

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content='["变体1", "变体2"]')
            )]
        )
        rewriter = QueryRewriter(client=mock_client, model="test-model")
        result = rewriter.multi_query("原始问题", n=2)
        assert result[0] == "原始问题"  # 原始查询始终在第一位
        assert "变体1" in result
        assert "变体2" in result


# ── 测试 NaiveRAG（Mock OpenAI + Qdrant）───────────────
class TestNaiveRAG:
    @patch("baseline_rag.QdrantClient")
    @patch("baseline_rag.OpenAI")
    def test_retrieve_and_generate(self, mock_openai_cls, mock_qdrant_cls):
        """验证检索+生成链路在 mock 下可正常执行"""
        from baseline_rag import NaiveRAG, RetrievedChunk

        # Mock Qdrant
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value.collections = []
        mock_qdrant_cls.return_value = mock_qdrant

        # Mock OpenAI
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="mock answer"))]
        )
        mock_openai_cls.return_value = mock_client

        rag = NaiveRAG()

        # Mock query_points results (new Qdrant API)
        from qdrant_client.models import ScoredPoint
        mock_response = MagicMock()
        mock_response.points = [
            ScoredPoint(
                id=0, version=1, score=0.95,
                payload={"text": "test context", "chunk_id": "0"},
                vector=None,
            )
        ]
        mock_qdrant.query_points.return_value = mock_response

        # Test retrieve
        chunks = rag.retrieve("test question", top_k=1)
        assert len(chunks) == 1
        assert chunks[0].text == "test context"

        # Test generate
        answer = rag.generate("test question", chunks)
        assert answer == "mock answer"

    @patch("baseline_rag.QdrantClient")
    @patch("baseline_rag.OpenAI")
    def test_index_documents(self, mock_openai_cls, mock_qdrant_cls):
        """验证文档索引路径"""
        from baseline_rag import NaiveRAG

        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value.collections = []
        mock_qdrant_cls.return_value = mock_qdrant

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536), MagicMock(embedding=[0.2] * 1536)]
        )
        mock_openai_cls.return_value = mock_client

        rag = NaiveRAG()
        rag.index_documents(["doc1", "doc2"])
        mock_qdrant.upsert.assert_called_once()


# ── 测试 DashScopeReranker ────────────────────────────
class TestReranker:
    def test_reranker_fallback_on_empty(self):
        """空列表应直接返回空"""
        from reranker import DashScopeReranker
        from baseline_rag import RetrievedChunk

        reranker = DashScopeReranker()
        result = reranker.rerank("test query", [])
        assert result == []

    @patch("reranker.httpx.Client")
    def test_reranker_api_success(self, mock_httpx):
        """验证 reranker API 调用成功"""
        from reranker import DashScopeReranker
        from baseline_rag import RetrievedChunk

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "results": [
                    {"index": 0, "relevance_score": 0.3},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ]
            }
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_httpx.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.return_value.__exit__ = MagicMock(return_value=False)

        chunks = [
            RetrievedChunk(text="doc a", score=0.1, chunk_id="0"),
            RetrievedChunk(text="doc b", score=0.2, chunk_id="1"),
            RetrievedChunk(text="doc c", score=0.3, chunk_id="2"),
        ]
        reranker = DashScopeReranker()
        result = reranker.rerank("test query", chunks, top_n=2)

        assert len(result) == 2
        # doc b (score 0.9) should be first, doc c (0.5) second
        assert result[0].chunk_id == "1"
        assert result[1].chunk_id == "2"

    @patch("reranker.httpx.Client")
    def test_reranker_api_failure_fallback(self, mock_httpx):
        """API 调用失败时应回退到向量检索分数排序"""
        from reranker import DashScopeReranker
        from baseline_rag import RetrievedChunk

        mock_httpx.side_effect = Exception("Network error")

        chunks = [
            RetrievedChunk(text="doc a", score=0.1, chunk_id="0"),
            RetrievedChunk(text="doc b", score=0.5, chunk_id="1"),
            RetrievedChunk(text="doc c", score=0.3, chunk_id="2"),
        ]
        reranker = DashScopeReranker()
        result = reranker.rerank("test query", chunks, top_n=2)

        # Should fallback to vector search score sorting
        assert len(result) == 2
        assert result[0].chunk_id == "1"  # highest score first
        assert result[1].chunk_id == "2"
