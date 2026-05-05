from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    source: str
    title: str
    chunk_index: int
    total_chunks: int
    tenant_id: str
    rrf_score: float
    rerank_score: float


_RRF_K = 60
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _get_default_reranker() -> CrossEncoder:
    """创建默认 CrossEncoder 实例（延迟初始化，避免 import 时下载模型）"""
    return CrossEncoder(_RERANKER_MODEL)


class HybridRetriever:
    def __init__(
        self,
        collection_name: str,
        qdrant_client: QdrantClient,
        bm25_index_path: str | Path | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._client = qdrant_client
        self._embedder = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")
        self._reranker = CrossEncoder(_RERANKER_MODEL)

        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[dict[str, Any]] = []

        if bm25_index_path and Path(bm25_index_path).exists():
            self._load_bm25(bm25_index_path)

    def build_bm25_from_qdrant(self, tenant_id: str | None = None) -> None:
        all_chunks: list[dict[str, Any]] = []
        offset = None

        scroll_filter = None
        if tenant_id:
            scroll_filter = Filter(
                must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
            )

        while True:
            results, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_chunks.extend([r.payload for r in results])
            if next_offset is None:
                break
            offset = next_offset

        tokenized = [self._tokenize(c.get("content", "")) for c in all_chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus = all_chunks

    def save_bm25(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "corpus": self._bm25_corpus}, f)

    def _load_bm25(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._bm25_corpus = data["corpus"]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        return tokens

    def _dense_search(
        self, query: str, tenant_id: str, top_k: int
    ) -> list[tuple[str, float, dict]]:
        query_vec = list(self._embedder.embed([query]))[0].tolist()

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            query_filter=Filter(
                must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
            ),
            limit=top_k,
            with_payload=True,
        )
        return [(r.id, r.score, r.payload) for r in results]

    def _bm25_search(
        self, query: str, tenant_id: str, top_k: int
    ) -> list[tuple[str, float, dict]]:
        if self._bm25 is None:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        scored_chunks = [
            (self._bm25_corpus[i].get("chunk_id", str(i)), scores[i], self._bm25_corpus[i])
            for i in range(len(scores))
            if self._bm25_corpus[i].get("tenant_id") == tenant_id
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    @staticmethod
    def _rrf_fusion(
        dense_results: list[tuple],
        bm25_results: list[tuple],
        k: int = _RRF_K,
    ) -> list[tuple[str, float, dict]]:
        rrf_scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for rank, (chunk_id, _score, payload) in enumerate(dense_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            payloads[chunk_id] = payload

        for rank, (chunk_id, _score, payload) in enumerate(bm25_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            payloads[chunk_id] = payload

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_id, score, payloads[chunk_id]) for chunk_id, score in sorted_results]

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k_per_source: int = 20,
        final_top_n: int = 5,
    ) -> list[RetrievedChunk]:
        dense_results = self._dense_search(query, tenant_id, top_k_per_source)
        bm25_results = self._bm25_search(query, tenant_id, top_k_per_source)

        fused = self._rrf_fusion(dense_results, bm25_results)

        rerank_candidates = fused[: final_top_n * 3]
        if not rerank_candidates:
            return []

        pairs = [(query, c[2].get("content", "")) for c in rerank_candidates]
        rerank_scores: list[float] = self._reranker.predict(pairs).tolist()

        scored = sorted(
            zip(rerank_candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for (chunk_id, rrf_score, payload), rerank_score in scored[:final_top_n]:
            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    content=payload.get("content", ""),
                    source=payload.get("source", ""),
                    title=payload.get("title", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    total_chunks=payload.get("total_chunks", 1),
                    tenant_id=payload.get("tenant_id", tenant_id),
                    rrf_score=rrf_score,
                    rerank_score=rerank_score,
                )
            )
        return results