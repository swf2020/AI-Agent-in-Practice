from __future__ import annotations

from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from chunker import DocumentChunk
from core_config import EMBED_MODEL as _EMBED_MODEL, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL, VECTOR_DIM as _VECTOR_DIM


class VectorIndexer:
    def __init__(
        self,
        collection_name: str | None = None,
        qdrant_url: str | None = None,
    ) -> None:
        self.collection_name = collection_name or QDRANT_COLLECTION
        self._client = QdrantClient(
            url=qdrant_url or QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        self._embedder = TextEmbedding(model_name=_EMBED_MODEL)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=_VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )
            for field_name in ("tenant_id", "file_hash", "doc_type"):
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema="keyword",
                )

    def _is_already_indexed(self, file_hash: str, tenant_id: str) -> bool:
        results, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="file_hash", match=MatchValue(value=file_hash)),
                    FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0

    def _delete_by_hash(self, file_hash: str, tenant_id: str) -> None:
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="file_hash", match=MatchValue(value=file_hash)),
                    FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                ]
            ),
        )

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 64,
        force_reindex: bool = False,
    ) -> dict[str, int]:
        if not chunks:
            return {"indexed": 0, "skipped": 0}

        file_hash = chunks[0].file_hash
        tenant_id = chunks[0].tenant_id

        if not force_reindex and self._is_already_indexed(file_hash, tenant_id):
            return {"indexed": 0, "skipped": len(chunks)}

        stats = {"indexed": 0, "skipped": 0}

        for i in tqdm(range(0, len(chunks), batch_size), desc="索引中", unit="batch"):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            embeddings = list(self._embedder.embed(texts))

            points = [
                PointStruct(
                    id=chunk.chunk_id,
                    vector=emb.tolist(),
                    payload=self._chunk_to_payload(chunk),
                )
                for chunk, emb in zip(batch, embeddings)
            ]

            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            stats["indexed"] += len(batch)

        return stats

    @staticmethod
    def _chunk_to_payload(chunk: DocumentChunk) -> dict[str, Any]:
        return {
            "content": chunk.content,
            "source": chunk.source,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "file_hash": chunk.file_hash,
            "tenant_id": chunk.tenant_id,
            "doc_type": chunk.doc_type,
            **chunk.extra_metadata,
        }