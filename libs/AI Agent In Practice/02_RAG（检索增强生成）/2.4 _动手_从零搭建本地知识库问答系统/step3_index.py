"""
向量化 + Qdrant 索引构建
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterator

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from step2_chunk import Chunk, chunk_document
from step1_parse import parse_document

# ── 配置 ──────────────────────────────────────────────────
COLLECTION_NAME = "local_kb"
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # 512 维，中文效果好，约 90MB
VECTOR_DIM = 512
QDRANT_PATH = "./qdrant_storage"               # 本地持久化目录
BATCH_SIZE = 64                                # 批量向量化大小


def get_embed_model() -> SentenceTransformer:
    """
    加载 Embedding 模型（首次运行会从 HuggingFace 下载）。

    为什么选 BGE-small-zh-v1.5？
    - 512 维向量，索引体积小，检索快
    - 中文 MTEB 排名靠前，性价比最高
    - 支持查询前缀：查询时加 "为这个句子生成表示以用于检索相关文章："
      文档时不加前缀，这是 BGE 的特殊使用约定。
    """
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_qdrant_client() -> QdrantClient:
    """获取本地持久化 Qdrant 客户端"""
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=QDRANT_PATH)


def ensure_collection(client: QdrantClient) -> None:
    """确保 Collection 存在，不存在则创建"""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"已创建 Collection：{COLLECTION_NAME}")
        # 创建后立即优化索引
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config={"indexing_threshold": 1},
        )
        print("已设置索引阈值")
    else:
        print(f"Collection 已存在：{COLLECTION_NAME}")


def batch_iter(items: list, size: int) -> Iterator[list]:
    """将列表按批次切分的生成器"""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def index_chunks(chunks: list[Chunk], model: SentenceTransformer, client: QdrantClient) -> int:
    """
    批量向量化并写入 Qdrant。

    返回写入的 Chunk 总数。
    """
    ensure_collection(client)

    # 获取当前最大 ID，避免覆盖已有数据
    count_result = client.count(collection_name=COLLECTION_NAME)
    start_id = count_result.count

    total = 0
    for batch in batch_iter(chunks, BATCH_SIZE):
        texts = [c.text for c in batch]

        # BGE 文档向量化：不加查询前缀
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        points = [
            PointStruct(
                # [Fix #2] 使用 source+chunk_index 的 hash 作为 ID，避免多文档索引时 ID 冲突
                id=_chunk_id(chunk.source, chunk.chunk_index),
                vector=vector.tolist(),
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "strategy": chunk.strategy,
                    **chunk.metadata,
                },
            )
            for i, (chunk, vector) in enumerate(zip(batch, vectors))
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        start_id += len(batch)
        total += len(batch)
        print(f"  已写入 {total}/{len(chunks)} 块")

    return total


def _chunk_id(source: str, chunk_index: int) -> int:
    """根据来源和块索引生成稳定的唯一 ID（12 位十六进制转整数）

    [Fix #2] 避免多文档索引或重复索引时 ID 冲突
    """
    h = hashlib.md5(f"{source}:{chunk_index}".encode()).hexdigest()[:12]
    return int(h, 16)


def index_document(source: str, strategy: str = "section") -> None:
    """完整索引一个文档的便捷函数"""
    print(f"\n📄 处理文档：{source}")

    t0 = time.time()
    doc = parse_document(source)
    print(f"  解析完成：{len(doc.content)} 字符，耗时 {time.time()-t0:.1f}s")

    t1 = time.time()
    chunks = chunk_document(doc, strategy=strategy)
    print(f"  切块完成：{len(chunks)} 块，耗时 {time.time()-t1:.2f}s")

    t2 = time.time()
    model = get_embed_model()
    client = get_qdrant_client()
    count = index_chunks(chunks, model, client)
    print(f"  索引完成：写入 {count} 块，耗时 {time.time()-t2:.1f}s")
    print(f"  总耗时：{time.time()-t0:.1f}s ✅")


if __name__ == "__main__":
    import sys
    sources = sys.argv[1:] or ["https://docs.python.org/3/library/pathlib.html"]
    for src in sources:
        index_document(src)