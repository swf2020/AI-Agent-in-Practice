"""Naive RAG 基线实现，作为对比实验的起点。"""

from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
from core_config import (
    get_dashscope_api_key, get_embedding_base_url,
    get_embedding_model, get_embedding_dim,
    get_chat_model_id, get_api_key, get_base_url,
)

load_dotenv()


@dataclass
class RetrievedChunk:
    """检索结果的数据结构，贯穿整条链路。"""
    text: str
    score: float
    chunk_id: str
    source: str = ""


class NaiveRAG:
    """
    Naive RAG 基线：Embedding → 向量检索 → 直接生成。
    不做任何查询改写、重排或压缩。
    """

    COLLECTION = "naive_rag_demo"

    def __init__(self) -> None:
        # Embedding 客户端（DashScope，text-embedding-v4）
        self.embed_client = OpenAI(
            api_key=get_dashscope_api_key(),
            base_url=get_embedding_base_url(),
        )
        # 聊天模型客户端（通过 core_config 统一配置）
        self.chat_client = OpenAI(
            api_key=get_api_key(),
            base_url=get_base_url(),
        )
        # 使用内存模式的 Qdrant，无需启动独立服务
        self.qdrant = QdrantClient(":memory:")
        self._init_collection()

    def _init_collection(self) -> None:
        """初始化向量集合（幂等操作）。"""
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.COLLECTION not in existing:
            self.qdrant.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(
                    size=get_embedding_dim(), distance=Distance.COSINE
                ),
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量 Embedding，使用配置的 Embedding 模型。"""
        resp = self.embed_client.embeddings.create(
            model=get_embedding_model(), input=texts
        )
        return [item.embedding for item in resp.data]

    def index_documents(self, chunks: list[str]) -> None:
        """将文档切块写入向量库。"""
        embeddings = self.embed(chunks)
        points = [
            PointStruct(
                id=i,
                vector=emb,
                payload={"text": text, "chunk_id": str(i)},
            )
            for i, (text, emb) in enumerate(zip(chunks, embeddings))
        ]
        self.qdrant.upsert(collection_name=self.COLLECTION, points=points)
        print(f"已索引 {len(chunks)} 个切块")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """向量检索，返回 Top-K 候选。"""
        q_vec = self.embed([query])[0]
        response = self.qdrant.query_points(
            collection_name=self.COLLECTION,
            query=q_vec,
            limit=top_k,
        )
        return [
            RetrievedChunk(
                text=h.payload["text"],
                score=h.score,
                chunk_id=h.payload["chunk_id"],
            )
            for h in response.points
        ]

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> str:
        """基于检索结果生成答案。"""
        context = "\n\n---\n\n".join(c.text for c in context_chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个精准的问答助手。请严格基于以下上下文回答问题，"
                    "如果上下文中没有相关信息，直接说'不知道'，不要编造。"
                ),
            },
            {
                "role": "user",
                "content": f"上下文：\n{context}\n\n问题：{query}",
            },
        ]
        resp = self.chat_client.chat.completions.create(
            model=get_chat_model_id(),
            messages=messages,
            temperature=0.1,
        )
        return resp.choices[0].message.content

    def query(self, question: str, top_k: int = 5) -> dict:
        """完整的 RAG 查询链路，返回答案和检索到的切块。"""
        chunks = self.retrieve(question, top_k=top_k)
        answer = self.generate(question, chunks)
        return {"answer": answer, "retrieved_chunks": chunks}
