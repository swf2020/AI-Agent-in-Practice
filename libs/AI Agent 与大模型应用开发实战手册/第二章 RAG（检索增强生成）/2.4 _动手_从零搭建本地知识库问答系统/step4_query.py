"""
RAG 查询链路：问题向量化 → 相似检索 → Prompt 构建 → LLM 生成
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from step3_index import COLLECTION_NAME, EMBED_MODEL_NAME, QDRANT_PATH

load_dotenv()

# BGE 查询专用前缀（文档向量化时不加，查询时必须加）
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    metadata: dict


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    """完整的 RAG 问答流水线"""

    def __init__(self, top_k: int = 5, score_threshold: float = 0.5):
        """
        Args:
            top_k: 检索返回的最多 Chunk 数
            score_threshold: 相似度阈值，低于此分数的结果丢弃，
                             防止将不相关内容注入 Prompt 产生幻觉
        """
        self.top_k = top_k
        self.score_threshold = score_threshold

        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.qdrant = QdrantClient(path=QDRANT_PATH)
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),  # Ollama 不需要真实 key
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """向量化问题并检索最相关的 Chunk"""
        # 加 BGE 查询前缀后向量化
        query_with_prefix = f"{BGE_QUERY_PREFIX}{question}"
        query_vector = self.embed_model.encode(
            query_with_prefix,
            normalize_embeddings=True,
        ).tolist()

        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=self.top_k,
            score_threshold=self.score_threshold,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                text=r.payload["text"],
                source=r.payload["source"],
                score=r.score,
                metadata={k: v for k, v in r.payload.items() if k not in {"text", "source"}},
            )
            for r in results
        ]

    def build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """
        构建 RAG Prompt。

        设计原则：
        1. 明确告诉模型只能基于提供的上下文作答
        2. 无答案时主动说"不知道"，而非编造
        3. 每段上下文标注来源，方便生成带引用的回答
        """
        if not chunks:
            return f"""你是一个知识库问答助手。

用户问题：{question}

很抱歉，在知识库中没有找到与该问题相关的内容，请直接告知用户你无法回答该问题。"""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_name = chunk.source.split("/")[-1]  # 取文件名
            context_parts.append(f"【参考资料 {i}】来源：{source_name}\n{chunk.text}")

        context = "\n\n---\n\n".join(context_parts)

        return f"""你是一个知识库问答助手，请严格根据以下参考资料回答用户问题。

规则：
- 只能基于参考资料中的内容作答，不得凭空编造
- 如果参考资料中没有足够信息，请直接说"根据现有资料无法回答该问题"
- 回答时注明信息来自哪份参考资料（如：根据参考资料 1...）

参考资料：
{context}

用户问题：{question}

请用中文回答："""

    def ask(self, question: str) -> RAGAnswer:
        """端到端问答：检索 → 构建 Prompt → 生成回答"""
        chunks = self.retrieve(question)
        prompt = self.build_prompt(question, chunks)

        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # 问答场景用低温度，减少随机性
            max_tokens=1024,
        )

        return RAGAnswer(
            question=question,
            answer=response.choices[0].message.content,
            sources=chunks,
        )


if __name__ == "__main__":
    pipeline = RAGPipeline(top_k=5, score_threshold=0.5)
    question = "pathlib 如何读取文件内容？"
    result = pipeline.ask(question)

    print(f"问题：{result.question}\n")
    print(f"回答：\n{result.answer}\n")
    print(f"引用来源（{len(result.sources)} 条）：")
    for c in result.sources:
        print(f"  [{c.score:.3f}] {c.source}")