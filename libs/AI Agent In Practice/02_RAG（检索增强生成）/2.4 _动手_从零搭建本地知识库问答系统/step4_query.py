"""
RAG 查询链路：问题向量化 → 相似检索 → Prompt 构建 → LLM 生成
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from step3_index import COLLECTION_NAME, EMBED_MODEL_NAME, QDRANT_PATH
from core_config import MODEL_REGISTRY

load_dotenv()

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


def get_llm_client(model_key: str = "DeepSeek-V3"):
    """获取 LLM 客户端，返回 (litellm, model_id, api_key, api_base)

    不再污染全局 os.environ，改为显式返回 api_key 和 api_base，
    由调用方在 completion() 调用时显式传入。
    """  # [Fix #2]
    import litellm

    if model_key not in MODEL_REGISTRY:
        model_key = "DeepSeek-V3"

    cfg = MODEL_REGISTRY[model_key]
    api_key_env = cfg.get("api_key_env")
    base_url = cfg.get("base_url")

    api_key = os.environ.get(api_key_env) if api_key_env else None

    return litellm, cfg["litellm_id"], api_key, base_url  # [Fix #2] 显式返回，不污染环境变量


class RAGPipeline:
    def __init__(self, top_k: int = 5, score_threshold: float = 0.5, model_key: str = "DeepSeek-V3"):
        self.top_k = top_k
        self.score_threshold = score_threshold

        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.qdrant = QdrantClient(path=QDRANT_PATH)

        self.model_key = model_key  # [Fix #10] 保存 model_key 供异常处理使用
        self.llm_client, self.model_name, self.api_key, self.api_base = get_llm_client(model_key)  # [Fix #2, #10]
        print(f"✅ LLM 模型：{model_key} ({self.model_name})")

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """向量化问题并检索最相关的 Chunk"""
        query_with_prefix = f"{BGE_QUERY_PREFIX}{question}"
        query_vector = self.embed_model.encode(
            query_with_prefix,
            normalize_embeddings=True,
        ).tolist()

        # 首先尝试使用索引查询
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=self.top_k,
            score_threshold=self.score_threshold,
            with_payload=True,
        )

        # 如果索引查询没有返回结果，使用全量扫描作为备选
        if len(results.points) == 0:
            print("⚠️ 索引查询无结果，尝试全量扫描...")
            return self._retrieve_full_scan(query_vector)

        return [
            RetrievedChunk(
                text=r.payload["text"],
                source=r.payload["source"],
                score=r.score,
                metadata={k: v for k, v in r.payload.items() if k not in {"text", "source"}},
            )
            for r in results.points
        ]

    def _retrieve_full_scan(self, query_vector: list[float]) -> list[RetrievedChunk]:
        """全量扫描检索（当索引未就绪时使用）

        使用分页 scroll 遍历所有数据点，避免大数据量遗漏。
        通过 FULL_SCAN_LIMIT 环境变量控制安全上限。
        """  # [Fix #4]
        import numpy as np

        full_scan_limit = int(os.getenv("RAG_FULL_SCAN_LIMIT", "5000"))  # [Fix #4]

        query_np = np.array(query_vector)
        all_scores: list[tuple[float, int, dict]] = []  # (score, id, payload)

        offset = None
        total_scanned = 0
        while total_scanned < full_scan_limit:
            batch = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            points, next_offset = batch[0], batch[1]
            if not points:
                break

            for point in points:
                if point.vector and point.payload:
                    vector_np = np.array(point.vector)
                    score = float(
                        np.dot(query_np, vector_np)
                        / (np.linalg.norm(query_np) * np.linalg.norm(vector_np))
                    )
                    if score >= self.score_threshold:
                        all_scores.append((score, point.id, point.payload))

            total_scanned += len(points)
            offset = next_offset
            if next_offset is None:
                break

        # 按相似度排序并取 top_k
        all_scores.sort(key=lambda x: x[0], reverse=True)
        top_results = all_scores[: self.top_k]

        return [
            RetrievedChunk(
                text=r[2]["text"],
                source=r[2]["source"],
                score=r[0],
                metadata={k: v for k, v in r[2].items() if k not in {"text", "source"}},
            )
            for r in top_results
        ]

    def build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """构建 RAG Prompt"""
        if not chunks:
            return f"""你是一个知识库问答助手。

用户问题：{question}

很抱歉，在知识库中没有找到与该问题相关的内容，请直接告知用户你无法回答该问题。"""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_name = chunk.source.split("/")[-1]
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
        """端到端问答：检索 → 构建 Prompt → LLM 生成"""  # [Fix #2, #7, #10]
        chunks = self.retrieve(question)
        prompt = self.build_prompt(question, chunks)

        # 显式传入 api_key 和 api_base，避免依赖环境变量
        completion_kwargs: dict = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        try:
            response = self.llm_client.completion(**completion_kwargs)  # [Fix #7]
        except self.llm_client.exceptions.AuthenticationError:
            return RAGAnswer(
                question=question,
                answer=(
                    "❌ API Key 无效，请检查：\n"
                    "   1. 是否已将正确的 Key 填入 .env 文件\n"
                    f"   2. 当前模型 {self.model_key} 对应的环境变量是否正确设置"
                ),
                sources=chunks,
            )
        except self.llm_client.exceptions.RateLimitError:
            print("⚠️  触发速率限制，等待 60 秒后重试...")
            time.sleep(60)
            response = self.llm_client.completion(**completion_kwargs)
        except Exception as e:
            return RAGAnswer(
                question=question,
                answer=f"❌ LLM 调用失败：{type(e).__name__}: {e}\n请检查网络连接和 API 配额。",
                sources=chunks,
            )

        return RAGAnswer(
            question=question,
            answer=response.choices[0].message.content,
            sources=chunks,
        )


if __name__ == "__main__":
    import sys

    model_key = os.getenv("RAG_MODEL", "DeepSeek-V3")
    if len(sys.argv) > 1:
        model_key = sys.argv[1]

    # 降低阈值以获取更多结果
    pipeline = RAGPipeline(top_k=5, score_threshold=0.4, model_key=model_key)
    question = "pathlib 的层次结构是什么样的？"
    result = pipeline.ask(question)

    print(f"问题：{result.question}\n")
    print(f"回答：\n{result.answer}\n")
    print(f"引用来源（{len(result.sources)} 条）：")
    for c in result.sources:
        print(f"  [{c.score:.3f}] {c.source}")

    # 显式关闭 Qdrant 连接，避免程序退出时触发析构函数错误
    pipeline.qdrant.close()
    