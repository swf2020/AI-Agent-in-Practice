"""Reranker 模块：使用 DashScope qwen3-rerank API 进行精排。"""

import httpx
from typing import TYPE_CHECKING

from core_config import (
    get_dashscope_api_key, get_reranker_model,
    get_reranker_top_n,
)

if TYPE_CHECKING:
    from baseline_rag import RetrievedChunk


class DashScopeReranker:
    """
    使用 DashScope qwen3-rerank 对候选文档精排。

    优势：
    - 无需加载本地模型，减少显存/内存占用
    - qwen3-rerank 是阿里云最新发布的中文重排序模型
    - 通过 OpenAI 兼容接口调用，首次运行即可使用

    模型选型说明：
    - qwen3-rerank：中文场景推荐，效果好，无需本地 GPU
    - bge-reranker-v2-m3（本地）：如需离线使用可切换回本地方案
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        top_n: int | None = None,
    ) -> None:
        self.model = model or get_reranker_model()
        self.api_key = api_key or get_dashscope_api_key()
        self.top_n = top_n if top_n is not None else get_reranker_top_n()
        # DashScope 原生 rerank API 端点（非 OpenAI 兼容）
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        print(f"✅ Reranker 已配置：{self.model}")

    def rerank(
        self,
        query: str,
        chunks: list["RetrievedChunk"],
        top_n: int | None = None,
    ) -> list["RetrievedChunk"]:
        """
        对候选切块重新打分并排序。

        Args:
            query: 用户原始查询（不是改写后的查询）
            chunks: 向量检索的候选切块（通常 Top-20）
            top_n: 精排后保留的数量（送入 LLM 的上下文）

        Returns:
            按分值降序排列的 Top-N 切块，score 字段更新为 reranker 分值。
        """
        if not chunks:
            return []

        n = top_n if top_n is not None else self.top_n
        documents = [chunk.text for chunk in chunks]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
            },
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            # DashScope 返回格式: output.results[].relevance_score
            results = data.get("output", {}).get("results", [])
            # results 是无序的，需要按 index 映射回原始文档顺序
            score_map = {}
            for item in results:
                idx = item.get("index", 0)
                score = item.get("relevance_score", 0)
                score_map[idx] = score

            scores = [score_map.get(i, 0) for i in range(len(chunks))]

        except Exception as e:
            # 降级方案：回退到向量检索原始分数排序
            print(f"⚠️ Reranker API 调用失败，回退到向量检索分数排序: {e}")
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:n]

        # [Fix #10] 创建新 RetrievedChunk 对象，避免污染调用方原始数据
        reranked = [
            RetrievedChunk(
                text=chunk.text,
                score=score,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
            )
            for chunk, score in zip(chunks, scores)
        ]

        # 按分值降序，取 Top-N
        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked[:n]
