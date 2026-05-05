"""查询改写模块：HyDE 与 Multi-Query 双策略。"""

import json
from openai import OpenAI
from core_config import get_chat_model_id
from baseline_rag import RetrievedChunk


class QueryRewriter:
    """
    封装 HyDE 和 Multi-Query 两种改写策略。
    设计原则：策略可单独使用，也可组合使用。
    """

    def __init__(self, client: OpenAI, model: str | None = None) -> None:
        self.client = client
        self.model = model or get_chat_model_id()

    def hyde(self, query: str) -> str:
        """
        HyDE：生成一篇假设性的"理想答案文档"。

        为何 HyDE 有效：
        向量模型训练时见过大量文档，对"文档风格"的 Embedding 比对"问题风格"
        的 Embedding 更稳定。用假设文档去检索，相当于把查询从问题空间
        映射到了文档空间。

        局限：HyDE 依赖 LLM 生成质量，若 LLM 对领域不熟悉，
        假设文档可能引入噪声，反而拉低召回质量。
        """
        prompt = (
            f"请为以下问题撰写一段简洁的参考答案（100字以内），"
            f"语气像专业文档，不必完全准确，只需覆盖关键概念：\n\n问题：{query}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # 稍高温度增加多样性，但不宜过高
        )
        return resp.choices[0].message.content.strip()

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        """
        Multi-Query：生成 N 个语义相近但措辞不同的查询。

        返回的列表包含原始查询（第一个），确保原始意图不丢失。
        """
        prompt = (
            f"请将以下问题改写为 {n} 个语义相近但措辞不同的查询，"
            f"以 JSON 数组格式返回，每个元素是一个字符串，不要加任何解释：\n\n"
            f"原始问题：{query}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        try:
            raw = json.loads(resp.choices[0].message.content)
            # 兼容模型返回 {"queries": [...]} 或直接 [...] 两种格式
            variants: list[str] = (
                raw if isinstance(raw, list)
                else next(iter(raw.values()))
            )
        except (json.JSONDecodeError, StopIteration):
            variants = []

        # 原始查询始终放第一位，保证召回的下界
        return [query] + [v for v in variants if v != query]

    def rrf_merge(
        self,
        results_list: list[list["RetrievedChunk"]],  # 多路检索结果
        k: int = 60,
        top_n: int = 20,
    ) -> list["RetrievedChunk"]:
        """
        Reciprocal Rank Fusion（RRF）：将多路检索结果融合去重。

        RRF 公式：score(d) = Σ 1 / (k + rank_i(d))
        k=60 是 Cormack et al. 2009 论文推荐的默认值，实践中无需调整。

        优势：不依赖各路检索的分值量纲，天然适合混合不同类型的检索器。
        """
        from collections import defaultdict

        chunk_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, "RetrievedChunk"] = {}

        for results in results_list:
            for rank, chunk in enumerate(results, start=1):
                chunk_scores[chunk.chunk_id] += 1.0 / (k + rank)
                chunk_map[chunk.chunk_id] = chunk

        merged = sorted(
            chunk_map.values(),
            key=lambda c: chunk_scores[c.chunk_id],
            reverse=True,
        )
        return merged[:top_n]