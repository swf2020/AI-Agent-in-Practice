"""Advanced RAG 完整流水线，整合查询改写、重排序与上下文压缩。"""

from dataclasses import dataclass, field
from baseline_rag import NaiveRAG, RetrievedChunk
from query_rewriter import QueryRewriter
from reranker import DashScopeReranker
from context_compressor import SummaryCompressor


@dataclass
class AdvancedRAGConfig:
    """Advanced RAG 配置，每个优化环节可独立开关，便于消融实验。"""

    # 查询改写配置
    use_hyde: bool = True
    use_multi_query: bool = True
    multi_query_n: int = 3          # Multi-Query 生成几个变体

    # 检索配置
    retrieval_top_k: int = 20       # 粗排候选数，给重排留充足空间

    # 重排配置
    use_reranker: bool = True
    reranker_top_n: int = 5         # 精排后送入 LLM 的切块数

    # 压缩配置
    use_compression: bool = False   # 默认关闭，延迟敏感场景不建议开启
    compression_max_words: int = 300


class AdvancedRAG(NaiveRAG):
    """
    继承 NaiveRAG，叠加高级检索能力。
    继承而非组合，是为了复用索引与 Embedding 逻辑，
    避免重复代码，同时保持 Naive RAG 的基线可对比性。
    """

    def __init__(self, config: AdvancedRAGConfig | None = None) -> None:
        super().__init__()
        self.config = config or AdvancedRAGConfig()
        self.rewriter = QueryRewriter(self.chat_client)
        self.reranker = DashScopeReranker() if self.config.use_reranker else None
        self.compressor = SummaryCompressor(self.chat_client)

    def _multi_retrieve(self, queries: list[str]) -> list[RetrievedChunk]:
        """
        对多个查询分别检索，用 RRF 融合结果。
        每路检索独立执行，后续 RRF 去重。
        """
        all_results: list[list[RetrievedChunk]] = []
        for q in queries:
            results = self.retrieve(q, top_k=self.config.retrieval_top_k)
            all_results.append(results)

        if len(all_results) == 1:
            return all_results[0]

        return self.rewriter.rrf_merge(
            all_results,
            top_n=self.config.retrieval_top_k,
        )

    def advanced_query(self, question: str) -> dict:
        """
        Advanced RAG 完整查询链路：
        查询改写 → 多路检索 → RRF 融合 → 重排 → 压缩 → 生成
        """
        cfg = self.config
        queries = [question]
        steps_log: list[str] = []

        # ① 查询改写
        if cfg.use_multi_query:
            queries = self.rewriter.multi_query(question, n=cfg.multi_query_n)
            steps_log.append(f"Multi-Query 生成 {len(queries)} 路查询")

        if cfg.use_hyde:
            hyde_doc = self.rewriter.hyde(question)
            # HyDE 生成的假设文档作为额外的检索查询加入
            queries.append(hyde_doc)
            steps_log.append("HyDE 生成假设文档")

        # ② 多路检索 + RRF 融合
        candidates = self._multi_retrieve(queries)
        steps_log.append(f"RRF 融合后候选数：{len(candidates)}")

        # ③ Cross-Encoder 重排
        if cfg.use_reranker and self.reranker:
            candidates = self.reranker.rerank(
                question, candidates, top_n=cfg.reranker_top_n
            )
            steps_log.append(f"重排后保留 Top-{cfg.reranker_top_n}")
        else:
            candidates = candidates[: cfg.reranker_top_n]

        # ④ 上下文压缩（可选）
        if cfg.use_compression and candidates:
            compressed_ctx = self.compressor.compress(
                question, candidates, max_words=cfg.compression_max_words
            )
            # 将压缩结果包装为单个 chunk，保持 generate 接口统一
            candidates = [
                RetrievedChunk(
                    text=compressed_ctx, score=1.0, chunk_id="compressed"
                )
            ]
            steps_log.append("上下文压缩完成")

        # ⑤ LLM 生成
        answer = self.generate(question, candidates)

        return {
            "answer": answer,
            "retrieved_chunks": candidates,
            "steps": steps_log,
            "queries_used": queries,
        }