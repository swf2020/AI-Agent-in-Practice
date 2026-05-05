"""主入口：演示 Advanced RAG 查询流程。"""

from advanced_rag import AdvancedRAG, AdvancedRAGConfig
from baseline_rag import NaiveRAG

# ── 示例文档 ───────────────────────────────────────────────────────────────────
DOCS = [
    "向量数据库 Qdrant 支持 HNSW 索引，查询延迟在毫秒级别，适合大规模相似度检索。",
    "BM25 是经典的稀疏检索算法，基于词频和逆文档频率，擅长精确关键词匹配。",
    "混合检索结合稠密向量检索和稀疏 BM25 检索，通过 RRF 融合提升召回率。",
    "Cross-Encoder 重排序通过联合编码 Query 和文档来计算精确相关性分数。",
    "HyDE 通过 LLM 生成假设答案文档来改善查询表示。",
    "查询改写通过生成多角度查询来覆盖更广泛的语义空间。",
]


def main() -> None:
    print("=" * 60)
    print("Advanced RAG 演示（重排序 + 查询改写）")
    print("=" * 60)

    # 初始化 Advanced RAG
    cfg = AdvancedRAGConfig(
        use_hyde=True,
        use_multi_query=True,
        multi_query_n=2,
        use_reranker=True,
        use_compression=True,
        retrieval_top_k=10,
        reranker_top_n=3,
    )
    rag = AdvancedRAG(config=cfg)
    rag.index_documents(DOCS)

    # 演示查询
    question = "为什么 Cross-Encoder 的重排效果比向量检索更准确？"
    print(f"\n问题：{question}")
    result = rag.advanced_query(question)
    print(f"答案：{result['answer']}")
    print(f"优化步骤：{' → '.join(result['steps'])}")


if __name__ == "__main__":
    main()
