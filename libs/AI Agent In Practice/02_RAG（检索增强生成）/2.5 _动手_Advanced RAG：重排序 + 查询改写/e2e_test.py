"""端到端验证：对比 Naive RAG vs Advanced RAG 效果。"""

import time
from advanced_rag import AdvancedRAG, AdvancedRAGConfig
from baseline_rag import NaiveRAG

# ── 准备测试语料（模拟企业知识库切块）──────────────────────────────────────
DOCS = [
    "向量数据库 Qdrant 支持 HNSW 索引，查询延迟在毫秒级别，适合大规模相似度检索。",
    "BM25 是经典的稀疏检索算法，基于词频和逆文档频率，擅长精确关键词匹配。",
    "混合检索结合稠密向量检索和稀疏 BM25 检索，通过 RRF 融合提升召回率。",
    "RAG 的核心挑战是检索精度，向量检索可能因语义漂移返回不相关文档。",
    "Cross-Encoder 重排序通过联合编码 Query 和文档来计算精确相关性分数，准确度高于 Bi-Encoder。",
    "HyDE（Hypothetical Document Embeddings）通过 LLM 生成假设答案文档来改善查询表示。",
    "LLMLingua 利用小语言模型压缩 Prompt，可将上下文 Token 减少 50% 同时保留关键信息。",
    "BGE-Reranker 是 BAAI 发布的双语重排序模型，在 BEIR 基准上表现优异。",
    "Naive RAG 直接将检索结果喂给 LLM，没有重排和压缩，容易引入噪声。",
    "查询改写（Query Rewriting）通过生成多角度查询来覆盖更广泛的语义空间。",
]

TEST_QUESTIONS = [
    "为什么 Cross-Encoder 的重排效果比向量检索更准确？",
    "HyDE 和 Multi-Query 分别是如何改进检索的？",
    "如何在 RAG 中控制输入 LLM 的 Token 数量？",
]


def run_experiment() -> None:
    print("=" * 60)
    print("🚀 初始化 Naive RAG 基线...")
    naive = NaiveRAG()
    naive.index_documents(DOCS)

    print("\n🚀 初始化 Advanced RAG（仅开启重排+改写，关闭压缩）...")
    cfg = AdvancedRAGConfig(
        use_hyde=True,
        use_multi_query=True,
        use_reranker=True,
        use_compression=False,  # 首次验证先关闭，减少调用
        retrieval_top_k=10,
        reranker_top_n=3,
    )
    advanced = AdvancedRAG(config=cfg)
    advanced.index_documents(DOCS)  # 共用同一份文档

    print("\n" + "=" * 60)
    for q in TEST_QUESTIONS:
        print(f"\n📌 问题：{q}")
        print("-" * 40)

        # Naive RAG
        t0 = time.time()
        naive_result = naive.query(q, top_k=3)
        naive_time = time.time() - t0

        # Advanced RAG
        t0 = time.time()
        adv_result = advanced.advanced_query(q)
        adv_time = time.time() - t0

        print(f"[Naive  RAG | {naive_time:.2f}s] {naive_result['answer'][:120]}...")
        print(f"[Advanced   | {adv_time:.2f}s] {adv_result['answer'][:120]}...")
        print(f"  优化步骤：{' → '.join(adv_result['steps'])}")
        print(f"  使用查询数：{len(adv_result['queries_used'])}")

        # [Fix #14] 添加基本质量断言，确保 Advanced RAG 正常运行
        assert len(adv_result["answer"]) > 0, "Advanced RAG 未生成回答"
        assert len(adv_result["steps"]) > 0, "Advanced RAG 优化步骤为空"
        assert len(adv_result["queries_used"]) > 0, "Advanced RAG 使用查询为空"

    print("\n" + "=" * 60)
    print("✅ 所有断言通过，Naive vs Advanced RAG 对比完成。")


if __name__ == "__main__":
    run_experiment()