"""端到端冒烟测试：一键验证全流程"""
from __future__ import annotations


def main() -> None:
    print("=" * 50)
    print("Step 1：解析文档")
    from step1_parse import parse_document
    doc = parse_document("https://docs.python.org/3/library/pathlib.html")
    assert len(doc.content) > 100, "文档解析失败：内容为空"
    print(f"  ✅ 解析成功，字符数：{len(doc.content)}")

    print("\nStep 2：切块")
    from step2_chunk import chunk_document
    chunks = chunk_document(doc, strategy="section")
    assert len(chunks) > 0, "切块失败：无切块结果"
    print(f"  ✅ 切块成功，共 {len(chunks)} 块")

    print("\nStep 3：向量化 & 索引")
    from step3_index import get_embed_model, get_qdrant_client, index_chunks
    model = get_embed_model()
    client = get_qdrant_client()
    # [Fix #9] 使用 min() 防止切块不足 5 块时 IndexError
    test_count = min(len(chunks), 5)
    count = index_chunks(chunks[:test_count], model, client)
    assert count == test_count, f"索引写入数量异常：预期 {test_count}，实际 {count}"
    print(f"  ✅ 索引成功，写入 {count} 块")

    print("\nStep 4：问答查询")
    from step4_query import RAGPipeline
    pipeline = RAGPipeline(top_k=3, score_threshold=0.3)
    result = pipeline.ask("pathlib 是什么？")
    assert len(result.answer) > 10, "LLM 回答异常：内容过短"
    print("  ✅ 问答成功")
    print(f"  问题：{result.question}")
    print(f"  回答：{result.answer[:200]}...")
    print(f"  引用：{len(result.sources)} 条")

    print("\n" + "=" * 50)
    print("✅ 所有步骤通过！RAG 系统运行正常。")


if __name__ == "__main__":
    main()