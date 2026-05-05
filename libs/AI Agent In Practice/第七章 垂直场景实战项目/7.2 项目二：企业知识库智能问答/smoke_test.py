from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from chunker import chunk_document
from document_parser import DocumentParser
from generator import AnswerGenerator
from indexer import VectorIndexer
from retriever import HybridRetriever

load_dotenv()


def run_smoke_test():
    print("=== 企业知识库智能问答 - 冒烟测试 ===")
    
    test_content = """# 公司请假政策

## 年假规定
员工每年享有15天带薪年假。入职不满一年的员工按比例计算。

## 病假规定
病假需提供医院证明，每月最多可请3天带薪病假。

## 审批流程
请假申请需提前3天提交，由直属主管审批。"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(test_content)
        test_file = f.name
    
    try:
        print("\n1. 文档解析测试...")
        parser = DocumentParser()
        parsed_doc = parser.parse(test_file, tenant_id="test_tenant")
        print(f"   ✓ 解析成功：{parsed_doc.title}")
        
        print("\n2. 文档分块测试...")
        chunks = chunk_document(parsed_doc)
        print(f"   ✓ 分块完成：{len(chunks)} 个块")
        
        print("\n3. 向量索引测试...")
        indexer = VectorIndexer(collection_name="test_kb")
        stats = indexer.index_chunks(chunks, force_reindex=True)
        print(f"   ✓ 索引完成：{stats}")
        
        print("\n4. 混合检索测试...")
        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        retriever = HybridRetriever(collection_name="test_kb", qdrant_client=client)
        retriever.build_bm25_from_qdrant(tenant_id="test_tenant")
        
        results = retriever.retrieve("年假有多少天", tenant_id="test_tenant")
        print(f"   ✓ 检索完成：{len(results)} 条结果")
        
        print("\n5. 回答生成测试...")
        generator = AnswerGenerator()
        answer = generator.generate("年假有多少天", results)
        print(f"   ✓ 生成完成")
        print(f"   答案：{answer.answer}")
        print(f"   引用数：{len(answer.references)}")
        
        print("\n=== 测试完成 ✓ ===")
        
    finally:
        os.unlink(test_file)
        if "client" in locals():
            client.delete_collection(collection_name="test_kb")


if __name__ == "__main__":
    run_smoke_test()