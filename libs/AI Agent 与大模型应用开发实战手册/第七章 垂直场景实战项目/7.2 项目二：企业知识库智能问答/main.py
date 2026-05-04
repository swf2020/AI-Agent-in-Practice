from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from core_config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
from generator import AnswerGenerator, GeneratedAnswer
from retriever import HybridRetriever

load_dotenv()

_retriever: HybridRetriever | None = None
_generator: AnswerGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _generator

    client = QdrantClient(
        url=os.getenv("QDRANT_URL", QDRANT_URL),
        api_key=os.getenv("QDRANT_API_KEY", QDRANT_API_KEY),
    )
    collection = os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION)

    _retriever = HybridRetriever(
        collection_name=collection,
        qdrant_client=client,
        bm25_index_path=Path("bm25_index.pkl") if Path("bm25_index.pkl").exists() else None,
    )

    if _retriever._bm25 is None:
        print("构建 BM25 索引中...")
        _retriever.build_bm25_from_qdrant()
        _retriever.save_bm25("bm25_index.pkl")

    _generator = AnswerGenerator()
    print("服务就绪 ✓")
    yield
    client.close()


app = FastAPI(
    title="企业知识库问答 API",
    description="支持多租户权限隔离、混合检索、角注引用的企业知识库 Q&A 服务",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="用户问题")
    tenant_id: str = Field(..., min_length=1, description="租户 ID")
    top_n: int = Field(default=5, ge=1, le=10, description="最终返回的检索块数量")


class QueryResponse(BaseModel):
    answer: str
    references: list[dict]
    is_abstained: bool
    top_rerank_score: float


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest) -> JSONResponse:
    if _retriever is None or _generator is None:
        raise HTTPException(status_code=503, detail="服务未就绪，请稍后重试")

    chunks = _retriever.retrieve(
        query=request.query,
        tenant_id=request.tenant_id,
        final_top_n=request.top_n,
    )

    result: GeneratedAnswer = _generator.generate(
        query=request.query,
        chunks=chunks,
    )

    return JSONResponse(
        content={
            "answer": result.answer,
            "references": result.references,
            "is_abstained": result.is_abstained,
            "top_rerank_score": round(result.top_rerank_score, 4),
        }
    )


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "bm25_ready": _retriever is not None and _retriever._bm25 is not None}