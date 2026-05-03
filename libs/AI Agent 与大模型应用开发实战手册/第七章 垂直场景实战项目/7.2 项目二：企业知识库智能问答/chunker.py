from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_parser import ParsedDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    source: str
    title: str
    chunk_index: int
    total_chunks: int
    file_hash: str
    tenant_id: str
    doc_type: str
    extra_metadata: dict = field(default_factory=dict)


_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", "！", "？", " ", ""],
    length_function=len,
)


def chunk_document(doc: ParsedDocument) -> list[DocumentChunk]:
    raw_chunks: list[str] = _SPLITTER.split_text(doc.content)
    total = len(raw_chunks)

    chunks: list[DocumentChunk] = []
    for idx, text in enumerate(raw_chunks):
        chunk = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            content=text,
            source=doc.source,
            title=doc.title,
            chunk_index=idx,
            total_chunks=total,
            file_hash=doc.file_hash,
            tenant_id=doc.metadata.get("tenant_id", "default"),
            doc_type=doc.metadata.get("doc_type", "unknown"),
            extra_metadata={
                k: v for k, v in doc.metadata.items()
                if k not in ("tenant_id", "doc_type")
            },
        )
        chunks.append(chunk)

    return chunks