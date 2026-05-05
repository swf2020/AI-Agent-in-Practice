from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from markitdown import MarkItDown


DocumentType = Literal["pdf", "docx", "pptx", "html", "url", "txt", "unknown"]


@dataclass
class ParsedDocument:
    content: str
    source: str
    doc_type: DocumentType
    file_hash: str
    title: str = ""
    metadata: dict = field(default_factory=dict)


def _compute_hash(source: str | Path) -> str:
    if isinstance(source, Path) and source.exists():
        sha256 = hashlib.sha256()
        with open(source, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    return hashlib.sha256(str(source).encode()).hexdigest()


def _detect_type(source: str | Path) -> DocumentType:
    s = str(source).lower()
    if s.startswith("http://") or s.startswith("https://"):
        return "url"
    suffix = Path(s).suffix.lstrip(".")
    return suffix if suffix in ("pdf", "docx", "pptx", "html", "txt") else "unknown"


def _extract_title(source: str | Path, content: str) -> str:
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    if isinstance(source, Path):
        return source.stem.replace("_", " ").replace("-", " ").title()
    path = urlparse(str(source)).path.rstrip("/")
    return path.split("/")[-1] or str(source)


class DocumentParser:
    def __init__(self) -> None:
        self._md = MarkItDown()

    def parse(
        self,
        source: str | Path,
        tenant_id: str = "default",
        extra_metadata: dict | None = None,
    ) -> ParsedDocument:
        source = Path(source) if not str(source).startswith("http") else str(source)
        doc_type = _detect_type(source)
        file_hash = _compute_hash(source)

        result = self._md.convert(str(source))
        content: str = result.text_content or ""
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        title = _extract_title(source, content)

        metadata: dict = {
            "tenant_id": tenant_id,
            "doc_type": doc_type,
            **(extra_metadata or {}),
        }

        return ParsedDocument(
            content=content,
            source=str(source),
            doc_type=doc_type,
            file_hash=file_hash,
            title=title,
            metadata=metadata,
        )

    def parse_batch(
        self,
        sources: list[str | Path],
        tenant_id: str = "default",
        extra_metadata: dict | None = None,
    ) -> list[ParsedDocument]:
        import logging
        results = []
        for src in sources:
            try:
                results.append(self.parse(src, tenant_id, extra_metadata))
            except Exception as exc:
                logging.warning(f"解析失败 {src}: {exc}")
        return results