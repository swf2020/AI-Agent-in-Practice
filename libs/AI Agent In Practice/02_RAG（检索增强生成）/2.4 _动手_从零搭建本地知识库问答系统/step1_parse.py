"""
文档解析模块：支持 PDF / Word / 网页 → 结构化文本
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import pypdf
from markitdown import MarkItDown


@dataclass
class ParsedDocument:
    """解析后的文档，保留来源信息供后续引用溯源"""
    content: str                          # 纯文本内容（Markdown 格式）
    source: str                           # 文件路径或 URL
    doc_type: Literal["pdf", "word", "web"]
    metadata: dict = field(default_factory=dict)


def parse_pdf(path: str | Path) -> ParsedDocument:
    """
    解析 PDF 文档。

    策略：优先用 pypdf 提取文字层；若文字层为空（扫描件），
    退化提示用户使用 OCR 工具（如 marker-pdf）预处理。
    """
    path = Path(path)
    reader = pypdf.PdfReader(str(path))
    pages: list[str] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            # 添加页码标记，方便后续引用溯源
            pages.append(f"<!-- Page {i + 1} -->\n{text}")

    if not pages:
        raise ValueError(
            f"{path.name} 可能是扫描版 PDF，pypdf 无法提取文字层。"
            "建议用 marker-pdf 或 pymupdf 进行 OCR 预处理。"
        )

    content = "\n\n".join(pages)
    return ParsedDocument(
        content=content,
        source=str(path),
        doc_type="pdf",
        metadata={"page_count": len(reader.pages), "filename": path.name},
    )


def parse_word_or_office(path: str | Path) -> ParsedDocument:
    """
    解析 Word / Excel / PPT 等 Office 格式。

    使用 MarkItDown 统一转为 Markdown，保留标题层级结构，
    这对后续章节切块策略至关重要。
    """
    path = Path(path)
    md = MarkItDown()
    result = md.convert(str(path))
    return ParsedDocument(
        content=result.text_content,
        source=str(path),
        doc_type="word",
        metadata={"filename": path.name},
    )


def parse_webpage(url: str) -> ParsedDocument:
    """
    解析网页内容。

    MarkItDown 内部使用 trafilatura 抽取正文，
    自动过滤导航栏、广告等噪声区域。
    """
    md = MarkItDown()
    result = md.convert_url(url)
    return ParsedDocument(
        content=result.text_content,
        source=url,
        doc_type="web",
        metadata={"url": url},
    )


def parse_document(source: str) -> ParsedDocument:
    """统一入口：根据来源自动分发到对应解析器"""
    if source.startswith("http://") or source.startswith("https://"):
        return parse_webpage(source)

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(path)
    elif suffix in {".docx", ".xlsx", ".pptx", ".doc"}:
        return parse_word_or_office(path)
    else:
        raise ValueError(f"不支持的文件格式：{suffix}")


# ── 快速验证 ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else "https://docs.python.org/3/library/pathlib.html"
    doc = parse_document(source)
    print(f"来源：{doc.source}")
    print(f"类型：{doc.doc_type}")
    print(f"字符数：{len(doc.content)}")
    print(f"前 300 字：\n{doc.content[:300]}")