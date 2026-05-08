"""
两种切块策略的实现与对比：
1. 固定大小切块（按 Token 数）
2. 章节切块（利用 Markdown 标题层级）

注：语义切块（按句子边界 + 相似度聚合）效果最好但耗时显著，
教学版暂不收录，留作扩展练习。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from step1_parse import ParsedDocument


@dataclass
class Chunk:
    """单个文本片段，携带溯源信息"""
    text: str
    source: str
    chunk_index: int
    strategy: str
    metadata: dict


# ── 策略 1：固定大小切块 ────────────────────────────────────
def chunk_fixed_size(
    doc: ParsedDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    按 Token 数做固定大小切块。

    为什么用 Token 而非字符数？
    因为 Embedding 模型的输入限制是 Token，用字符数切块
    在中文环境下会因字符/Token 比例不固定而导致截断。

    chunk_overlap 保证跨块的句子不会完全割裂语义。
    """
    enc = tiktoken.get_encoding("cl100k_base")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len(enc.encode(text)),  # 按 Token 计长度
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )

    texts = splitter.split_text(doc.content)
    return [
        Chunk(
            text=t,
            source=doc.source,
            chunk_index=i,
            strategy="fixed",
            metadata={**doc.metadata, "token_count": len(enc.encode(t))},
        )
        for i, t in enumerate(texts)
    ]


# ── 策略 2：章节切块（推荐用于结构化文档）──────────────────
def chunk_by_section(
    doc: ParsedDocument,
    max_chunk_size: int = 1000,
) -> list[Chunk]:
    """
    按 Markdown 标题层级切块。

    适用场景：Word/PPT 文档、技术文档、API 参考。
    这类文档的章节边界本身就是语义边界，按标题切比按字符切
    更能保证每个 Chunk 内容的语义完整性。

    超过 max_chunk_size 的节会用固定大小策略二次切分。
    """
    enc = tiktoken.get_encoding("cl100k_base")

    # 匹配所有级别的 Markdown 标题（# ~ ######）
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(doc.content))

    if not matches:
        # 没有检测到标题结构，退化为固定大小切块
        return chunk_fixed_size(doc, chunk_size=max_chunk_size)

    # 按标题边界切分文本
    sections: list[tuple[str, str]] = []  # (标题, 正文)
    for i, match in enumerate(matches):
        heading = match.group(0)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc.content)
        body = doc.content[start:end].strip()
        sections.append((heading, body))

    chunks: list[Chunk] = []
    for heading, body in sections:
        section_text = f"{heading}\n\n{body}"
        token_count = len(enc.encode(section_text))

        if token_count <= max_chunk_size:
            chunks.append(
                Chunk(
                    text=section_text,
                    source=doc.source,
                    chunk_index=len(chunks),
                    strategy="section",
                    metadata={**doc.metadata, "heading": heading.strip()},
                )
            )
        else:
            # 超长章节二次切分，保留标题作为前缀
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=50,
                length_function=lambda t: len(enc.encode(t)),
            )
            for sub_text in sub_splitter.split_text(body):
                chunks.append(
                    Chunk(
                        text=f"{heading}\n\n{sub_text}",
                        source=doc.source,
                        chunk_index=len(chunks),
                        strategy="section+fixed",
                        metadata={**doc.metadata, "heading": heading.strip()},
                    )
                )

    return chunks


# ── 策略选择入口 ────────────────────────────────────────────
def chunk_document(
    doc: ParsedDocument,
    strategy: Literal["fixed", "section"] = "section",
) -> list[Chunk]:
    """
    根据文档类型自动选择最优切块策略：
    - Word/PPT 等结构化文档 → section（利用标题层级）
    - PDF/网页等非结构化文档 → fixed（按 Token 均匀切分）
    """
    if strategy == "section" or doc.doc_type == "word":
        return chunk_by_section(doc)
    return chunk_fixed_size(doc)


# ── 对比实验 ──────────────────────────────────────────────
if __name__ == "__main__":
    from step1_parse import parse_document

    # 用网页做演示，结构相对清晰
    doc = parse_document("https://docs.python.org/3/library/pathlib.html")
    fixed_chunks = chunk_fixed_size(doc, chunk_size=512)
    section_chunks = chunk_by_section(doc, max_chunk_size=1000)

    print(f"固定大小切块：{len(fixed_chunks)} 块，平均长度 {sum(len(c.text) for c in fixed_chunks)//len(fixed_chunks)} 字符")
    print(f"章节切块：    {len(section_chunks)} 块，平均长度 {sum(len(c.text) for c in section_chunks)//len(section_chunks)} 字符")
    print(f"\n章节切块示例（第 1 块）：\n{section_chunks[0].text[:300]}")