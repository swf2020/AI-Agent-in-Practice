from __future__ import annotations

import os
from dataclasses import dataclass, field

from openai import OpenAI

from core_config import CONFIDENCE_THRESHOLD as _CONFIDENCE_THRESHOLD, get_api_key, get_base_url, get_litellm_id
from retriever import RetrievedChunk

_ABSTAIN_RESPONSE = (
    "抱歉，根据现有文档库，我无法找到与您问题相关的可靠信息。"
    "请尝试换种方式提问，或联系知识库管理员补充相关文档。"
)


@dataclass
class GeneratedAnswer:
    answer: str
    references: list[dict] = field(default_factory=list)
    is_abstained: bool = False
    top_rerank_score: float = float("-inf")


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] 来源：《{chunk.title}》（{chunk.source}）\n{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)


def _build_prompt(query: str, context: str) -> str:
    system_prompt = """你是一个严谨的企业知识库助手。请严格遵守以下规则：

1. **仅基于提供的参考文档**回答问题，不得引用参考文档以外的知识。
2. 在回答中用角注格式标注引用来源，例如"根据公司政策[1]，..."。
   - 角注编号对应参考文档列表中的编号。
   - 同一句话可引用多个来源，如 [1][3]。
3. 如果参考文档中没有足够信息回答问题，请直接说"根据现有文档，我无法回答此问题"，不得编造。
4. 回答语言与用户问题语言保持一致。
5. 回答简洁准确，避免复述用户问题。"""

    user_content = f"""参考文档：

{context}

---

用户问题：{query}"""

    return system_prompt, user_content


class AnswerGenerator:
    def __init__(
        self,
        model: str | None = None,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    ) -> None:
        self._client = OpenAI(
            api_key=get_api_key(),
            base_url=get_base_url(),
        )
        self._model = model or get_litellm_id()
        self._threshold = confidence_threshold

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        max_tokens: int = 1024,
    ) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                answer=_ABSTAIN_RESPONSE,
                is_abstained=True,
                top_rerank_score=float("-inf"),
            )

        top_score = chunks[0].rerank_score

        if top_score < self._threshold:
            return GeneratedAnswer(
                answer=_ABSTAIN_RESPONSE,
                is_abstained=True,
                top_rerank_score=top_score,
            )

        valid_chunks = [c for c in chunks if c.rerank_score >= self._threshold]
        context = _build_context(valid_chunks)
        system_prompt, user_content = _build_prompt(query, context)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )

        answer_text = response.choices[0].message.content or ""

        references = [
            {
                "index": i + 1,
                "title": chunk.title,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "rerank_score": round(chunk.rerank_score, 4),
            }
            for i, chunk in enumerate(valid_chunks)
        ]

        return GeneratedAnswer(
            answer=answer_text,
            references=references,
            is_abstained=False,
            top_rerank_score=top_score,
        )