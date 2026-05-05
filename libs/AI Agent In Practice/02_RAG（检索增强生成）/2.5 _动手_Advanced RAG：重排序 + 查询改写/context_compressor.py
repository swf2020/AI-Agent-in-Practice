"""上下文压缩模块：LLMLingua 与 LLM 摘要压缩两种实现。"""

from openai import OpenAI
from core_config import get_chat_model_id
from baseline_rag import RetrievedChunk


class LLMLinguaCompressor:
    """
    基于 LLMLingua 的 Token 级别压缩。
    需要额外安装：uv pip install llmlingua==0.2.2
    """

    def __init__(self, rate: float = 0.5) -> None:
        """
        Args:
            rate: 压缩率，0.5 表示保留 50% 的 Token。
                  实验建议：0.4~0.6 之间，低于 0.3 可能损失关键信息。
        """
        from llmlingua import PromptCompressor

        # 使用轻量级模型做 Token 打分，不参与最终生成
        # 国内可替换为 Qwen/Qwen1.5-1.8B-Chat 等本地模型
        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
        )
        self.rate = rate

    def compress(self, query: str, context: str) -> str:
        """
        压缩上下文文本，保留与 query 最相关的 Token。

        LLMLingua2 使用 query-aware 压缩：
        会优先保留与 query 语义相关的 Token。
        """
        result = self.compressor.compress_prompt(
            context,
            rate=self.rate,
            question=query,
        )
        compressed = result["compressed_prompt"]
        original_tokens = result["origin_tokens"]
        compressed_tokens = result["compressed_tokens"]

        ratio = 1 - compressed_tokens / max(original_tokens, 1)
        print(
            f"📉 LLMLingua 压缩：{original_tokens} → {compressed_tokens} tokens "
            f"（节省 {ratio:.1%}）"
        )
        return compressed


class SummaryCompressor:
    """
    基于 LLM 摘要的上下文压缩。
    实现最简单，适合快速集成，无需额外加载本地模型。
    """

    def __init__(self, client: OpenAI, model: str | None = None) -> None:
        self.client = client
        self.model = model or get_chat_model_id()

    def compress(
        self,
        query: str,
        chunks: list["RetrievedChunk"],
        max_words: int = 300,
    ) -> str:
        """
        让 LLM 对多个检索切块做摘要，只保留与 query 相关的内容。

        注意：这会引入额外的 LLM 调用。适合最终上下文需要控制在
        1000 Token 以内的场景，不适合实时对话（延迟会增加 500ms+）。
        """
        context = "\n\n---\n\n".join(
            f"[切块 {i+1}]\n{c.text}" for i, c in enumerate(chunks)
        )
        prompt = (
            f"以下是检索到的多段文本，请提取与问题直接相关的信息，"
            f"压缩为不超过 {max_words} 字的摘要，保留具体数字和关键术语：\n\n"
            f"问题：{query}\n\n文本：\n{context}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()