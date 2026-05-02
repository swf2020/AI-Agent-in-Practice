"""BGE-Reranker Cross-Encoder 精排模块。"""

from sentence_transformers import CrossEncoder


class BGEReranker:
    """
    使用 BAAI/bge-reranker-v2-m3 对候选文档精排。

    模型选型说明：
    - bge-reranker-base（约 280MB）：速度快，适合延迟敏感场景
    - bge-reranker-v2-m3（约 570MB）：多语言，中文效果更好，推荐首选
    - bge-reranker-large（约 1.3GB）：精度最高，GPU 推理才有性价比

    首次运行会自动从 HuggingFace 下载模型，国内可设置镜像：
    export HF_ENDPOINT=https://hf-mirror.com
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        # device=None 时 sentence-transformers 自动选择 CUDA/MPS/CPU
        self.model = CrossEncoder(model_name, max_length=512)
        print(f"✅ Reranker 已加载：{model_name}")

    def rerank(
        self,
        query: str,
        chunks: list["RetrievedChunk"],
        top_n: int = 5,
    ) -> list["RetrievedChunk"]:
        """
        对候选切块重新打分并排序。

        Args:
            query: 用户原始查询（不是改写后的查询）
            chunks: 向量检索的候选切块（通常 Top-20）
            top_n: 精排后保留的数量（送入 LLM 的上下文）

        Returns:
            按相关性降序排列的 Top-N 切块，score 字段更新为 Cross-Encoder 分值。
        """
        if not chunks:
            return []

        # Cross-Encoder 的输入是 (query, document) 对
        pairs = [(query, chunk.text) for chunk in chunks]
        scores: list[float] = self.model.predict(pairs).tolist()

        # 将 Cross-Encoder 分值回写到 RetrievedChunk
        for chunk, score in zip(chunks, scores):
            chunk.score = score

        # 按分值降序，取 Top-N
        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return reranked[:top_n]