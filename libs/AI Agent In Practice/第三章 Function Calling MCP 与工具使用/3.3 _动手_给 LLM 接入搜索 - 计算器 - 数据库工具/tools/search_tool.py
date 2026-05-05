"""Tavily 实时搜索工具"""

from __future__ import annotations

import os
from typing import Any

from tavily import TavilyClient

from tools.base import BaseTool


class TavilySearchTool(BaseTool):
    """接入 Tavily API 的实时搜索工具。

    适用场景：
    - 需要最新信息（股价、新闻、近期事件）
    - 需要引用来源 URL 的场景
    - 替代 LLM 可能过时的训练知识
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        search_depth: str = "basic",  # "basic" 或 "advanced"，advanced 消耗更多 quota
    ) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if self._api_key:
            self._client = TavilyClient(api_key=self._api_key)
        else:
            self._client = None
        self._max_results = max_results
        self._search_depth = search_depth

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "搜索互联网上的实时信息。当用户询问近期事件、"
                    "当前价格、最新新闻、或任何可能超出你知识截止日期的信息时使用。"
                    "不要用于可以直接回答的通用知识问题。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询词，使用清晰具体的关键词，避免自然语言长句",
                        },
                        "topic": {
                            "type": "string",
                            "enum": ["general", "news", "finance"],
                            "description": "搜索类型：general=通用，news=新闻，finance=金融",
                            "default": "general",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def run(self, query: str, topic: str = "general") -> str:
        """执行搜索并返回格式化结果。"""
        if not self._client:
            return "⚠️ 搜索功能不可用：未配置 TAVILY_API_KEY 环境变量。请在 https://tavily.com 获取 API Key 后使用。"
        
        response = self._client.search(
            query=query,
            topic=topic,
            max_results=self._max_results,
            search_depth=self._search_depth,
            include_answer=True,  # Tavily 会额外提供一个 AI 摘要答案
        )

        # 构建 LLM 友好的文本格式：摘要 + 各结果条目
        parts: list[str] = []

        if response.get("answer"):
            parts.append(f"【摘要】{response['answer']}\n")

        for i, result in enumerate(response.get("results", []), 1):
            parts.append(
                f"[{i}] {result['title']}\n"
                f"URL: {result['url']}\n"
                f"内容: {result['content'][:400]}...\n"  # 截断避免 context 爆炸
            )

        return "\n".join(parts) if parts else "未找到相关搜索结果"
