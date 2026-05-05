"""
Token 消耗与成本追踪器。
线程安全（使用 dataclass + 字典，单线程 asyncio 环境下无需加锁）。
"""
from dataclasses import dataclass
from collections import defaultdict
import litellm


@dataclass
class ModelUsage:
    """单个模型的累计用量"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostTracker:
    """
    按 (feature, model) 两个维度追踪 LLM 调用成本。

    使用方式：
        tracker = CostTracker()
        tracker.record(response, feature="rag_query")
        print(tracker.report())
    """

    def __init__(self) -> None:
        # defaultdict 避免每次判断 key 是否存在
        self._usage: dict[tuple[str, str], ModelUsage] = defaultdict(ModelUsage)

    def record(self, response: litellm.ModelResponse, feature: str = "default") -> None:
        """
        从 LiteLLM 响应对象中提取用量并累计。

        Args:
            response: LiteLLM 返回的 ModelResponse 对象
            feature:  业务功能标识，如 "rag_query"、"summarize"
        """
        usage = response.usage
        model = response.model or "unknown"
        key = (feature, model)

        self._usage[key].prompt_tokens += usage.prompt_tokens
        self._usage[key].completion_tokens += usage.completion_tokens

        # litellm.completion_cost 根据模型名查表计算费用
        # 支持 500+ 模型的定价，每周从官方更新
        try:
            cost = litellm.completion_cost(completion_response=response)
            self._usage[key].total_cost_usd += cost
        except Exception:
            # 部分自托管模型无定价数据，静默跳过
            pass

    def report(self) -> dict:
        """返回结构化消费报告，便于序列化为 JSON 打日志"""
        result = {}
        for (feature, model), usage in self._usage.items():
            result.setdefault(feature, {})[model] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": round(usage.total_cost_usd, 6),
            }
        return result

    def total_cost(self) -> float:
        """返回所有维度的总成本（美元）"""
        return sum(u.total_cost_usd for u in self._usage.values())

    def reset(self) -> None:
        """重置统计（通常在测试或定时汇报后调用）"""
        self._usage.clear()