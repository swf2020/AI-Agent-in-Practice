"""
A 股数据适配层：将 AKShare / 东方财富数据格式统一为
TradingAgents 内部期望的标准格式。

标准格式要求（来自 TradingAgents 源码 tradingagents/dataflows/）：
- OHLCV: Open, High, Low, Close, Volume（列名大写）
- 日期索引：datetime 类型，时区为 UTC
- 股票代码：以 ticker 参数传入（如 "600519"）
"""
from __future__ import annotations

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AStockAdapter:
    """
    将 AKShare 数据格式适配为 TradingAgents 标准格式。

    用法示例：
        adapter = AStockAdapter()
        df = adapter.get_price_history("600519", days=90)
    """

    # AKShare 返回字段 → TradingAgents 标准字段映射
    # 东方财富接口返回中文列名，需要转换
    COLUMN_MAP = {
        "日期":   "Date",
        "开盘":   "Open",
        "最高":   "High",
        "最低":   "Low",
        "收盘":   "Close",
        "成交量": "Volume",
        "成交额": "Amount",   # TradingAgents 可选字段
        "振幅":   "Amplitude",
        "涨跌幅": "Change",
        "涨跌额": "ChangeAmt",
        "换手率": "Turnover",
    }

    def get_price_history(
        self,
        ticker: str,
        days: int = 90,
        adjust: str = "qfq",  # 前复权；"hfq"=后复权；""=不复权
    ) -> pd.DataFrame:
        """
        获取 A 股历史行情，返回 TradingAgents 兼容格式。

        Args:
            ticker: A 股代码，如 "600519"（贵州茅台）
            days: 获取最近 N 个交易日数据
            adjust: 复权方式，分析建议使用前复权 "qfq"

        Returns:
            标准化 OHLCV DataFrame，日期为索引

        Raises:
            ValueError: 股票代码不存在或无数据时
        """
        end_date = datetime.now().strftime("%Y%m%d")
        # 多取 50 天缓冲，因为 days 是交易日而不是自然日
        start_date = (
            datetime.now() - timedelta(days=days + 50)
        ).strftime("%Y%m%d")

        try:
            df = ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception as e:
            raise ValueError(
                f"无法获取 {ticker} 数据，请确认股票代码正确且市场开放。原始错误：{e}"
            ) from e

        if df.empty:
            raise ValueError(f"股票 {ticker} 在指定日期范围内无交易数据")

        # 字段映射
        df = df.rename(columns=self.COLUMN_MAP)

        # 日期索引标准化
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df.index = df.index.tz_localize("Asia/Shanghai").tz_convert("UTC")

        # 保留核心字段（TradingAgents 只需要 OHLCV）
        core_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[core_cols].astype(float)

        # 截取实际需要的天数（AKShare 返回数据可能超出范围）
        return df.tail(days)

    def get_fundamental_info(self, ticker: str) -> dict:
        """
        获取 A 股基本面信息，用于替换 TradingAgents 的 FinnHub 基本面工具。

        Returns:
            与 FinnHub /stock/profile2 字段对齐的字典
        """
        try:
            # 东方财富实时行情（含市值、PE、PB 等）
            info = ak.stock_individual_info_em(symbol=ticker)
            # info 是两列 DataFrame：["item", "value"]
            info_dict = dict(zip(info["item"], info["value"]))
        except Exception as e:
            logger.warning(f"获取 {ticker} 基本面信息失败：{e}")
            return {}

        # 映射到 TradingAgents 期望的字段名
        return {
            "name":        info_dict.get("股票简称", ticker),
            "ticker":      ticker,
            "exchange":    info_dict.get("所处交所", "SSE/SZSE"),
            "marketCap":   info_dict.get("总市值", 0),
            "pe":          info_dict.get("市盈率(动态)", None),
            "pb":          info_dict.get("市净率", None),
            "52weekHigh":  info_dict.get("52周最高", None),
            "52weekLow":   info_dict.get("52周最低", None),
            "currency":    "CNY",
            "country":     "CN",
        }

    def get_news(self, ticker: str, limit: int = 20) -> list[dict]:
        """
        获取 A 股新闻，替换 TradingAgents 的 NewsAPI 工具。

        Returns:
            与 TradingAgents News Tool 期望格式对齐的新闻列表
        """
        try:
            news_df = ak.stock_news_em(symbol=ticker)
        except Exception as e:
            logger.warning(f"获取 {ticker} 新闻失败：{e}")
            return []

        results = []
        for _, row in news_df.head(limit).iterrows():
            results.append({
                "headline": row.get("新闻标题", ""),
                "summary":  row.get("新闻内容", "")[:500],  # 截断避免超 Token
                "datetime": str(row.get("发布时间", "")),
                "source":   row.get("新闻来源", "东方财富"),
                "url":      row.get("新闻链接", ""),
            })

        return results