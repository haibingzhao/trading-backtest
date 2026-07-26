"""市场规则感知层。

提供做空许可查询和市场分类，基于标的代码格式自动检测。
内联正则模式（复用 _market_hooks.py 的模式），使策略框架不依赖引擎包。
"""

from __future__ import annotations

import re


# 市场分类正则模式（与 backtest/engines/_market_hooks.py 保持同步）
_MARKET_PATTERNS = [
    (re.compile(r"^\d{6}\.(SZ|SH|BJ)$", re.I), "a_share"),
    (re.compile(r"^(51|15|56)\d{4}\.(SZ|SH)$", re.I), "a_share"),
    (re.compile(r"^[A-Z]+\.US$", re.I), "us_equity"),
    (re.compile(r"^\d{3,5}\.HK$", re.I), "hk_equity"),
    (re.compile(r"^[A-Z]+-USDT$", re.I), "crypto"),
    (re.compile(r"^[A-Z]+/USDT$", re.I), "crypto"),
    (re.compile(r"^[A-Za-z]{1,2}\d{3,4}\.(ZCE|DCE|SHFE|INE|CFFEX|GFEX)$", re.I), "futures"),
    (re.compile(r"^[A-Z]{2,4}[FGHJKMNQUVXZ]\d{1,2}$", re.I), "futures"),
    (re.compile(r"^[A-Z]{2,4}\d{4}$", re.I), "futures"),
    (re.compile(r"^[A-Z]{2,4}\.(CME|CBOT|NYMEX|COMEX|ICE|EUREX)$", re.I), "futures"),
    (re.compile(r"^[A-Z]{3}/[A-Z]{3}$"), "forex"),
    (re.compile(r"^[A-Z]{6}\.FX$"), "forex"),
]


class MarketRules:
    """市场规则查询工具。

    基于标的代码格式自动检测市场类型，判断是否允许做空。

    做空限制:
    - A 股 (a_share): 禁止做空
    - 港股 / 美股 / 加密货币 / 期货 / 外汇: 允许做空
    """

    SHORT_BLOCKED = frozenset({"a_share"})
    SHORT_ALLOWED = frozenset({"us_equity", "hk_equity", "crypto", "forex", "futures"})

    @staticmethod
    def detect_market(code: str) -> str:
        """检测标的所属市场类型。

        Args:
            code: 标的代码（如 '600000.SH', '09988.HK', 'BABA.US'）。

        Returns:
            市场类型字符串: a_share / us_equity / hk_equity / crypto / futures / forex。
            未匹配时默认返回 'a_share'。
        """
        for pattern, market in _MARKET_PATTERNS:
            if pattern.match(code):
                return market
        return "a_share"

    @classmethod
    def can_short(cls, code: str) -> bool:
        """判断标的是否允许做空（基于市场规则）。

        Args:
            code: 标的代码。

        Returns:
            True 如果市场规则允许做空。
        """
        market = cls.detect_market(code)
        return market not in cls.SHORT_BLOCKED
