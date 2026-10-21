"""Futu OpenAPI-backed loader for HK, US and China A-share OHLCV data."""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders.base import (
    NoAvailableSourceError,
    loader_cache_get,
    loader_cache_put,
    validate_date_range,
)
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

#: canonical interval token → ``futu.KLType`` 成员名。
#:
#: ⚠️ **与 ``tools/futunn/endpoints.py::KTYPE_BY_UNIT`` 必须逐键一致**
#: （框架内有一致性测试钉住二者；改一侧必须同步另一侧）。
#: 保留本表而不是直接 import 框架，是为了让本 loader 在**没有** ``tools`` 包时仍可独立使用。
#: ⚠️ ``m`` = 分钟、``M`` = 月 —— **大小写敏感**，绝不整体 upper()/lower()。
_INTERVAL_MAP: dict[str, str] = {
    "1m": "K_1M", "3m": "K_3M", "5m": "K_5M", "10m": "K_10M", "15m": "K_15M",
    "30m": "K_30M", "1h": "K_60M", "2h": "K_120M", "3h": "K_180M", "4h": "K_240M",
    "1d": "K_DAY", "1w": "K_WEEK", "1M": "K_MON", "1y": "K_YEAR",
}

#: 逐 token 精确别名（同样不做大小写折叠）。
_INTERVAL_ALIASES: dict[str, str] = {
    "min": "1m", "day": "1d", "daily": "1d", "hour": "1h", "hourly": "1h",
    "week": "1w", "weekly": "1w", "month": "1M", "monthly": "1M",
    "year": "1y", "yearly": "1y",
    "D": "1d", "H": "1h", "W": "1w", "Y": "1y", "MO": "1M",
}

#: 分钟为单位的 60/120/180/240 等价于对应小时。
_INTERVAL_MINUTES_TO_HOURS: dict[int, str] = {60: "1h", 120: "2h", 180: "3h", 240: "4h"}

#: 单位词形 → 单位字母（只放词，不放单字母，避免把 ``m`` 误映射成 ``M``）。
_INTERVAL_UNIT_WORDS: dict[str, str] = {
    "min": "m", "mins": "m", "minute": "m", "minutes": "m",
    "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "day": "d", "days": "d", "week": "w", "weeks": "w",
    "month": "M", "months": "M", "mon": "M", "mo": "M", "mos": "M",
    "year": "y", "years": "y",
}

_INTERVAL_TOKEN_RE = re.compile(r"^(\d+)([A-Za-z]+)$")


def _normalize_interval_unit(unit: str) -> str:
    """单位写法 → 单位字母；``m`` / ``M`` **有意不折叠**（分钟 vs 月）。"""
    word = _INTERVAL_UNIT_WORDS.get(unit.lower())
    if word is not None:
        return word
    if unit in ("m", "M"):
        return unit
    return unit.lower()


def canonical_interval(interval: str) -> str:
    """任意写法 → canonical interval token（``day``/``1D`` → ``1d``；``60m`` → ``1h``）。

    Raises:
        ValueError: 空串、格式不对，或组合未登记。
    """
    key = (interval or "").strip()
    if not key:
        raise ValueError(f"interval 不能为空；合法取值：{'、'.join(sorted(_INTERVAL_MAP))}")
    if key in _INTERVAL_MAP:
        return key
    mapped = _INTERVAL_ALIASES.get(key)
    if mapped is not None:
        return mapped
    match = _INTERVAL_TOKEN_RE.match(key)
    if match is None:
        raise ValueError(
            f"无法识别 interval {interval!r}；合法取值：{'、'.join(sorted(_INTERVAL_MAP))}"
        )
    number = int(match.group(1))
    unit = _normalize_interval_unit(match.group(2))
    if unit == "m" and f"{number}m" not in _INTERVAL_MAP:
        hour_token = _INTERVAL_MINUTES_TO_HOURS.get(number)
        if hour_token is not None:
            return hour_token
    token = f"{number}{unit}"
    if token not in _INTERVAL_MAP:
        raise ValueError(
            f"无法识别 interval {interval!r}；合法取值：{'、'.join(sorted(_INTERVAL_MAP))}"
        )
    return token


def _to_futu_symbol(code: str) -> str:
    """Convert project symbol to Futu OpenAPI format.

    Examples:
        700.HK    -> HK.00700
        5.HK      -> HK.00005
        000001.SZ -> SZ.000001
        600519.SH -> SH.600519
        AAPL.US   -> US.AAPL
        AAPL      -> US.AAPL  (bare alpha tickers are treated as US)
    """
    upper = code.strip().upper()
    if upper.endswith(".HK"):
        return f"HK.{upper[:-3].zfill(5)}"
    if upper.endswith(".SZ"):
        return f"SZ.{upper[:-3].zfill(6)}"
    if upper.endswith(".SH"):
        return f"SH.{upper[:-3].zfill(6)}"
    if upper.endswith(".US"):
        return f"US.{upper[:-3]}"
    # Bare alpha ticker (e.g. AAPL) -> US market; numeric-only codes are
    # left untouched (ambiguous between A-share / HK without a suffix).
    if upper.isalpha():
        return f"US.{upper}"
    return upper


def _to_futu_ktype(interval: str):
    """Map project interval string to a futu KLType enum value.

    Lazy-imports futu so the module can be imported without futu installed.

    ⚠️ 未登记的 interval **抛 ``ValueError``**（此前是静默回落 ``K_DAY``：同一个
    ``1m`` 在本 loader 落到日线、在 REST loader 落到月线，两边都错且都不报错）。
    ⚠️ ``1M`` = 月、``1m`` = 1 分钟（大小写敏感，见 :data:`_INTERVAL_MAP`）。
    """
    from futu import KLType  # noqa: PLC0415
    return getattr(KLType, _INTERVAL_MAP[canonical_interval(interval)])


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a Futu kline DataFrame to the standard OHLCV schema.

    Args:
        df: Raw DataFrame returned by ``request_history_kline()``.

    Returns:
        DataFrame with columns [open, high, low, close, volume] indexed
        by ``trade_date`` (timezone-naive DatetimeIndex), sorted ascending.
    """
    if df.empty:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)

    result = df.copy()
    result.index = pd.to_datetime(result["time_key"])
    result.index.name = "trade_date"
    result = result[_OHLCV_COLUMNS].copy()
    result = result.apply(pd.to_numeric, errors="coerce")
    result["volume"] = result["volume"].fillna(0.0)
    result = result.dropna(subset=["open", "high", "low", "close"])
    return result.sort_index()


@register
class FutuLoader:
    """Fetch HK, US and China A-share bars from Futu OpenAPI.

    Requires FutuOpenD running locally (https://www.futunn.com/download/openAPI)
    and the environment variables ``FUTU_HOST`` / ``FUTU_PORT`` to be set.
    """

    name = "futu"
    markets = {"hk_equity", "a_share", "us_equity"}
    requires_auth = True

    def __init__(self) -> None:
        self._host = os.environ.get("FUTU_HOST", "127.0.0.1")
        self._port = int(os.environ.get("FUTU_PORT", "11111"))

    def is_available(self) -> bool:
        """Return True if env vars are set and FutuOpenD is reachable."""
        if not os.environ.get("FUTU_HOST") or not os.environ.get("FUTU_PORT"):
            return False
        try:
            import futu  # noqa: PLC0415
            ctx = futu.OpenQuoteContext(host=self._host, port=self._port)
            ctx.close()
            return True
        except Exception:
            return False

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV history from Futu OpenAPI.

        Args:
            codes: Project symbols such as ``700.HK`` or ``000001.SZ``.
            start_date: Start date in ``YYYY-MM-DD`` format.
            end_date: End date in ``YYYY-MM-DD`` format.
            interval: Backtest interval. Canonical set (case-sensitive):
                ``1m``/``3m``/``5m``/``10m``/``15m``/``30m``/``1h``/``2h``/
                ``3h``/``4h``/``1d``/``1w``/``1M``/``1y``, where ``1M`` means a
                month and ``1m`` one minute. Aliases like ``1D``/``60m``/
                ``daily`` are accepted. Unregistered values raise ``ValueError``
                instead of silently falling back to daily. Default ``1D``.
            fields: Ignored; included for interface compatibility.

        Returns:
            Mapping of input symbol to normalised OHLCV dataframe.

        Raises:
            ValueError: If ``interval`` is not a registered token (raised before
                any network I/O, so it is never reported as a connection error).
            NoAvailableSourceError: If FutuOpenD connection fails.
        """
        del fields
        if not codes:
            return {}
        validate_date_range(start_date, end_date)

        results: Dict[str, pd.DataFrame] = {}

        # Serve cached symbols first; only open a FutuOpenD connection when at
        # least one symbol is uncached, so a fully-cached request needs no
        # running gateway.
        pending: List[str] = []
        for code in codes:
            cached = loader_cache_get(
                source=self.name,
                symbol=code,
                timeframe=interval,
                start_date=start_date,
                end_date=end_date,
                fields=None,
            )
            if cached is not None:
                results[code] = cached.copy()
            else:
                pending.append(code)

        if not pending:
            return results

        # ⚠️ Parameter validation must run BEFORE the I/O block: inside it, a
        # bad ``interval`` gets wrapped into ``NoAvailableSourceError`` and is
        # then indistinguishable from "the gateway is offline" — two problems
        # whose debugging directions are opposite. (The REST loader has the
        # same ordering; see the R31 entry in the Futunn plan.)
        ktype = _to_futu_ktype(interval)

        try:
            import futu  # noqa: PLC0415
            ctx = futu.OpenQuoteContext(host=self._host, port=self._port)
        except Exception as exc:
            raise NoAvailableSourceError(
                f"Cannot connect to FutuOpenD at {self._host}:{self._port}: {exc}"
            ) from exc

        try:
            for code in pending:
                futu_code = _to_futu_symbol(code)
                # ⚠️ autype 必须**显式**下发：吃 SDK 缺省（'qfq'）意味着上游改默认口径
                # 时，本 loader 与 REST loader 会静默分叉。'qfq' = 前复权（不含息），
                # 与改动前的缺省口径一致 → 显式化不改变既有结果。
                ret, data, *_ = ctx.request_history_kline(
                    futu_code,
                    start=start_date,
                    end=end_date,
                    ktype=ktype,
                    autype="qfq",
                    max_count=10_000,
                )
                if ret != futu.RET_OK:
                    logger.warning("Futu returned error for %s: %s", futu_code, data)
                    continue
                normalized = _normalize_frame(data)
                loader_cache_put(
                    source=self.name,
                    symbol=code,
                    timeframe=interval,
                    start_date=start_date,
                    end_date=end_date,
                    fields=None,
                    frame=normalized,
                )
                results[code] = normalized
        finally:
            ctx.close()

        return results
