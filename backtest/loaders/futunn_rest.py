"""Futunn OpenAPI (REST + API Key) loader for HK / US / China A-share OHLCV.

第二条 Futu 通道：**不依赖本机 OpenD 网关**，直连富途 REST 网关并用
``FUTUNN_APP_KEY`` + 私钥签名取数。与 :mod:`backtest.loaders.futu`（OpenD 通道）
产出**同样的** normalize 结果与**同样的** symbol 换算（复用 ``_to_futu_symbol``），
因此两条通道可以逐行比对、也可以随时切换。

⚠️ 本模块是 trading-backtest 里**唯一**依赖 trading-framework ``tools.futunn`` 的
loader。为不破坏「``trading-backtest`` 单独安装也能跑」这一前提，**所有跨仓 import
都放在方法内部**（模块级只依赖 pandas/自身包），这样：

- ``registry._ensure_registered()`` 导入本模块不会失败；
- 未安装 trading-framework 时 ``is_available()`` 返回 ``False``，fallback 链继续走。

⚠️ 与 OpenD 通道的差别（都是上游网关的硬约束，已由 ``tools.futunn.quotes`` 兜住）：

- 单次响应上限 1000 行且**静默截断**（``has_more=False``/``next_time=None``）→ 区间二分；
- ``num`` 上限 370；
- 盘中 ktype 的 ``end`` 是**独占**的（日线是含当天）→ 发送前统一推后一天。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

# 与 OpenD loader 共用符号换算（``700.HK`` → ``HK.00700`` 等），避免两条通道漂移。
# 该模块的 ``futu`` SDK import 是惰性的，所以这里导入不会拉起 SDK。
from backtest.loaders.futu import _to_futu_symbol
from backtest.loaders.base import (
    NoAvailableSourceError,
    cached_loader_fetch,
    positive_env_float,
    validate_date_range,
)
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

#: Target OHLCV schema, in column order (identical to the OpenD loader).
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

#: HTTP timeout for the REST channel (seconds).
_TIMEOUT_ENV = "FUTUNN_HTTP_TIMEOUT"
_DEFAULT_TIMEOUT = 15.0


@register
class FutunnRestLoader:
    """Fetch HK / US / China A-share bars from the Futunn REST API.

    Credentials come from the environment only (``FUTUNN_APP_KEY`` plus
    ``FUTUNN_PRIVATE_KEY`` or ``FUTUNN_PRIVATE_KEY_PATH``); nothing is read from
    or written to a config file here.
    """

    name = "futunn"
    markets = {"hk_equity", "a_share", "us_equity"}
    requires_auth = True

    def __init__(self) -> None:
        """Initialize the loader (no network, no credential read at construction)."""
        self._timeout = positive_env_float(_TIMEOUT_ENV, _DEFAULT_TIMEOUT)

    def is_available(self) -> bool:
        """Return whether the REST channel is importable and credentialed.

        Deliberately **does not** make a network call: this is consulted while
        walking fallback chains, so a probe here would add latency to every
        resolution. A rejected key/clock skew surfaces in :meth:`fetch`.
        """
        try:
            import tools.futunn.client  # noqa: F401, PLC0415
        except ImportError:
            return False
        return _credentials_present()

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV history from the Futunn REST API.

        Args:
            codes: Project symbols such as ``700.HK`` / ``AAPL.US`` / ``000001.SZ``.
            start_date: Inclusive start date in ``YYYY-MM-DD`` format.
            end_date: Inclusive end date in ``YYYY-MM-DD`` format.
            interval: Backtest interval — ``1D`` / ``1H`` / ``4H`` / ``1W`` / ``1M``
                （也接受 ``1d`` / ``1h`` / ``1m``（1 分钟）/ ``5m`` / ``30m`` / ``1MO``
                等 canonical 写法与别名）。⚠️ ``1M`` = 月、``1m`` = 1 分钟；
                **未登记的写法抛 ``ValueError``**（不再静默回落日线）。
            fields: Ignored; included for interface compatibility.

        Returns:
            Mapping of input symbol to a normalized OHLCV dataframe, indexed by a
            tz-naive ``trade_date`` DatetimeIndex in the **market's local time**
            (identical to what the OpenD loader produces).

        Raises:
            NoAvailableSourceError: If the REST client cannot be constructed
                (missing credentials / unloadable key).
            ValueError: If ``start_date`` > ``end_date``.
        """
        del fields
        if not codes:
            return {}
        validate_date_range(start_date, end_date)

        from tools.futunn.client import FutunnClient  # noqa: PLC0415
        from tools.futunn.endpoints import AUTYPE_DEFAULT, ktype_for_interval  # noqa: PLC0415

        # ⚠️ 严格解析：未登记的 interval 抛错，绝不静默回落日线。
        # ⚠️ ``1M`` = 月、``1m`` = 1 分钟（此前这里 .upper() 把 ``1m`` 折成 ``1M`` →
        #    静默拿到月线，与本 loader 文档承诺的「1 分钟」差 1000 倍粒度）。
        ktype = ktype_for_interval(interval)
        # ⚠️ 复权口径显式下发（缺省 qfq = 前复权，与网关缺省一致 → 行为不变）。
        autype = AUTYPE_DEFAULT
        try:
            client = FutunnClient.from_env(timeout_s=self._timeout)
        except (ValueError, OSError) as exc:
            raise NoAvailableSourceError(f"Futunn REST 通道不可用：{exc}") from exc

        results: Dict[str, pd.DataFrame] = {}
        for code in codes:
            try:
                frame = cached_loader_fetch(
                    source=self.name,
                    symbol=code,
                    timeframe=interval,
                    start_date=start_date,
                    end_date=end_date,
                    fields=None,
                    fetch=lambda code=code: self._fetch_one(
                        client, code, start_date, end_date, ktype, autype
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - 单个标的失败不该拖垮整批
                logger.warning("futunn failed for %s: %s", code, exc)
                continue
            if frame is not None and not frame.empty:
                results[code] = frame
        return results

    def _fetch_one(
        self,
        client,
        code: str,
        start_date: str,
        end_date: str,
        ktype: int,
        autype: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch and normalize one symbol; ``None`` when there is no usable bar."""
        from tools.futunn.quotes import fetch_history_kline  # noqa: PLC0415

        symbol = _to_futu_symbol(code)
        if "." not in symbol:
            logger.warning("futunn skipped %s: cannot map to a market-qualified code", code)
            return None
        bars, warnings = fetch_history_kline(
            client, symbol, ktype, start=start_date, end=end_date, autype=autype
        )
        for warning in warnings:
            logger.warning("futunn %s: %s", symbol, warning)
        if not bars:
            logger.warning("futunn returned no bars for %s (%s..%s)", symbol, start_date, end_date)
            return None
        return _normalize_frame(bars, symbol)


def _credentials_present() -> bool:
    """Return whether both an AppKey and a private-key source are configured."""
    from tools.futunn.client import credentials_present  # noqa: PLC0415

    return credentials_present()


def _normalize_frame(bars: List[dict], symbol: str) -> pd.DataFrame:
    """Normalise REST kline rows to the standard OHLCV schema.

    ``fetch_history_kline`` returns raw gateway rows keyed by ``time_key`` (epoch
    **milliseconds**). They are rendered with the shared
    :func:`tools.futunn.models.format_market_time` — the exact same rendering the
    connector's ``get_historical_bars`` uses — so the index is a tz-naive
    :class:`~pandas.DatetimeIndex` in the **market's local time**, matching what
    the OpenD loader gets from ``time_key`` strings (e.g. ``2026-06-01 00:00:00``
    for an HK daily bar). A REST frame and an OpenD frame therefore line up
    row-for-row with no timezone gymnastics.

    Args:
        bars: Rows from :func:`tools.futunn.quotes.fetch_history_kline`, ascending.
        symbol: Market-qualified code (``HK.00700``) — selects the market timezone.

    Returns:
        DataFrame with columns ``[open, high, low, close, volume]`` indexed by
        ``trade_date``, sorted ascending. Empty when no bar is usable.
    """
    if not bars:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)

    from tools.futunn.models import format_market_time  # noqa: PLC0415

    parsed: List[dict] = []
    for bar in bars:
        stamp = format_market_time(bar.get("time_key"), symbol)
        if not stamp:
            continue
        row: Dict[str, object] = {"trade_date": stamp}
        for column in _OHLCV_COLUMNS:
            row[column] = bar.get(column)
        parsed.append(row)
    if not parsed:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)

    frame = pd.DataFrame(parsed)
    frame.index = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame.index.name = "trade_date"
    frame = frame[frame.index.notna()]
    frame = frame[_OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame = frame.astype("float64")
    frame["volume"] = frame["volume"].fillna(0.0)
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return frame.sort_index()
