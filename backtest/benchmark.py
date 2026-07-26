"""Benchmark ticker resolution and fetch for backtest comparison.

Provides a lightweight, zero-dependency way to fetch benchmark reference
data given a set of strategy codes and a data source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backtest.loaders.registry import LOADER_REGISTRY, _ensure_registered

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Benchmark map: market type → default ticker
# -------------------------------------------------------------------

MARKET_BENCHMARKS: dict[str, Optional[str]] = {
    "us_equity":  "SPY",
    "hk_equity":  "HK.03100",   # Hang Seng China Enterprises ETF
    "a_share":    "000300.SH",  # CSI 300 (China A-share core index)
    "crypto":     "BTC-USDT",
    "futures":    "ES.CME",      # E-mini S&P 500 futures
    "forex":      None,         # no universal benchmark
}

# Ordered fallback chains per market: tried in sequence when the primary
# loader fails (rate-limit, network error, empty result, …).
BENCHMARK_FALLBACK_CHAINS: dict[str, list[str]] = {
    "us_equity":  ["yfinance", "yahoo", "stooq", "eastmoney", "sina"],
    "hk_equity":  ["yfinance", "yahoo", "eastmoney"],
    "a_share":    ["tushare", "akshare", "eastmoney", "baostock"],
    "crypto":     ["yfinance", "okx", "ccxt"],
    "futures":    ["tushare", "akshare"],
}

# Map yfinance-format benchmark tickers to project-format symbols that
# fallback loaders (yahoo, stooq, …) understand.  yfinance uses bare
# tickers ("SPY") while other loaders expect suffixed symbols ("SPY.US").
_BENCHMARK_SYMBOL_MAP: dict[str, str] = {
    "SPY":       "SPY.US",
    "QQQ":       "QQQ.US",
    "IWM":       "IWM.US",
    "ES.CME":    "ES.CME",
    "BTC-USDT":  "BTC-USDT",
}


@dataclass
class BenchmarkResult:
    ticker:     str
    ret_series: pd.Series       # per-bar returns, index = timestamps
    total_ret: float          # total return over the period


def resolve_benchmark(
    strategy_codes: list[str],
    source:       str,
    start_date:   str,
    end_date:     str,
    interval:     str = "1D",
    explicit:     Optional[str] = None,
) -> Optional[BenchmarkResult]:
    """Resolve the appropriate benchmark ticker and fetch its return series.

    Args:
        strategy_codes: Instruments being backtested (used for market inference).
        source:         Data source name (tushare / yfinance / okx / akshare / ccxt).
        start_date:     Backtest start date.
        end_date:       Backtest end date.
        interval:       Bar interval (1m / 5m / 15m / 30m / 1H / 4H / 1D).
        explicit:       Override ticker (e.g. "SPY" passed via config).

    Returns:
        BenchmarkResult with return series and total return, or None if no
        benchmark applies (forex, or fetch failure).
    """
    ticker, market = _resolve_ticker(strategy_codes, source, explicit)
    if ticker is None:
        return None

    try:
        bench_df = _fetch_benchmark(ticker, start_date, end_date, interval, market=market)
    except Exception:
        return None

    if bench_df.empty or "close" not in bench_df.columns:
        return None

    close = bench_df["close"].dropna()
    if len(close) < 2:
        return None

    ret_series = close.pct_change().fillna(0.0)
    total_ret   = float((1 + ret_series).prod() - 1)

    return BenchmarkResult(ticker=ticker, ret_series=ret_series, total_ret=total_ret)


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _resolve_ticker(
    codes:     list[str],
    source:    str,
    explicit:  Optional[str],
) -> tuple[Optional[str], str]:
    """Pick the benchmark ticker and infer the market type.

    Returns:
        (ticker, market) — ticker may be None for markets with no benchmark.
    """
    market = _infer_market(codes, source)

    if explicit:
        return explicit, market

    ticker = MARKET_BENCHMARKS.get(market)
    return ticker, market


def _infer_market(codes: list[str], source: str) -> str:
    """Rough market inference from symbol patterns and source."""
    if not codes:
        return "us_equity"

    first = codes[0].upper()

    if source in ("okx", "ccxt") or "-" in first or "/" in first:
        return "crypto"
    if first.endswith(".US"):
        return "us_equity"
    if first.endswith(".HK"):
        return "hk_equity"
    if source in ("tushare", "akshare"):
        if first.isdigit() and len(first) == 6:
            return "a_share"
        if first.startswith(("IF", "IC", "IH", "IM", "T", "TF")):
            return "futures"
        return "a_share"

    return "us_equity"


def _fetch_benchmark(
    ticker:    str,
    start_date: str,
    end_date:   str,
    interval:   str,
    market:     str = "us_equity",
) -> pd.DataFrame:
    """Fetch benchmark OHLCV data with fallback chain.

    Tries each loader in BENCHMARK_FALLBACK_CHAINS[market] in order.
    Returns the first non-empty result; returns empty DataFrame if all fail.
    """
    _ensure_registered()
    chain = BENCHMARK_FALLBACK_CHAINS.get(market, ["yfinance"])

    for loader_name in chain:
        loader_cls = LOADER_REGISTRY.get(loader_name)
        if loader_cls is None:
            continue
        try:
            loader = loader_cls()
        except Exception as exc:
            logger.debug("benchmark: loader %s failed to construct: %s", loader_name, exc)
            continue
        if not loader.is_available():
            continue
        # yfinance uses bare tickers ("SPY"); other loaders need project
        # format ("SPY.US").  Try the mapped symbol for non-yfinance loaders.
        fetch_ticker = ticker
        if loader_name != "yfinance":
            fetch_ticker = _BENCHMARK_SYMBOL_MAP.get(ticker, ticker)
        try:
            result = loader.fetch(
                [fetch_ticker], start_date, end_date, interval=interval,
            )
        except Exception as exc:
            logger.warning("benchmark: %s fetch failed for %s: %s", loader_name, fetch_ticker, exc)
            continue

        df = _extract_df(result, fetch_ticker)
        if not df.empty:
            logger.info("benchmark: fetched %s via %s (as %s)", ticker, loader_name, fetch_ticker)
            return df
        logger.debug("benchmark: %s returned empty data for %s", loader_name, fetch_ticker)

    logger.warning("benchmark: all sources failed for %s (market=%s)", ticker, market)
    return pd.DataFrame()


def _extract_df(result, ticker: str) -> pd.DataFrame:
    """Normalise a loader's fetch result into a DataFrame."""
    if isinstance(result, dict):
        df = result.get(ticker)
    elif isinstance(result, pd.DataFrame):
        df = result
    else:
        return pd.DataFrame()

    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame()
    return df