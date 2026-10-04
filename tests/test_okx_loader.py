"""Unit tests for the OKX loader's ``-USD`` alias handling.

Regression guard for the ``market_data(codes=["BTC-USD"])`` failure: OKX V5
``instId`` quotes only against USDT, so ``BTC-USD`` is answered with
``code == "51001"`` and an empty ``data`` array, which the loader reports as a
failed symbol (``unresolved`` upstream).
"""
from __future__ import annotations

import pandas as pd
import pytest

import backtest.loaders.okx as okx
from backtest.loaders.okx import DataLoader, _to_okx_inst_id


def _frame() -> pd.DataFrame:
    """Minimal OHLCV frame shaped like ``_fetch_candles`` output."""
    idx = pd.to_datetime(["2026-09-01", "2026-09-02"])
    return pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [10.0, 20.0],
        },
        index=pd.DatetimeIndex(idx, name="trade_date"),
    )


class TestToOkxInstId:
    """``-USD`` must map onto the USDT pair, everything else passes through."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("BTC-USD", "BTC-USDT"),
            ("ETH-USD", "ETH-USDT"),
            ("DOGE-USD", "DOGE-USDT"),
            # Already a valid instId — untouched.
            ("BTC-USDT", "BTC-USDT"),
            # Different quote — untouched.
            ("ETH-BTC", "ETH-BTC"),
            # ``-USD`` not at the tail must not be rewritten (a naive
            # ``symbol.replace("-USD", "-USDT")`` would corrupt this).
            ("BTC-USD-SWAP", "BTC-USD-SWAP"),
        ],
    )
    def test_mapping(self, raw: str, expected: str) -> None:
        assert _to_okx_inst_id(raw) == expected

    def test_usd_alias_is_not_double_hyphenated(self) -> None:
        """``symbol[:-3] + "-USDT"`` yields ``BTC--USDT`` — also a 51001.

        The off-by-one is invisible without an assertion on the exact string,
        so pin it explicitly.
        """
        for raw in ("BTC-USD", "ETH-USD", "SOL-USD"):
            inst_id = _to_okx_inst_id(raw)
            assert "--" not in inst_id, f"{raw} -> {inst_id}"
            assert inst_id == raw.replace("-USD", "-USDT")


class TestFetchUsdAlias:
    """``fetch`` queries the USDT instId but keys results by the original code."""

    def _patch(self, monkeypatch, seen: list[str], frame=_frame):
        def fake_cached_loader_fetch(*, source, symbol, timeframe, start_date,
                                     end_date, fields, fetch):
            seen.append(symbol)
            return fetch()

        def fake_fetch_candles(self, inst_id, start_ts, end_ts, bar="1D",
                               max_pages=20):
            return frame() if frame is not None else None

        monkeypatch.setattr(okx, "cached_loader_fetch", fake_cached_loader_fetch)
        monkeypatch.setattr(okx.DataLoader, "_fetch_candles", fake_fetch_candles)

    def test_usd_alias_resolves_and_keeps_original_key(self, monkeypatch) -> None:
        seen: list[str] = []
        self._patch(monkeypatch, seen)

        result = DataLoader().fetch(["btc-usd"], "2026-09-01", "2026-09-30")

        # Queried as the USDT pair ...
        assert seen == ["BTC-USDT"]
        # ... but keyed by the caller's code so upstream resolution matches.
        assert list(result) == ["BTC-USD"]
        assert not result["BTC-USD"].empty

    def test_usdt_and_usd_forms_both_work(self, monkeypatch) -> None:
        seen: list[str] = []
        self._patch(monkeypatch, seen)

        result = DataLoader().fetch(
            ["BTC-USD", "BTC-USDT"], "2026-09-01", "2026-09-30",
        )

        assert seen == ["BTC-USDT", "BTC-USDT"]
        assert sorted(result) == ["BTC-USD", "BTC-USDT"]

    def test_slash_form_is_normalised(self, monkeypatch) -> None:
        """``BTC/USD`` (ccxt spelling) is accepted too."""
        seen: list[str] = []
        self._patch(monkeypatch, seen)

        result = DataLoader().fetch(["BTC/USD"], "2026-09-01", "2026-09-30")

        assert seen == ["BTC-USDT"]
        assert list(result) == ["BTC-USD"]

    def test_empty_payload_marks_symbol_unresolved(self, monkeypatch) -> None:
        """No data -> absent from the mapping (never a ``None``/empty entry)."""
        seen: list[str] = []
        self._patch(monkeypatch, seen, frame=None)

        assert DataLoader().fetch(["BTC-USD"], "2026-09-01", "2026-09-30") == {}
