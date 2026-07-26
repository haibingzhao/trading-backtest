"""Unit tests for backtest.metrics module."""
import numpy as np
import pandas as pd
import pytest

from backtest.metrics import calc_bars_per_year, calc_metrics, win_rate_and_stats
from backtest.models import TradeRecord


def _make_trade(
    pnl: float,
    symbol: str = "AAPL",
    direction: int = 1,
    holding_bars: int = 5,
    exit_reason: str = "signal",
) -> TradeRecord:
    """Helper to create a TradeRecord with minimal boilerplate."""
    return TradeRecord(
        symbol=symbol,
        direction=direction,
        entry_price=100.0,
        exit_price=100.0 + pnl / 10,  # size=10
        entry_time=pd.Timestamp("2024-01-01"),
        exit_time=pd.Timestamp("2024-01-10"),
        size=10.0,
        leverage=1.0,
        pnl=pnl,
        pnl_pct=pnl / 1000.0,
        exit_reason=exit_reason,
        holding_bars=holding_bars,
        commission=1.0,
    )


class TestCalcBarsPerYear:
    def test_daily_tushare(self):
        assert calc_bars_per_year("1D", "tushare") == 252

    def test_daily_okx(self):
        assert calc_bars_per_year("1D", "okx") == 365

    def test_1m_tushare(self):
        assert calc_bars_per_year("1m", "tushare") == 252 * 240

    def test_1m_ccxt(self):
        assert calc_bars_per_year("1m", "ccxt") == 365 * 1440

    def test_unknown_source_defaults(self):
        # Unknown source defaults to 252 trading days, 1 bar/day
        assert calc_bars_per_year("1D", "unknown_source") == 252


class TestWinRateAndStats:
    def test_empty_trades(self):
        stats = win_rate_and_stats([])
        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0.0

    def test_all_wins(self):
        trades = [_make_trade(pnl=100.0) for _ in range(5)]
        stats = win_rate_and_stats(trades)
        assert stats["win_rate"] == 1.0
        assert stats["max_consecutive_loss"] == 0

    def test_all_losses(self):
        trades = [_make_trade(pnl=-50.0) for _ in range(3)]
        stats = win_rate_and_stats(trades)
        assert stats["win_rate"] == 0.0
        assert stats["max_consecutive_loss"] == 3

    def test_mixed_trades(self):
        trades = [
            _make_trade(pnl=200.0),
            _make_trade(pnl=-100.0),
            _make_trade(pnl=-50.0),
            _make_trade(pnl=150.0),
        ]
        stats = win_rate_and_stats(trades)
        assert stats["win_rate"] == 0.5
        assert stats["max_consecutive_loss"] == 2
        assert stats["profit_factor"] == pytest.approx(350.0 / 150.0, rel=1e-3)

    def test_avg_holding_bars(self):
        trades = [
            _make_trade(pnl=100.0, holding_bars=10),
            _make_trade(pnl=50.0, holding_bars=20),
        ]
        stats = win_rate_and_stats(trades)
        assert stats["avg_holding_bars"] == 15.0


class TestCalcMetrics:
    def test_empty_equity_curve(self):
        metrics = calc_metrics(pd.Series(dtype=float), [], 100000.0)
        assert metrics["total_return"] == 0
        assert metrics["trade_count"] == 0

    def test_flat_equity(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        equity = pd.Series(100000.0, index=dates)
        metrics = calc_metrics(equity, [], 100000.0)
        assert metrics["total_return"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-10)

    def test_positive_return(self):
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        equity = pd.Series(np.linspace(100000, 120000, 252), index=dates)
        metrics = calc_metrics(equity, [], 100000.0)
        assert metrics["total_return"] == pytest.approx(0.20, rel=1e-3)
        assert metrics["sharpe"] > 0
        assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-6)

    def test_drawdown(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        values = [100000] * 50 + [90000] * 50  # 10% drawdown
        equity = pd.Series(values, index=dates, dtype=float)
        metrics = calc_metrics(equity, [], 100000.0)
        assert metrics["max_drawdown"] == pytest.approx(-0.10, rel=1e-3)

    def test_with_trades(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        equity = pd.Series(np.linspace(100000, 110000, 100), index=dates)
        trades = [_make_trade(pnl=500.0), _make_trade(pnl=-200.0)]
        metrics = calc_metrics(equity, trades, 100000.0)
        assert metrics["trade_count"] == 2
        assert metrics["win_rate"] == 0.5

    def test_with_benchmark(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        equity = pd.Series(np.linspace(100000, 115000, 100), index=dates)
        bench_ret = pd.Series(0.001, index=dates)  # 0.1% per bar
        metrics = calc_metrics(equity, [], 100000.0, bench_ret=bench_ret)
        assert metrics["benchmark_return"] > 0
        assert "information_ratio" in metrics
