"""Unit tests for backtest.models module."""
import pandas as pd
import pytest

from backtest.models import EquitySnapshot, Position, TradeRecord


class TestPosition:
    def test_creation(self):
        pos = Position(
            symbol="AAPL",
            direction=1,
            entry_price=150.0,
            entry_time=pd.Timestamp("2024-01-01"),
            size=100.0,
        )
        assert pos.symbol == "AAPL"
        assert pos.direction == 1
        assert pos.leverage == 1.0
        assert pos.entry_bar_idx == 0
        assert pos.entry_commission == 0.0

    def test_frozen(self):
        pos = Position(
            symbol="BTC-USDT",
            direction=-1,
            entry_price=40000.0,
            entry_time=pd.Timestamp("2024-06-01"),
            size=0.5,
            leverage=5.0,
        )
        with pytest.raises(AttributeError):
            pos.symbol = "ETH-USDT"  # type: ignore[misc]


class TestTradeRecord:
    def test_creation(self):
        trade = TradeRecord(
            symbol="09988.HK",
            direction=1,
            entry_price=80.0,
            exit_price=90.0,
            entry_time=pd.Timestamp("2024-01-01"),
            exit_time=pd.Timestamp("2024-01-15"),
            size=1000.0,
            leverage=1.0,
            pnl=10000.0,
            pnl_pct=0.125,
            exit_reason="signal",
            holding_bars=10,
            commission=50.0,
        )
        assert trade.pnl == 10000.0
        assert trade.holding_bars == 10
        assert trade.exit_reason == "signal"

    def test_frozen(self):
        trade = TradeRecord(
            symbol="AAPL",
            direction=1,
            entry_price=100.0,
            exit_price=110.0,
            entry_time=pd.Timestamp("2024-01-01"),
            exit_time=pd.Timestamp("2024-01-05"),
            size=10.0,
            leverage=1.0,
            pnl=100.0,
            pnl_pct=0.1,
            exit_reason="trailing_stop",
            holding_bars=4,
            commission=2.0,
        )
        with pytest.raises(AttributeError):
            trade.pnl = 999.0  # type: ignore[misc]


class TestEquitySnapshot:
    def test_creation(self):
        snap = EquitySnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            capital=900000.0,
            unrealized=5000.0,
            equity=905000.0,
            positions=2,
        )
        assert snap.capital == 900000.0
        assert snap.unrealized == 5000.0
        assert snap.equity == 905000.0
        assert snap.positions == 2
