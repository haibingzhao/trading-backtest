"""trading-backtest — Multi-market backtesting engine.

Usage::

    from backtest import BacktestRunner, run_backtest

    # High-level API
    results = run_backtest(
        codes=["AAPL", "MSFT"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        signal_engine=MySignalEngine(),
    )

    # CLI
    python -m backtest.runner <run_dir>
"""

from backtest.runner import main as run_backtest_cli
from backtest.models import EquitySnapshot, TradeRecord
from backtest.loaders.base import DataLoaderProtocol, NoAvailableSourceError
from backtest.loaders.registry import LOADER_REGISTRY, get_loader_cls_with_fallback
from backtest.engines.base import BaseEngine
from backtest.metrics import calc_metrics

__all__ = [
    "run_backtest_cli",
    "EquitySnapshot",
    "TradeRecord",
    "DataLoaderProtocol",
    "NoAvailableSourceError",
    "BaseEngine",
    "LOADER_REGISTRY",
    "get_loader_cls_with_fallback",
    "calc_metrics",
]
