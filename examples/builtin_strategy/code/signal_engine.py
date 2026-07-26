"""Built-in Strategy Framework Example.

Uses the built-in DefaultSignalEngine which provides:
- Regime detection (trending vs ranging markets)
- Grid strategy for ranging markets
- Trend-following with pyramid entries for trending markets
- Risk management (drawdown halt, consecutive stop pause, daily loss limit)

All parameters are configured via config.json signal_params.
"""
from backtest.strategy import DefaultSignalEngine as SignalEngine
