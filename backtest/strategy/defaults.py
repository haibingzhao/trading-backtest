"""默认 SignalEngine 实现。

开箱即用，行为等价于原始 09988_hk_dual_mode 的 signal_engine.py。
"""

from __future__ import annotations

from backtest.strategy.base import StrategyBase


class DefaultSignalEngine(StrategyBase):
    """开箱即用的信号引擎。

    完全继承 StrategyBase 的默认实现，无需任何覆盖。
    用法::

        # runs/my_strategy/code/signal_engine.py
        from backtest.strategy import DefaultSignalEngine as SignalEngine
    """
