"""Simple EMA Crossover Strategy.

A minimal signal engine demonstrating the SignalEngine interface.
Generates long/short signals based on EMA fast/slow line crossover.
"""
import pandas as pd


class SignalEngine:
    """EMA crossover signal engine.

    Args:
        ema_fast: Fast EMA period (default: 12).
        ema_slow: Slow EMA period (default: 26).
    """

    def __init__(self, ema_fast: int = 12, ema_slow: int = 26, **kwargs):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def generate(self, data_map: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        """Generate trading signals based on EMA crossover.

        Args:
            data_map: {code: DataFrame} with OHLCV columns and DatetimeIndex.

        Returns:
            {code: Series} with signal values: +1 (long), -1 (short), 0 (flat).
        """
        signals = {}
        for code, df in data_map.items():
            fast_ema = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
            slow_ema = df["close"].ewm(span=self.ema_slow, adjust=False).mean()

            signal = pd.Series(0, index=df.index, dtype=int)
            signal[fast_ema > slow_ema] = 1   # Golden cross → long
            signal[fast_ema < slow_ema] = -1  # Death cross → short

            signals[code] = signal
        return signals
