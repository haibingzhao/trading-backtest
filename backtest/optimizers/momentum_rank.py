"""Momentum ranking optimizer: weight by trailing return rank.

Cross-asset momentum ranking — rank assets by trailing return (or risk-adjusted
momentum), allocate higher weight to top-ranked assets. Optionally filter out
negative momentum assets.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtest.optimizers.base import BaseOptimizer


class MomentumRankOptimizer(BaseOptimizer):
    """Cross-asset momentum ranking optimizer.

    Ranks active assets by their trailing-period return (or risk-adjusted
    momentum) and allocates weights based on rank. Assets with negative
    momentum can be excluded.

    Args:
        lookback: Lookback period for momentum calculation.
        top_n: Number of top assets to select (0 = all positive momentum).
        filter_negative: If True, exclude assets with negative momentum.
        vol_adjust: If True, use risk-adjusted momentum (RAM = momentum / vol)
            and filter out abnormally volatile assets.
        vol_lookback: Lookback window for volatility calculation (only used
            when vol_adjust=True).
    """

    def __init__(
        self,
        lookback: int = 60,
        top_n: int = 0,
        filter_negative: bool = True,
        vol_adjust: bool = False,
        vol_lookback: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(lookback=lookback, **kwargs)
        self.top_n = top_n
        self.filter_negative = filter_negative
        self.vol_adjust = vol_adjust
        self.vol_lookback = vol_lookback

    def _build_context(
        self, window: pd.DataFrame, active: List[str]
    ) -> Dict[str, Any] | None:
        """Build context with mean returns for momentum ranking."""
        cov = window.cov().values
        mu = window.mean().values
        if np.isnan(cov).any() or np.isnan(mu).any():
            return None

        ctx: Dict[str, Any] = {"cov": cov, "mu": mu, "momentum": mu}

        # Compute per-asset volatility for vol-adjust mode
        if self.vol_adjust:
            vol_window = window.iloc[-self.vol_lookback:] if len(window) >= self.vol_lookback else window
            vol = vol_window.std().values
            ctx["vol"] = vol

        return ctx

    def _calc_weights(self, ctx: Dict[str, Any]) -> np.ndarray:
        """Rank assets by momentum, allocate weights by rank."""
        momentum = ctx["momentum"]
        n = len(momentum)

        # Volatility adjustment: RAM = momentum / vol, filter extreme vol
        vol_mask = np.ones(n, dtype=bool)
        if self.vol_adjust and "vol" in ctx:
            vol = ctx["vol"]
            vol_safe = np.where(vol > 1e-8, vol, 1e-8)
            momentum = momentum / vol_safe
            # Filter assets with vol > 2x median (abnormally volatile)
            median_vol = float(np.median(vol))
            if median_vol > 1e-8:
                vol_mask = vol <= 2.0 * median_vol

        # Filter negative momentum if requested
        if self.filter_negative:
            positive_mask = (momentum > 0) & vol_mask
            if not positive_mask.any():
                return self._equal_weight(n)
        else:
            positive_mask = vol_mask.copy()

        # Rank assets (higher momentum = higher rank = higher weight)
        ranks = np.zeros(n)
        valid_indices = np.where(positive_mask)[0]
        if len(valid_indices) == 0:
            return self._equal_weight(n)

        valid_momentum = momentum[valid_indices]
        # Rank from 1 (lowest) to N (highest)
        order = valid_momentum.argsort().argsort() + 1
        ranks[valid_indices] = order

        # Apply top-N filter
        if self.top_n > 0 and self.top_n < len(valid_indices):
            # Keep only top-N by momentum
            threshold = np.sort(valid_momentum)[-self.top_n]
            top_mask = momentum >= threshold
            ranks = np.where(top_mask, ranks, 0.0)

        # Linear decay weights (higher rank = higher weight)
        weights = ranks / ranks.sum() if ranks.sum() > 0 else self._equal_weight(n)

        return weights


def optimize(
    ret: pd.DataFrame,
    pos: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback: int = 60,
    top_n: int = 0,
    filter_negative: bool = True,
    vol_adjust: bool = False,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """Module-level entry: momentum-rank-adjusted positions."""
    return MomentumRankOptimizer(
        lookback=lookback, top_n=top_n, filter_negative=filter_negative,
        vol_adjust=vol_adjust, vol_lookback=vol_lookback,
    ).optimize(ret, pos, dates)
