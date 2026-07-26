"""Enhanced strategy vs buy-and-hold benchmark comparison.

Provides detailed Alpha/Beta/Tracking Error/Capture Ratio analysis beyond
the basic benchmark comparison already in metrics.py.

Usage: triggered by config["benchmark_comparison"] in the backtest config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ComparisonMetrics:
    """Strategy vs benchmark comparison metrics.

    Attributes:
        beta: Market beta (covariance / benchmark variance).
        alpha: Jensen's alpha (annualized excess return adjusted for beta).
        tracking_error: Active return standard deviation (annualized).
        up_capture: Up-market capture ratio (strategy return when benchmark > 0).
        down_capture: Down-market capture ratio (strategy return when benchmark < 0).
        max_underwater_bars: Max consecutive bars underperforming benchmark.
        information_ratio: Active return / tracking error.
    """

    beta: float
    alpha: float
    tracking_error: float
    up_capture: float
    down_capture: float
    max_underwater_bars: int
    information_ratio: float


def detailed_benchmark_comparison(
    equity_curve: pd.Series,
    bench_ret: pd.Series,
    initial_cash: float,
    bars_per_year: int = 252,
) -> Dict[str, Any]:
    """Compute detailed strategy vs benchmark comparison.

    Args:
        equity_curve: Strategy equity time series.
        bench_ret: Benchmark return series (aligned with equity_curve).
        initial_cash: Starting capital.
        bars_per_year: Annualization factor.

    Returns:
        Dict with comparison metrics and chart data.
    """
    if len(equity_curve) < 10 or len(bench_ret) < 10:
        return {"error": "insufficient data for comparison"}

    # Align indices
    common_idx = equity_curve.index.intersection(bench_ret.index)
    if len(common_idx) < 10:
        return {"error": "insufficient overlapping data"}

    equity_aligned = equity_curve.loc[common_idx]
    bench_aligned = bench_ret.loc[common_idx]

    # Strategy returns
    port_ret = equity_aligned.pct_change().dropna()
    bench_ret_clean = bench_aligned.reindex(port_ret.index).fillna(0.0)

    if len(port_ret) < 5:
        return {"error": "insufficient return observations"}

    # Beta: cov(port, bench) / var(bench)
    cov_matrix = np.cov(port_ret.values, bench_ret_clean.values)
    bench_var = cov_matrix[1, 1]
    if bench_var > 1e-10:
        beta = float(cov_matrix[0, 1] / bench_var)
    else:
        beta = 0.0

    # Annualized returns
    n_bars = len(port_ret)
    port_total = float((1 + port_ret).prod() - 1)
    bench_total = float((1 + bench_ret_clean).prod() - 1)
    port_ann = float((1 + port_total) ** (bars_per_year / max(n_bars, 1)) - 1)
    bench_ann = float((1 + bench_total) ** (bars_per_year / max(n_bars, 1)) - 1)

    # Jensen's Alpha: port_ann - (rf + beta * (bench_ann - rf))
    # Assuming rf = 0 for simplicity (or use risk-free rate if available)
    alpha = port_ann - beta * bench_ann

    # Tracking Error: std(active returns) * sqrt(bars_per_year)
    active_ret = port_ret.values - bench_ret_clean.reindex(port_ret.index).fillna(0.0).values
    tracking_error = float(np.std(active_ret) * np.sqrt(bars_per_year))

    # Information Ratio: alpha / tracking_error
    if tracking_error > 1e-10:
        information_ratio = alpha / tracking_error
    else:
        information_ratio = 0.0

    # Up/Down Capture Ratios
    up_mask = bench_ret_clean > 0
    down_mask = bench_ret_clean < 0

    if up_mask.sum() > 0:
        up_capture = float(port_ret[up_mask].mean() / bench_ret_clean[up_mask].mean())
    else:
        up_capture = 0.0

    if down_mask.sum() > 0:
        down_capture = float(port_ret[down_mask].mean() / bench_ret_clean[down_mask].mean())
    else:
        down_capture = 0.0

    # Max consecutive bars underperforming benchmark
    underperforming = (port_ret.values < bench_ret_clean.reindex(port_ret.index).fillna(0.0).values)
    max_underwater = _max_consecutive_bool(underperforming)

    # Build chart data: cumulative returns overlay
    port_cum = (1 + port_ret).cumprod() - 1
    bench_cum = (1 + bench_ret_clean.reindex(port_ret.index).fillna(0.0)).cumprod() - 1

    chart_data = [
        {
            "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
            "strategy_return": round(float(port_cum.loc[idx]), 6),
            "benchmark_return": round(float(bench_cum.loc[idx]), 6),
        }
        for idx in port_cum.index[::max(len(port_cum) // 100, 1)]  # Sample for chart
    ]

    # Relative performance
    relative = port_cum - bench_cum
    max_outperformance = float(relative.max())
    max_underperformance = float(relative.min())

    return {
        "metrics": {
            "beta": round(beta, 4),
            "alpha": round(alpha, 6),
            "tracking_error": round(tracking_error, 6),
            "information_ratio": round(information_ratio, 4),
            "up_capture": round(up_capture, 4),
            "down_capture": round(down_capture, 4),
            "max_underwater_bars": max_underwater,
        },
        "summary": {
            "strategy_total_return": round(port_total, 6),
            "benchmark_total_return": round(bench_total, 6),
            "strategy_annualized": round(port_ann, 6),
            "benchmark_annualized": round(bench_ann, 6),
            "excess_return": round(port_total - bench_total, 6),
            "max_outperformance": round(max_outperformance, 6),
            "max_underperformance": round(max_underperformance, 6),
        },
        "chart_data": chart_data,
    }


def _max_consecutive_bool(arr: np.ndarray) -> int:
    """Find maximum consecutive True values in boolean array."""
    max_count = 0
    count = 0
    for v in arr:
        if v:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count
