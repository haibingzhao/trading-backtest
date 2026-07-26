"""Rolling performance monitor for strategy diagnostics.

Computes rolling Sharpe ratio, rolling max drawdown, and rolling excess
return over a configurable window.  Can auto-detect deteriorating periods.

Usage::

    python -m backtest.rolling_monitor <run_dir>
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rolling_sharpe(
    equity_curve: pd.Series,
    window: int = 60,
    bars_per_year: float = 252,
) -> pd.Series:
    """Compute rolling annualized Sharpe ratio.

    Args:
        equity_curve: Cumulative equity series with DatetimeIndex.
        window: Rolling window size in bars.
        bars_per_year: Annualization factor.

    Returns:
        Series of rolling Sharpe values (NaN for the first ``window`` bars).
    """
    ret = equity_curve.pct_change().fillna(0.0)
    roll_mean = ret.rolling(window, min_periods=max(window // 2, 2)).mean()
    roll_std = ret.rolling(window, min_periods=max(window // 2, 2)).std()
    roll_std = roll_std.replace(0, np.nan)
    return (roll_mean / roll_std) * np.sqrt(bars_per_year)


def rolling_drawdown(
    equity_curve: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling maximum drawdown over the trailing window.

    Returns negative values (e.g. -0.15 means 15% drawdown).
    """
    def _max_dd_in_window(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.0
        peak = np.maximum.accumulate(x)
        dd = (x - peak) / np.where(peak > 0, peak, 1)
        return float(np.min(dd))

    return equity_curve.rolling(window, min_periods=max(window // 2, 2)).apply(
        _max_dd_in_window, raw=True,
    )


def rolling_excess(
    equity_curve: pd.Series,
    bench_ret: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling cumulative excess return over the trailing window."""
    port_ret = equity_curve.pct_change().fillna(0.0)
    excess = port_ret - bench_ret.reindex(equity_curve.index).fillna(0.0)
    return excess.rolling(window, min_periods=max(window // 2, 2)).sum()


def detect_bad_periods(
    roll_sharpe: pd.Series,
    threshold: float = 0.0,
    min_duration: int = 5,
) -> List[Dict[str, Any]]:
    """Detect periods where rolling Sharpe drops below *threshold*.

    Returns a list of dicts: ``{"start", "end", "duration_bars", "min_sharpe"}``.
    """
    bad = roll_sharpe < threshold
    if not bad.any():
        return []

    periods: List[Dict[str, Any]] = []
    start = None
    for i, (ts, is_bad) in enumerate(bad.items()):
        if is_bad and start is None:
            start = ts
        elif not is_bad and start is not None:
            duration = i - bad.index.get_loc(start)
            if duration >= min_duration:
                seg = roll_sharpe.loc[start:ts]
                periods.append({
                    "start": str(start.date()) if hasattr(start, "date") else str(start),
                    "end": str(ts.date()) if hasattr(ts, "date") else str(ts),
                    "duration_bars": duration,
                    "min_sharpe": round(float(seg.min()), 4),
                })
            start = None

    # Handle trailing bad period
    if start is not None:
        duration = len(bad) - bad.index.get_loc(start)
        if duration >= min_duration:
            seg = roll_sharpe.loc[start:]
            periods.append({
                "start": str(start.date()) if hasattr(start, "date") else str(start),
                "end": str(bad.index[-1].date()) if hasattr(bad.index[-1], "date") else str(bad.index[-1]),
                "duration_bars": duration,
                "min_sharpe": round(float(seg.min()), 4),
            })

    return periods


def generate_rolling_json(
    equity_curve: pd.Series,
    bench_ret: pd.Series,
    bars_per_year: float = 252,
    window: int | None = None,
) -> Dict[str, Any]:
    """Generate rolling monitor data as a JSON-serializable dict.

    Uses ~1/4 year window by default (roughly one quarter).
    """
    if window is None:
        window = max(int(bars_per_year / 4), 20)

    rs = rolling_sharpe(equity_curve, window, bars_per_year)
    rd = rolling_drawdown(equity_curve, window)
    re = rolling_excess(equity_curve, bench_ret, window)
    bad = detect_bad_periods(rs, threshold=0.0, min_duration=5)

    def _to_list(s: pd.Series) -> List[Dict[str, Any]]:
        out = []
        for ts, val in s.dropna().items():
            label = str(ts.date()) if hasattr(ts, "date") else str(ts)
            out.append({"date": label, "value": round(float(val), 4)})
        return out

    return {
        "window": window,
        "bars_per_year": bars_per_year,
        "rolling_sharpe": _to_list(rs),
        "rolling_drawdown": _to_list(rd),
        "rolling_excess_return": _to_list(re),
        "bad_periods": bad,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(run_dir: Path) -> None:
    """CLI entry: read equity.csv from artifacts, output rolling monitor."""
    artifacts = run_dir / "artifacts"
    eq_path = artifacts / "equity.csv"
    if not eq_path.exists():
        print(json.dumps({"error": "equity.csv not found in artifacts"}))
        sys.exit(1)

    eq_df = pd.read_csv(eq_path, index_col=0, parse_dates=True)
    equity_curve = eq_df["equity"]
    bench_ret = eq_df.get("active_ret", pd.Series(0.0, index=eq_df.index))
    # active_ret = port_ret - bench_ret, so bench_ret ≈ port_ret - active_ret
    port_ret = equity_curve.pct_change().fillna(0.0)
    bench_ret_series = port_ret - eq_df["active_ret"] if "active_ret" in eq_df.columns else pd.Series(0.0, index=eq_df.index)

    # Detect bars_per_year from config
    bars_per_year = 252.0
    config_path = run_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        source = cfg.get("source", "")
        if source in ("tushare", "akshare", "futu"):
            bars_per_year = 244.0

    result = generate_rolling_json(equity_curve, bench_ret_series, bars_per_year)

    out_path = artifacts / "rolling_monitor.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary
    n_bad = len(result["bad_periods"])
    print(f"滚动监控报告已生成: {out_path}")
    print(f"窗口: {result['window']} bars")
    print(f"恶化时段: {n_bad} 个")
    for bp in result["bad_periods"]:
        print(f"  {bp['start']} ~ {bp['end']} ({bp['duration_bars']} bars, min sharpe={bp['min_sharpe']})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.rolling_monitor <run_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]).expanduser())
