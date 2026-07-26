"""Periodic (annual / monthly) performance breakdown.

Splits the equity curve into calendar-year and calendar-month segments
and computes per-period metrics.  Designed to be called from
``BaseEngine.run_backtest()`` as an opt-in enhancement.

Usage (CLI)::

    python -m backtest.periodic_metrics <run_dir>
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

def annual_metrics(
    equity_curve: pd.Series,
    trades: list,
    initial_cash: float,
    bars_per_year: float | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Return per-calendar-year metrics.

    Returns a dict keyed by year string, e.g. ``{"2023": {...}, "2024": {...}}``.
    Each value contains: ``return_pct``, ``sharpe``, ``max_drawdown``,
    ``trade_count``, ``volatility``.
    """
    from backtest.metrics import calc_metrics

    if equity_curve.empty:
        return {}

    ret_series = equity_curve.pct_change().fillna(0.0)
    years = equity_curve.index.year.unique()

    result: Dict[str, Dict[str, Any]] = {}
    for year in sorted(years):
        mask = equity_curve.index.year == year
        yr_equity = equity_curve.loc[mask]
        yr_ret = ret_series.loc[mask]

        if len(yr_equity) < 2:
            continue

        # Year start equity = first value of this year's segment
        yr_start_equity = float(yr_equity.iloc[0])
        yr_end_equity = float(yr_equity.iloc[-1])
        yr_return = (yr_end_equity - yr_start_equity) / yr_start_equity if yr_start_equity else 0.0

        # Annualized volatility & Sharpe within the year
        yr_vol = float(yr_ret.std()) * np.sqrt(bars_per_year or 252) if len(yr_ret) > 1 else 0.0
        yr_sharpe = (float(yr_ret.mean()) * (bars_per_year or 252) - 0.0) / yr_vol if yr_vol > 1e-8 else 0.0

        # Max drawdown within the year
        peak = yr_equity.cummax()
        dd = (yr_equity - peak) / peak.replace(0, 1)
        yr_max_dd = float(dd.min())

        # Trade count for this year
        yr_trades = [
            t for t in trades
            if hasattr(t, "entry_time") and t.entry_time.year == year
        ]

        result[str(year)] = {
            "return_pct": round(yr_return * 100, 2),
            "sharpe": round(yr_sharpe, 4),
            "max_drawdown": round(yr_max_dd * 100, 2),
            "volatility": round(yr_vol * 100, 2),
            "trade_count": len(yr_trades),
            "start_equity": round(yr_start_equity, 2),
            "end_equity": round(yr_end_equity, 2),
        }

    return result


def monthly_summary(
    equity_curve: pd.Series,
    trades: list,
) -> Dict[str, Any]:
    """Return monthly statistics summary.

    Returns dict with: ``monthly_returns`` (list of {year, month, return_pct}),
    ``win_rate``, ``avg_win``, ``avg_loss``, ``profit_factor``,
    ``max_consecutive_wins``, ``max_consecutive_losses``.
    """
    if equity_curve.empty:
        return {}

    # Monthly returns from equity curve
    monthly_eq = equity_curve.resample("ME").last().dropna()
    monthly_ret = monthly_eq.pct_change().dropna()

    rows: List[Dict[str, Any]] = []
    for ts, ret in monthly_ret.items():
        rows.append({
            "year": ts.year,
            "month": ts.month,
            "label": f"{ts.year}-{ts.month:02d}",
            "return_pct": round(float(ret) * 100, 2),
        })

    if not rows:
        return {"monthly_returns": [], "n_months": 0}

    rets = np.array([r["return_pct"] for r in rows])
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    # Consecutive wins/losses
    max_consec_wins = _max_consecutive(rets > 0)
    max_consec_losses = _max_consecutive(rets < 0)

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0

    return {
        "monthly_returns": rows,
        "n_months": len(rows),
        "win_rate": round(len(wins) / len(rets) * 100, 1) if len(rets) else 0.0,
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 1e-8 else float("inf"),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "best_month_pct": round(float(rets.max()), 2) if len(rets) else 0.0,
        "worst_month_pct": round(float(rets.min()), 2) if len(rets) else 0.0,
    }


def format_periodic_text(
    annual: Dict[str, Dict[str, Any]],
    monthly: Dict[str, Any],
) -> str:
    """Format periodic metrics as a human-readable text report."""
    lines = [
        "=" * 70,
        "年度/月度分段报告",
        "=" * 70,
        "",
        "── 年度分解 ──",
    ]

    if annual:
        lines.append(f"{'年份':>6} {'收益率%':>10} {'夏普比':>8} {'最大回撤%':>10} {'波动率%':>8} {'交易数':>6}")
        lines.append("-" * 56)
        for year, m in sorted(annual.items()):
            lines.append(
                f"{year:>6} {m['return_pct']:>10.2f} {m['sharpe']:>8.2f} "
                f"{m['max_drawdown']:>10.2f} {m['volatility']:>8.2f} {m['trade_count']:>6}"
            )
    else:
        lines.append("  (无年度数据)")

    lines.append("")
    lines.append("── 月度统计 ──")

    n_months = monthly.get("n_months", 0)
    if n_months:
        lines.append(f"总月数: {n_months}")
        lines.append(f"月度胜率: {monthly.get('win_rate', 0):.1f}%")
        lines.append(f"平均盈利月: {monthly.get('avg_win_pct', 0):.2f}%")
        lines.append(f"平均亏损月: {monthly.get('avg_loss_pct', 0):.2f}%")
        pf = monthly.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        lines.append(f"盈亏比(Profit Factor): {pf_str}")
        lines.append(f"最大连续盈利月: {monthly.get('max_consecutive_wins', 0)}")
        lines.append(f"最大连续亏损月: {monthly.get('max_consecutive_losses', 0)}")
        lines.append(f"最佳月: {monthly.get('best_month_pct', 0):.2f}%")
        lines.append(f"最差月: {monthly.get('worst_month_pct', 0):.2f}%")
    else:
        lines.append("  (无月度数据)")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_consecutive(mask: np.ndarray) -> int:
    """Count the maximum consecutive True values in a boolean array."""
    if len(mask) == 0:
        return 0
    max_count = 0
    current = 0
    for val in mask:
        if val:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(run_dir: Path) -> None:
    """CLI entry: read equity.csv + trades.csv from artifacts, print report."""
    artifacts = run_dir / "artifacts"
    eq_path = artifacts / "equity.csv"
    if not eq_path.exists():
        print(json.dumps({"error": "equity.csv not found in artifacts"}))
        sys.exit(1)

    eq_df = pd.read_csv(eq_path, index_col=0, parse_dates=True)
    equity_curve = eq_df["equity"]

    initial_cash = float(equity_curve.iloc[0]) if len(equity_curve) else 1_000_000.0

    # Detect bars_per_year from config
    config_path = run_dir / "config.json"
    bars_per_year = None
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        source = cfg.get("source", "")
        interval = cfg.get("interval", "1D")
        if interval == "1D" and source in ("tushare", "akshare", "futu"):
            bars_per_year = 244
        elif interval == "1D":
            bars_per_year = 252

    annual = annual_metrics(equity_curve, [], initial_cash, bars_per_year)
    monthly = monthly_summary(equity_curve, [])
    print(format_periodic_text(annual, monthly))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.periodic_metrics <run_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]).expanduser())
