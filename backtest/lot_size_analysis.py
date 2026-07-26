"""Lot-size constraint impact analysis.

Quantifies the capital efficiency loss caused by lot-size rounding
(e.g. A-share 100-share lots, HK 100-share lots).

Usage::

    python -m backtest.lot_size_analysis <run_dir>
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantify_lot_impact(
    trades_csv: Path,
    initial_cash: float,
    lot_size: int = 100,
) -> Dict[str, Any]:
    """Analyze the impact of lot-size rounding on trading performance.

    Reads ``trades.csv`` and estimates:
    - Average rounding loss percentage per trade
    - Number of missed trades (rounded to 0)
    - Capital utilization rate

    Args:
        trades_csv: Path to artifacts/trades.csv.
        initial_cash: Initial capital from config.
        lot_size: Lot size constraint (e.g. 100 for A-shares).

    Returns:
        Dict with impact metrics.
    """
    if not trades_csv.exists():
        return {"error": "trades.csv not found"}

    df = pd.read_csv(trades_csv)
    if df.empty:
        return {"error": "No trades found"}

    # Only analyze buy orders (entry positions)
    buys = df[df["side"] == "buy"].copy()
    if buys.empty:
        return {"n_trades": 0, "error": "No buy trades found"}

    # For each buy trade, estimate the theoretical (unrounded) position size
    # actual_qty is the rounded (lot-constrained) quantity
    # theoretical_qty = actual_qty + rounding_loss (where actual_qty = floor(theoretical / lot_size) * lot_size)
    analysis_rows = []
    for _, row in buys.iterrows():
        actual_qty = row.get("qty", 0)
        price = row.get("price", 0)
        if price <= 0 or actual_qty <= 0:
            continue

        # Estimate theoretical: the engine rounds down to lot_size multiples
        # So actual_qty = floor(theoretical / lot_size) * lot_size
        # The remainder (theoretical % lot_size) is the rounding loss
        # We can estimate the notional of the rounded-away shares:
        # max possible rounding loss = (lot_size - 1) * price
        # average expected loss = (lot_size / 2) * price (uniform distribution assumption)
        # But we can be more precise: if actual_qty > 0,
        # the notional actually invested = actual_qty * price
        # the notional we wanted to invest ≈ (actual_qty + lot_size/2) * price (estimate)
        actual_notional = actual_qty * price
        # The rounding loss is at most (lot_size - 1) shares per trade
        max_loss_shares = lot_size - 1
        max_loss_notional = max_loss_shares * price
        # Average expected loss ≈ lot_size / 2 shares
        avg_loss_shares = lot_size / 2
        avg_loss_notional = avg_loss_shares * price

        # Rounding loss as percentage of actual position
        loss_pct = (avg_loss_notional / actual_notional * 100) if actual_notional > 0 else 0

        analysis_rows.append({
            "timestamp": row.get("timestamp"),
            "code": row.get("code"),
            "actual_qty": actual_qty,
            "price": price,
            "actual_notional": round(actual_notional, 2),
            "avg_loss_notional": round(avg_loss_notional, 2),
            "max_loss_notional": round(max_loss_notional, 2),
            "loss_pct": round(loss_pct, 4),
        })

    if not analysis_rows:
        return {"n_trades": 0, "error": "No analyzable buy trades"}

    analysis_df = pd.DataFrame(analysis_rows)
    n_trades = len(analysis_df)

    # Aggregate metrics
    avg_loss_pct = float(analysis_df["loss_pct"].mean())
    total_avg_loss = float(analysis_df["avg_loss_notional"].sum())
    total_invested = float(analysis_df["actual_notional"].sum())
    capital_utilization = total_invested / initial_cash if initial_cash > 0 else 0

    # Check for missed trades (qty rounded to 0)
    all_trades = pd.read_csv(trades_csv)
    zero_buys = all_trades[(all_trades["side"] == "buy") & (all_trades["qty"] == 0)]
    missed_trades = len(zero_buys)

    return {
        "lot_size": lot_size,
        "n_analyzed_trades": n_trades,
        "avg_rounding_loss_pct": round(avg_loss_pct, 4),
        "total_estimated_rounding_loss": round(total_avg_loss, 2),
        "total_invested_notional": round(total_invested, 2),
        "capital_utilization": round(capital_utilization * 100, 2),
        "missed_trades": missed_trades,
        "max_single_trade_loss_pct": round(float(analysis_df["loss_pct"].max()), 4),
        "trades_with_high_loss": int((analysis_df["loss_pct"] > 5.0).sum()),
    }


def format_lot_impact_text(impact: Dict[str, Any]) -> str:
    """Format lot-size impact as readable text."""
    if "error" in impact:
        return f"手数约束分析: {impact['error']}"

    lines = [
        "=" * 60,
        "手数约束影响分析",
        "=" * 60,
        f"整手大小: {impact['lot_size']} 股",
        f"分析交易数: {impact['n_analyzed_trades']}",
        "",
        f"平均取整损失: {impact['avg_rounding_loss_pct']:.4f}%",
        f"总估算取整损失: ¥{impact['total_estimated_rounding_loss']:,.2f}",
        f"总投入金额: ¥{impact['total_invested_notional']:,.2f}",
        f"资金利用率: {impact['capital_utilization']:.2f}%",
        f"错过交易数(取整为0): {impact['missed_trades']}",
        f"最大单笔损失: {impact['max_single_trade_loss_pct']:.4f}%",
        f"高损失交易(>5%): {impact['trades_with_high_loss']}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(run_dir: Path) -> None:
    """CLI entry: analyze lot-size impact from trades.csv."""
    artifacts = run_dir / "artifacts"
    trades_csv = artifacts / "trades.csv"

    config_path = run_dir / "config.json"
    initial_cash = 1_000_000.0
    lot_size = 100  # Default A-share

    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        initial_cash = cfg.get("initial_cash", initial_cash)
        # Detect lot size from market
        source = cfg.get("source", "")
        codes = cfg.get("codes", [])
        if any(".HK" in c for c in codes) or source == "futu":
            lot_size = 100  # HK also uses lot sizes (varies by stock)
        elif any(".US" in c for c in codes):
            lot_size = 1  # US: no lot constraint

    impact = quantify_lot_impact(trades_csv, initial_cash, lot_size)

    out_path = artifacts / "lot_size_impact.json"
    out_path.write_text(json.dumps(impact, indent=2, ensure_ascii=False), encoding="utf-8")

    print(format_lot_impact_text(impact))
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.lot_size_analysis <run_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]).expanduser())
