"""Rule burden (degrees of freedom) assessment for overfitting risk.

Counts the number of non-default strategy parameters and compares against
the number of trades to estimate overfitting risk.

Reference: Brian Peterson — "Rule burden is a form of overfitting."

Usage::

    python -m backtest.rule_burden <run_dir>
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Default parameter values of StrategyBase.__init__ — used to detect
# which parameters the user explicitly overrode in signal_params.
_STRATEGY_DEFAULTS: Dict[str, Any] = {
    # Indicator params
    "adx_period": 14, "adx_smooth": 6, "atr_period": 14,
    "ema_fast": 12, "ema_slow": 26,
    "sar_accel": 0.02, "sar_max_accel": 0.2,
    "rsi_period": 14, "ma_period": 20, "atr_ma_period": 60,
    # Regime params
    "trend_strength_threshold": 25.0, "trend_strength_grid_max": 20.0,
    "mode_confirm_bars": 1, "mode_cooldown_bars": 1, "osc_confirm_bars": 2,
    "vol_high_thresh": 1.3, "vol_low_thresh": 0.7,
    # Grid params
    "grid_levels": 5, "grid_reset_days": 5, "grid_stop_loss_pct": 0.10,
    # Trend params
    "risk_per_trade": 0.03, "initial_stop_atr_mult": 3.0,
    "trailing_stop_atr_mult": 2.0, "max_pyramid": 4,
    "max_position_ratio": 0.8, "reversal_ema_gap_pct": 0.003,
    "min_hold_bars": 2,
    # Short params
    "allow_short": False, "max_short_ratio": 0.5,
    "short_stop_atr_mult": 2.5, "short_squeeze_rsi": 75.0,
    # Mean reversion
    "mr_enabled": False, "mr_zscore_entry": 2.0, "mr_zscore_exit": 0.5,
    "mr_zscore_stop": 3.0, "mr_rsi_oversold": 30.0,
    "mr_rsi_overbought": 70.0, "mr_max_position_ratio": 0.5,
    # Risk
    "max_drawdown_halt": 0.08, "max_daily_loss": 0.03,
    "max_consecutive_stops": 3,
    # Tracking
    "verbose": False,
}

# Total number of configurable parameters in StrategyBase
TOTAL_PARAMS = len(_STRATEGY_DEFAULTS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_strategy_params(signal_params: Dict[str, Any]) -> int:
    """Count non-default parameters explicitly set by the user.

    Args:
        signal_params: The ``signal_params`` dict from config.json.

    Returns:
        Number of parameters that differ from defaults.
    """
    if not signal_params:
        return 0

    count = 0
    for key, value in signal_params.items():
        if key in ("create_entry_signal",):
            continue  # Skip non-scalar params
        default = _STRATEGY_DEFAULTS.get(key)
        if default is None:
            # Unknown param — count it as custom
            count += 1
        elif value != default:
            count += 1

    return count


def assess_overfit_risk(n_params: int, n_trades: int) -> Dict[str, Any]:
    """Assess overfitting risk based on parameter count and trade count.

    Rule of thumb (empirical):
    - ``trades / params >= 20`` → safe
    - ``10 <= trades / params < 20`` → warning
    - ``trades / params < 10`` → critical

    Args:
        n_params: Number of non-default strategy parameters.
        n_trades: Total number of round-trip trades in the backtest.

    Returns:
        Dict with ``level``, ``ratio``, ``n_params``, ``n_trades``, ``message``.
    """
    if n_params == 0:
        return {
            "level": "safe",
            "ratio": float("inf"),
            "n_params": 0,
            "n_trades": n_trades,
            "message": "使用默认参数，无过拟合风险",
        }

    ratio = n_trades / n_params

    if ratio >= 20:
        level = "safe"
        message = f"每参数 {ratio:.0f} 笔交易，样本量充足"
    elif ratio >= 10:
        level = "warning"
        message = f"每参数 {ratio:.0f} 笔交易，建议减少参数或增加数据量"
    else:
        level = "critical"
        message = f"每参数仅 {ratio:.1f} 笔交易，过拟合风险高！考虑简化策略"

    return {
        "level": level,
        "ratio": round(ratio, 1),
        "n_params": n_params,
        "n_trades": n_trades,
        "total_available_params": TOTAL_PARAMS,
        "message": message,
    }


def format_burden_text(assessment: Dict[str, Any]) -> str:
    """Format rule burden assessment as readable text."""
    level = assessment.get("level", "unknown")
    emoji = {"safe": "[OK]", "warning": "[!!]", "critical": "[!!]"}
    prefix = emoji.get(level, "[?]")

    lines = [
        "=" * 60,
        f"{prefix} 规则负担评估: {level.upper()}",
        "=" * 60,
        f"非默认参数数: {assessment.get('n_params', 0)} / {assessment.get('total_available_params', TOTAL_PARAMS)}",
        f"总交易数: {assessment.get('n_trades', 0)}",
        f"交易/参数比: {assessment.get('ratio', 0):.1f}",
        f"评估: {assessment.get('message', '')}",
        "",
        "参考: trades/params >= 20 为安全, 10-20 为警告, < 10 为危险",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(run_dir: Path) -> None:
    """CLI entry: assess rule burden from config + trades."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(json.dumps({"error": "config.json not found"}))
        sys.exit(1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    signal_params = config.get("signal_params", {})

    # Count trades from trades.csv
    artifacts = run_dir / "artifacts"
    trades_csv = artifacts / "trades.csv"
    n_trades = 0
    if trades_csv.exists():
        import pandas as pd
        df = pd.read_csv(trades_csv)
        # Count round trips (buy + sell pairs)
        n_trades = len(df[df["side"] == "buy"])

    n_params = count_strategy_params(signal_params)
    assessment = assess_overfit_risk(n_params, n_trades)

    # Write to artifacts
    out_path = artifacts / "rule_burden.json"
    if artifacts.exists():
        out_path.write_text(
            json.dumps(assessment, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(format_burden_text(assessment))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.rule_burden <run_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]).expanduser())
