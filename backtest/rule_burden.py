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

# --- Parameter classification for weighted rule burden ---
# Operational params: purely logistical, zero overfitting contribution.
_OPERATIONAL_PARAMS = frozenset({"verbose"})

# Structural params: boolean switches or design choices that enable/disable
# a module. They represent a single binary decision, not a continuous search.
_STRUCTURAL_PARAMS = frozenset({
    "allow_short", "mr_enabled",
})

# Risk/guard params: safety thresholds typically set by experience, not
# grid-searched. Lower overfitting contribution than signal params.
_RISK_PARAMS = frozenset({
    "max_drawdown_halt", "max_daily_loss", "max_consecutive_stops",
    "max_position_ratio", "max_short_ratio", "mr_max_position_ratio",
    "grid_stop_loss_pct",
})

# Weights for effective degrees of freedom
_WEIGHT_STRUCTURAL = 0.5   # binary design choice
_WEIGHT_RISK = 0.5         # experience-based safety threshold
_WEIGHT_OPTIMIZED = 1.0    # continuous param likely grid-searched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_strategy_params(signal_params: Dict[str, Any]) -> int:
    """Count non-default parameters explicitly set by the user (raw count).

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


def effective_degrees_of_freedom(signal_params: Dict[str, Any]) -> float:
    """Compute weighted effective degrees of freedom for overfitting assessment.

    Not all parameters contribute equally to overfitting risk:
    - Operational params (verbose): 0 weight
    - Structural switches (allow_short, mr_enabled): 0.5 weight
    - Risk/guard thresholds (max_drawdown_halt, etc.): 0.5 weight
    - Signal/optimization params (ema_fast, trailing_stop_atr_mult, etc.): 1.0

    Args:
        signal_params: The ``signal_params`` dict from config.json.

    Returns:
        Weighted effective degrees of freedom (float).
    """
    if not signal_params:
        return 0.0

    dof = 0.0
    for key, value in signal_params.items():
        if key in ("create_entry_signal",):
            continue
        if key in _OPERATIONAL_PARAMS:
            continue
        default = _STRATEGY_DEFAULTS.get(key)
        if default is not None and value == default:
            continue  # Not overridden

        # Classify and weight
        if key in _STRUCTURAL_PARAMS:
            dof += _WEIGHT_STRUCTURAL
        elif key in _RISK_PARAMS:
            dof += _WEIGHT_RISK
        else:
            dof += _WEIGHT_OPTIMIZED

    return dof


def assess_overfit_risk(
    n_params: int,
    n_trades: int,
    *,
    effective_dof: float | None = None,
    backtest_years: float | None = None,
    market_return: float | None = None,
    allow_short: bool = False,
) -> Dict[str, Any]:
    """Assess overfitting risk based on parameter count and trade count.

    Rule of thumb (empirical, based on effective degrees of freedom):
    - ``trades / dof >= 20`` → safe
    - ``10 <= trades / dof < 20`` → warning
    - ``trades / dof < 10`` → critical

    When ``effective_dof`` is provided, it is used for the ratio calculation
    instead of raw ``n_params``. This accounts for the fact that boolean
    switches and risk thresholds contribute less to overfitting than
    continuously optimized signal parameters.

    Context-aware adjustments:
    - If the underlying dropped > 15% over the period and shorting is
      disabled, low trade count is expected — severity is downgraded one
      level and a contextual note is added.

    Args:
        n_params: Raw number of non-default strategy parameters.
        n_trades: Total number of round-trip trades in the backtest.
        effective_dof: Weighted effective degrees of freedom (optional).
            If None, falls back to ``n_params``.
        backtest_years: Backtest duration in years (optional).
        market_return: Period return of the underlying, e.g. -0.25 means
            the stock dropped 25% (optional).
        allow_short: Whether the strategy allows short selling.

    Returns:
        Dict with ``level``, ``ratio``, ``n_params``, ``n_trades``, ``message``.
    """
    dof = effective_dof if effective_dof is not None else float(n_params)

    if dof == 0:
        return {
            "level": "safe",
            "ratio": float("inf"),
            "n_params": n_params,
            "effective_dof": 0.0,
            "n_trades": n_trades,
            "message": "使用默认参数，无过拟合风险",
        }

    ratio = n_trades / dof

    if ratio >= 20:
        level = "safe"
        message = f"每有效自由度 {ratio:.0f} 笔交易，样本量充足"
    elif ratio >= 10:
        level = "warning"
        message = f"每有效自由度 {ratio:.0f} 笔交易，统计显著性有限，若因市场无机会导致交易少则属正常"
    else:
        level = "critical"
        message = f"每有效自由度仅 {ratio:.1f} 笔交易，统计显著性有限，考虑增加数据量或简化参数"

    # --- Context-aware downgrade ---
    detail: str | None = None
    bear_no_short = (
        market_return is not None
        and market_return < -0.15
        and not allow_short
    )
    if bear_no_short:
        detail = "单边下跌市，低交易数属正常"
        # Downgrade severity one level
        if level == "critical":
            level = "warning"
            message = f"每有效自由度仅 {ratio:.1f} 笔交易，但标的区间跌幅 {market_return:.0%} 且不可做空，低交易数属正常"
        elif level == "warning":
            level = "safe"
            message = f"每有效自由度 {ratio:.0f} 笔交易，标的区间跌幅 {market_return:.0%} 且不可做空，低交易数属正常"

    result: Dict[str, Any] = {
        "level": level,
        "ratio": round(ratio, 1),
        "n_params": n_params,
        "effective_dof": round(dof, 1),
        "n_trades": n_trades,
        "total_available_params": TOTAL_PARAMS,
        "message": message,
    }
    if detail:
        result["detail"] = detail
    if backtest_years is not None:
        result["backtest_years"] = round(backtest_years, 2)
    if market_return is not None:
        result["market_return"] = round(market_return, 4)
    return result


def format_burden_text(assessment: Dict[str, Any]) -> str:
    """Format rule burden assessment as readable text."""
    level = assessment.get("level", "unknown")
    emoji = {"safe": "[OK]", "warning": "[!!]", "critical": "[!!]"}
    prefix = emoji.get(level, "[?]")

    n_params = assessment.get("n_params", 0)
    eff_dof = assessment.get("effective_dof", n_params)
    dof_note = f"（加权自由度 {eff_dof:.1f}）" if eff_dof != n_params else ""

    lines = [
        "=" * 60,
        f"{prefix} 规则负担评估: {level.upper()}",
        "=" * 60,
        f"非默认参数数: {n_params} / {assessment.get('total_available_params', TOTAL_PARAMS)}{dof_note}",
        f"总交易数: {assessment.get('n_trades', 0)}",
        f"交易/有效自由度比: {assessment.get('ratio', 0):.1f}",
        f"评估: {assessment.get('message', '')}",
    ]
    if assessment.get("detail"):
        lines.append(f"市场背景: {assessment['detail']}")
    if assessment.get("backtest_years") is not None:
        lines.append(f"回测时长: {assessment['backtest_years']:.2f} 年")
    if assessment.get("market_return") is not None:
        lines.append(f"标的区间收益: {assessment['market_return']:.2%}")
    lines.extend([
        "",
        "参考: trades/有效自由度 >= 20 为安全, 10-20 为警告, < 10 为危险",
        "=" * 60,
    ])
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
    eff_dof = effective_degrees_of_freedom(signal_params)
    assessment = assess_overfit_risk(n_params, n_trades, effective_dof=eff_dof)

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
