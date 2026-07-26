"""Fill-price stress testing for backtest robustness.

Runs the same backtest under different slippage assumptions to quantify
how sensitive the strategy is to execution-price uncertainty.

Modes:
    - ``base``: current fixed slippage rate (1x)
    - ``optimistic``: half slippage (0.5x)
    - ``worst_case``: double slippage (2x)
    - ``random``: random slippage in [0.5x, 2x] range (N times)

Usage::

    python -m backtest.fill_price_stress <run_dir>

Or in config.json::

    "fill_price_stress": {
        "modes": ["base", "optimistic", "worst_case", "random"],
        "n_random": 5
    }
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

STRESS_MODES: Dict[str, float] = {
    "optimistic": 0.5,
    "base": 1.0,
    "worst_case": 2.0,
}


def run_stress(
    config: Dict[str, Any],
    run_dir: Path,
    modes: List[str] | None = None,
    n_random: int = 5,
) -> Dict[str, Any]:
    """Run fill-price stress test across multiple slippage modes.

    Each mode overrides ``config["slippage_rate"]`` by a multiplier before
    running the backtest.  This avoids modifying the engine's
    ``apply_slippage()`` method.

    Args:
        config: Base backtest config.
        run_dir: Run directory.
        modes: List of stress modes to run.
        n_random: Number of random slippage runs.

    Returns:
        Dict with per-mode metrics and comparison summary.
    """
    from backtest.grid_search import _run_single_backtest

    if modes is None:
        modes = ["base", "optimistic", "worst_case"]

    base_slippage = config.get("slippage_rate", 0.001)
    dates = (config.get("start_date", ""), config.get("end_date", ""))

    results: Dict[str, Dict[str, Any]] = {}

    for mode in modes:
        if mode == "random":
            # Run multiple random slippage values
            random_results = []
            for i in range(n_random):
                rng = np.random.default_rng(seed=42 + i)
                multiplier = float(rng.uniform(0.5, 2.0))
                stress_config = {**config, "slippage_rate": base_slippage * multiplier}
                result = _run_single_backtest(
                    stress_config, run_dir, {},
                    dates[0], dates[1],
                )
                if "error" not in result:
                    random_results.append({
                        "multiplier": round(multiplier, 2),
                        **result,
                    })

            if random_results:
                avg_sharpe = np.mean([r.get("sharpe", 0) for r in random_results])
                avg_return = np.mean([r.get("annual_return", 0) for r in random_results])
                avg_dd = np.mean([r.get("max_drawdown", 0) for r in random_results])
                results["random"] = {
                    "n_runs": len(random_results),
                    "avg_sharpe": round(float(avg_sharpe), 4),
                    "avg_annual_return": round(float(avg_return) * 100, 2),
                    "avg_max_drawdown": round(float(avg_dd) * 100, 2),
                    "runs": random_results,
                }
        else:
            multiplier = STRESS_MODES.get(mode, 1.0)
            stress_config = {**config, "slippage_rate": base_slippage * multiplier}
            result = _run_single_backtest(
                stress_config, run_dir, {},
                dates[0], dates[1],
            )
            if "error" not in result:
                results[mode] = {
                    "multiplier": multiplier,
                    "slippage_rate": round(base_slippage * multiplier, 6),
                    "sharpe": round(result.get("sharpe", 0), 4),
                    "annual_return": round(result.get("annual_return", 0) * 100, 2),
                    "max_drawdown": round(result.get("max_drawdown", 0) * 100, 2),
                    "total_return": round(result.get("total_return", 0) * 100, 2),
                }
            else:
                results[mode] = {"error": result.get("error")}

    return {"modes": results, "base_slippage": base_slippage}


def format_stress_report(stress_result: Dict[str, Any]) -> str:
    """Format stress test results as a readable text report."""
    modes = stress_result.get("modes", {})
    base_slip = stress_result.get("base_slippage", 0)

    lines = [
        "=" * 70,
        "成交价压力测试报告",
        "=" * 70,
        f"基础滑点率: {base_slip:.4%}",
        "",
    ]

    # Table header
    lines.append(f"{'模式':>12} {'乘数':>6} {'夏普比':>8} {'年化收益%':>10} {'最大回撤%':>10} {'总收益%':>10}")
    lines.append("-" * 62)

    for mode in ["optimistic", "base", "worst_case"]:
        if mode in modes and "error" not in modes[mode]:
            m = modes[mode]
            lines.append(
                f"{mode:>12} {m['multiplier']:>6.1f}x "
                f"{m['sharpe']:>8.4f} {m['annual_return']:>10.2f} "
                f"{m['max_drawdown']:>10.2f} {m['total_return']:>10.2f}"
            )

    if "random" in modes and "error" not in modes.get("random", {}):
        r = modes["random"]
        lines.append(
            f"{'random(avg)':>12} {'var':>6} "
            f"{r['avg_sharpe']:>8.4f} {r['avg_annual_return']:>10.2f} "
            f"{r['avg_max_drawdown']:>10.2f} {'-':>10}"
        )

    # Sensitivity: difference between worst and best
    valid_modes = {k: v for k, v in modes.items() if "error" not in v and k != "random"}
    if len(valid_modes) >= 2:
        sharpes = [v.get("sharpe", 0) for v in valid_modes.values()]
        lines.append("")
        lines.append(f"夏普比极差: {max(sharpes) - min(sharpes):.4f} (越小越稳健)")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(run_dir: Path) -> None:
    """CLI entry: run fill-price stress test."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(json.dumps({"error": "config.json not found"}))
        sys.exit(1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    stress_cfg = config.get("fill_price_stress", {})
    modes = stress_cfg.get("modes", ["base", "optimistic", "worst_case"])
    n_random = stress_cfg.get("n_random", 5)

    result = run_stress(config, run_dir, modes=modes, n_random=n_random)

    # Write output
    out_dir = run_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fill_price_stress.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(format_stress_report(result))
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.fill_price_stress <run_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]).expanduser())
