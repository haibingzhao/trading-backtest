"""Walk-Forward optimization analysis.

Simulates periodic re-optimization to test parameter robustness over time.
Supports both Rolling (fixed train window) and Anchored (expanding train window) modes.

Usage:
    python -m backtest.walk_forward_opt <run_dir>

Or configure in config.json:
    "walk_forward_opt": {
        "param_grid": {"ema_fast": [10, 15, 20], "ema_slow": [21, 26, 50]},
        "n_splits": 5,
        "anchored": false,
        "objective": "sharpe"
    }
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtest.grid_search import (
    _run_single_backtest,
    generate_param_combinations,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowResult:
    """Single walk-forward window result."""

    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    in_sample_metric: float
    oos_metric: float
    degradation: float  # (is - oos) / |is|


def walk_forward_optimization(
    config: Dict[str, Any],
    run_dir: Path,
    param_grid: Dict[str, List[Any]],
    n_splits: int = 5,
    anchored: bool = False,
    objective: str = "sharpe",
    max_workers: int | None = None,
) -> Dict[str, Any]:
    """Run walk-forward optimization analysis.

    For each window:
    1. Split data into train (IS) and test (OOS) periods
    2. Run grid search on train period to find best params
    3. Evaluate best params on test period
    4. Record IS/OOS metrics and degradation

    Args:
        config: Base backtest config.
        run_dir: Run directory.
        param_grid: Parameter grid specification.
        n_splits: Number of walk-forward windows.
        anchored: If True, use expanding train window; if False, use rolling.
        objective: Metric to optimize.

    Returns:
        Dict with windows, summary stats, and parameter stability.
    """
    # Get full date range
    start_date = config.get("start_date", "")
    end_date = config.get("end_date", "")

    # Load data to get dates
    from backtest.loaders.registry import resolve_loader
    from backtest.runner import _fetch_auto

    codes = config.get("codes", [])
    source = config.get("source", "tushare")
    interval = config.get("interval", "1D")

    if source == "auto":
        data_map = _fetch_auto(codes, config, interval)
    else:
        LoaderCls = resolve_loader(source)
        loader = LoaderCls()
        data_map = loader.fetch(codes, start_date, end_date, interval=interval)

    if not data_map:
        raise ValueError("No data fetched for walk-forward analysis")

    # Get common date range
    all_dates = set()
    for df in data_map.values():
        if "trade_date" in df.index.names:
            all_dates.update(df.index.tolist())
        elif "trade_date" in df.columns:
            all_dates.update(df["trade_date"].tolist())

    if not all_dates:
        raise ValueError("No dates found in data")

    dates = pd.DatetimeIndex(sorted(all_dates))
    n_dates = len(dates)

    if n_dates < n_splits * 20:
        raise ValueError(
            f"Insufficient data: {n_dates} bars for {n_splits} splits. "
            "Need at least 20 bars per split."
        )

    # Calculate window boundaries
    window_size = n_dates // n_splits
    windows: List[WindowResult] = []

    logger.info(
        "Walk-forward optimization: %d splits, anchored=%s, %d param combinations",
        n_splits, anchored, len(list(generate_param_combinations(param_grid)))
    )

    for i in range(n_splits):
        # Test window: last portion of data
        test_start_idx = (i + 1) * window_size
        test_end_idx = min((i + 2) * window_size, n_dates) if i < n_splits - 1 else n_dates

        # Train window
        if anchored:
            train_start_idx = 0
        else:
            # Rolling: fixed-size train window before test
            train_start_idx = max(0, test_start_idx - window_size * 2)

        train_end_idx = test_start_idx

        if train_end_idx - train_start_idx < 30:
            logger.warning("Window %d: insufficient train data, skipping", i + 1)
            continue

        if test_end_idx - test_start_idx < 10:
            logger.warning("Window %d: insufficient test data, skipping", i + 1)
            continue

        train_dates = dates[train_start_idx:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        train_start = str(train_dates[0].date())
        train_end = str(train_dates[-1].date())
        test_start = str(test_dates[0].date())
        test_end = str(test_dates[-1].date())

        logger.info(
            "Window %d: train=%s to %s (%d bars), test=%s to %s (%d bars)",
            i + 1, train_start, train_end, len(train_dates),
            test_start, test_end, len(test_dates)
        )

        # Grid search on train period (parallel within each window)
        param_combos = list(generate_param_combinations(param_grid))
        best_params = None
        best_is_metric = -np.inf

        if max_workers is not None and max_workers > 1 and len(param_combos) > 1:
            # Parallel execution
            mp_ctx = multiprocessing.get_context("fork") if sys.platform == "darwin" else None
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp_ctx,
            ) as executor:
                futures = {
                    executor.submit(
                        _run_single_backtest,
                        config, run_dir, params, train_start, train_end,
                    ): params
                    for params in param_combos
                }
                for future in futures:
                    result = future.result()
                    if "error" in result:
                        continue
                    metric = result.get(objective, 0.0)
                    if metric > best_is_metric:
                        best_is_metric = metric
                        best_params = futures[future]
        else:
            # Serial execution (default, backward-compatible)
            for params in param_combos:
                result = _run_single_backtest(
                    config, run_dir, params, train_start, train_end
                )
                if "error" in result:
                    continue

                metric = result.get(objective, 0.0)
                if metric > best_is_metric:
                    best_is_metric = metric
                    best_params = params

        if best_params is None:
            logger.warning("Window %d: no valid params found, skipping", i + 1)
            continue

        # Evaluate best params on test period
        oos_result = _run_single_backtest(
            config, run_dir, best_params, test_start, test_end
        )

        if "error" in oos_result:
            logger.warning("Window %d: OOS backtest failed: %s", i + 1, oos_result.get("error"))
            continue

        oos_metric = oos_result.get(objective, 0.0)

        # Calculate degradation
        if abs(best_is_metric) > 1e-6:
            degradation = (best_is_metric - oos_metric) / abs(best_is_metric)
        else:
            degradation = 0.0

        windows.append(WindowResult(
            window_idx=i + 1,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            best_params=best_params,
            in_sample_metric=best_is_metric,
            oos_metric=oos_metric,
            degradation=degradation,
        ))

    # Summary statistics
    if windows:
        is_metrics = [w.in_sample_metric for w in windows]
        oos_metrics = [w.oos_metric for w in windows]
        degradations = [w.degradation for w in windows]

        # Parameter stability: how often each param value is selected
        param_counts: Dict[str, Dict[Any, int]] = {}
        for w in windows:
            for k, v in w.best_params.items():
                if k not in param_counts:
                    param_counts[k] = {}
                param_counts[k][v] = param_counts[k].get(v, 0) + 1

        summary = {
            "n_windows": len(windows),
            "mean_is_metric": float(np.mean(is_metrics)),
            "mean_oos_metric": float(np.mean(oos_metrics)),
            "mean_degradation": float(np.mean(degradations)),
            "std_degradation": float(np.std(degradations)),
            "param_stability": param_counts,
        }
    else:
        summary = {"n_windows": 0, "error": "No valid windows found"}

    # Serialize windows
    windows_list = [
        {
            "window": w.window_idx,
            "train_start": w.train_start,
            "train_end": w.train_end,
            "test_start": w.test_start,
            "test_end": w.test_end,
            "best_params": w.best_params,
            "in_sample_metric": round(w.in_sample_metric, 4),
            "oos_metric": round(w.oos_metric, 4),
            "degradation": round(w.degradation, 4),
        }
        for w in windows
    ]

    return {"windows": windows_list, "summary": summary}


def format_walk_forward_report(result: Dict[str, Any]) -> str:
    """Format walk-forward results as a readable report."""
    windows = result.get("windows", [])
    summary = result.get("summary", {})

    lines = [
        "=" * 70,
        "Walk-Forward 优化分析结果",
        "=" * 70,
        f"窗口数: {summary.get('n_windows', 0)}",
        f"平均样本内指标: {summary.get('mean_is_metric', 0.0):.4f}",
        f"平均样本外指标: {summary.get('mean_oos_metric', 0.0):.4f}",
        f"平均衰减率: {summary.get('mean_degradation', 0.0):.2%} ± {summary.get('std_degradation', 0.0):.2%}",
        "",
        "各窗口详情:",
        "-" * 70,
    ]

    for w in windows:
        lines.append(f"窗口 {w['window']}:")
        lines.append(f"  训练: {w['train_start']} ~ {w['train_end']}")
        lines.append(f"  测试: {w['test_start']} ~ {w['test_end']}")
        lines.append(f"  最优参数: {w['best_params']}")
        lines.append(f"  样本内: {w['in_sample_metric']:.4f}, 样本外: {w['oos_metric']:.4f}")
        lines.append(f"  衰减: {w['degradation']:.2%}")
        lines.append("")

    # Parameter stability
    param_stability = summary.get("param_stability", {})
    if param_stability:
        lines.append("参数稳定性:")
        lines.append("-" * 70)
        for param, counts in param_stability.items():
            lines.append(f"  {param}:")
            for value, count in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {value}: {count} 次")

    lines.append("=" * 70)
    return "\n".join(lines)


def main(run_dir: Path) -> None:
    """CLI entry point for walk-forward optimization."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(json.dumps({"error": "config.json not found"}))
        sys.exit(1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    wfo_cfg = config.get("walk_forward_opt", {})

    if not wfo_cfg:
        print(json.dumps({"error": "No walk_forward_opt config found in config.json"}))
        sys.exit(1)

    param_grid = wfo_cfg.get("param_grid", {})
    if not param_grid:
        print(json.dumps({"error": "No param_grid specified"}))
        sys.exit(1)

    n_splits = wfo_cfg.get("n_splits", 5)
    anchored = wfo_cfg.get("anchored", False)
    objective = wfo_cfg.get("objective", "sharpe")
    max_workers = wfo_cfg.get("max_workers", min(os.cpu_count() or 4, 8))

    try:
        result = walk_forward_optimization(
            config=config,
            run_dir=run_dir,
            param_grid=param_grid,
            n_splits=n_splits,
            anchored=anchored,
            objective=objective,
            max_workers=max_workers,
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    # Write results
    out_dir = run_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "walk_forward_opt.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print report
    print(format_walk_forward_report(result))
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.walk_forward_opt <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser()
    main(run_dir)
