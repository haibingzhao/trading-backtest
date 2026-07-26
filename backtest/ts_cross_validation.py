"""Time series cross-validation for parameter robustness.

Multi-fold cross-validation with embargo periods to evaluate parameter selection
reliability. Reports mean/std of OOS metrics and parameter stability across folds.

Usage:
    python -m backtest.ts_cross_validation <run_dir>

Or configure in config.json:
    "time_series_cv": {
        "param_grid": {"ema_fast": [10, 15, 20], "ema_slow": [21, 26, 50]},
        "n_splits": 5,
        "embargo_bars": 5,
        "expanding": true,
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
class CVSplitResult:
    """Single cross-validation split result."""

    split_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    in_sample_metric: float
    oos_metric: float


@dataclass(frozen=True)
class CVResult:
    """Cross-validation result summary."""

    method: str  # "expanding" or "rolling"
    n_splits: int
    splits: List[CVSplitResult]
    mean_oos_metric: float
    std_oos_metric: float
    param_stability: Dict[str, Dict[Any, int]]


def time_series_cv(
    config: Dict[str, Any],
    run_dir: Path,
    param_grid: Dict[str, List[Any]],
    n_splits: int = 5,
    embargo_bars: int = 5,
    expanding: bool = True,
    objective: str = "sharpe",
    max_workers: int | None = None,
) -> Dict[str, Any]:
    """Run time series cross-validation.

    For each fold:
    1. Split data into train and test with embargo period
    2. Run grid search on train to find best params
    3. Evaluate best params on test
    4. Aggregate results across folds

    Args:
        config: Base backtest config.
        run_dir: Run directory.
        param_grid: Parameter grid specification.
        n_splits: Number of CV folds.
        embargo_bars: Number of bars to skip between train and test (avoid leakage).
        expanding: If True, use expanding train window; if False, use rolling.
        objective: Metric to optimize.

    Returns:
        Dict with splits, summary, and parameter stability.
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
        raise ValueError("No data fetched for time series CV")

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

    # Calculate fold boundaries
    # Each fold has a test set of roughly equal size
    test_size = n_dates // (n_splits + 1)
    min_train_size = test_size  # Minimum train size for rolling

    splits: List[CVSplitResult] = []

    logger.info(
        "Time series CV: %d splits, expanding=%s, embargo=%d bars",
        n_splits, expanding, embargo_bars
    )

    for i in range(n_splits):
        # Test window
        test_start_idx = (i + 1) * test_size
        test_end_idx = min((i + 2) * test_size, n_dates) if i < n_splits - 1 else n_dates

        # Train window
        if expanding:
            train_start_idx = 0
        else:
            # Rolling: train size = test_size * 2 or remaining data
            train_start_idx = max(0, test_start_idx - test_size * 2 - embargo_bars)

        # Apply embargo: skip some bars between train and test
        train_end_idx = test_start_idx - embargo_bars

        if train_end_idx <= train_start_idx:
            logger.warning("Fold %d: insufficient train data after embargo, skipping", i + 1)
            continue

        if test_end_idx <= test_start_idx:
            logger.warning("Fold %d: insufficient test data, skipping", i + 1)
            continue

        train_dates = dates[train_start_idx:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        if len(train_dates) < 30 or len(test_dates) < 10:
            logger.warning("Fold %d: insufficient bars (train=%d, test=%d), skipping",
                          i + 1, len(train_dates), len(test_dates))
            continue

        train_start = str(train_dates[0].date())
        train_end = str(train_dates[-1].date())
        test_start = str(test_dates[0].date())
        test_end = str(test_dates[-1].date())

        logger.info(
            "Fold %d: train=%s to %s (%d bars), test=%s to %s (%d bars)",
            i + 1, train_start, train_end, len(train_dates),
            test_start, test_end, len(test_dates)
        )

        # Grid search on train period (parallel within each fold)
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
            logger.warning("Fold %d: no valid params found, skipping", i + 1)
            continue

        # Evaluate best params on test period
        oos_result = _run_single_backtest(
            config, run_dir, best_params, test_start, test_end
        )

        if "error" in oos_result:
            logger.warning("Fold %d: OOS backtest failed: %s", i + 1, oos_result.get("error"))
            continue

        oos_metric = oos_result.get(objective, 0.0)

        splits.append(CVSplitResult(
            split_idx=i + 1,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            best_params=best_params,
            in_sample_metric=best_is_metric,
            oos_metric=oos_metric,
        ))

    # Summary statistics
    if splits:
        oos_metrics = [s.oos_metric for s in splits]
        mean_oos = float(np.mean(oos_metrics))
        std_oos = float(np.std(oos_metrics))

        # Parameter stability
        param_counts: Dict[str, Dict[Any, int]] = {}
        for s in splits:
            for k, v in s.best_params.items():
                if k not in param_counts:
                    param_counts[k] = {}
                param_counts[k][v] = param_counts[k].get(v, 0) + 1

        method = "expanding" if expanding else "rolling"
    else:
        mean_oos = 0.0
        std_oos = 0.0
        param_counts = {}
        method = "expanding" if expanding else "rolling"

    # Serialize splits
    splits_list = [
        {
            "split": s.split_idx,
            "train_start": s.train_start,
            "train_end": s.train_end,
            "test_start": s.test_start,
            "test_end": s.test_end,
            "best_params": s.best_params,
            "in_sample_metric": round(s.in_sample_metric, 4),
            "oos_metric": round(s.oos_metric, 4),
        }
        for s in splits
    ]

    return {
        "method": method,
        "n_splits": len(splits),
        "splits": splits_list,
        "mean_oos_metric": round(mean_oos, 4),
        "std_oos_metric": round(std_oos, 4),
        "param_stability": param_counts,
    }


def format_cv_report(result: Dict[str, Any]) -> str:
    """Format CV results as a readable report."""
    splits = result.get("splits", [])
    method = result.get("method", "expanding")
    mean_oos = result.get("mean_oos_metric", 0.0)
    std_oos = result.get("std_oos_metric", 0.0)

    lines = [
        "=" * 70,
        f"时间序列交叉验证结果 ({method} 窗口)",
        "=" * 70,
        f"折数: {result.get('n_splits', 0)}",
        f"样本外指标均值: {mean_oos:.4f} ± {std_oos:.4f}",
        "",
        "各折详情:",
        "-" * 70,
    ]

    for s in splits:
        lines.append(f"折 {s['split']}:")
        lines.append(f"  训练: {s['train_start']} ~ {s['train_end']}")
        lines.append(f"  测试: {s['test_start']} ~ {s['test_end']}")
        lines.append(f"  最优参数: {s['best_params']}")
        lines.append(f"  样本内: {s['in_sample_metric']:.4f}, 样本外: {s['oos_metric']:.4f}")
        lines.append("")

    # Parameter stability
    param_stability = result.get("param_stability", {})
    if param_stability:
        lines.append("参数稳定性:")
        lines.append("-" * 70)
        for param, counts in param_stability.items():
            lines.append(f"  {param}:")
            for value, count in sorted(counts.items(), key=lambda x: -x[1]):
                pct = count / len(splits) * 100 if splits else 0
                lines.append(f"    {value}: {count} 次 ({pct:.0f}%)")

    lines.append("=" * 70)
    return "\n".join(lines)


def main(run_dir: Path) -> None:
    """CLI entry point for time series cross-validation."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(json.dumps({"error": "config.json not found"}))
        sys.exit(1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    cv_cfg = config.get("time_series_cv", {})

    if not cv_cfg:
        print(json.dumps({"error": "No time_series_cv config found in config.json"}))
        sys.exit(1)

    param_grid = cv_cfg.get("param_grid", {})
    if not param_grid:
        print(json.dumps({"error": "No param_grid specified"}))
        sys.exit(1)

    n_splits = cv_cfg.get("n_splits", 5)
    embargo_bars = cv_cfg.get("embargo_bars", 5)
    expanding = cv_cfg.get("expanding", True)
    objective = cv_cfg.get("objective", "sharpe")
    max_workers = cv_cfg.get("max_workers", min(os.cpu_count() or 4, 8))

    try:
        result = time_series_cv(
            config=config,
            run_dir=run_dir,
            param_grid=param_grid,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            expanding=expanding,
            objective=objective,
            max_workers=max_workers,
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    # Write results
    out_dir = run_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "ts_cross_validation.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print report
    print(format_cv_report(result))
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.ts_cross_validation <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser()
    main(run_dir)
