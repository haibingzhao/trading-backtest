"""Parameter grid search with in-sample/out-of-sample validation.

Scans parameter space and evaluates overfitting by comparing IS vs OOS performance.
Supports parallel execution via ProcessPoolExecutor.

Usage:
    python -m backtest.grid_search <run_dir>
    
Or configure in config.json:
    "grid_search": {
        "param_grid": {"ema_fast": [8, 12, 20], "ema_slow": [21, 26, 50]},
        "split_ratio": 0.7,
        "objective": "sharpe",
        "max_workers": 4
    }
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridSearchResult:
    """Single parameter combination result."""

    params: Dict[str, Any]
    is_metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    overfit_score: float  # 1 - oos_sharpe / is_sharpe (higher = more overfit)


@dataclass(frozen=True)
class GridSearchSummary:
    """Grid search summary."""

    results: List[GridSearchResult]
    best_params: Dict[str, Any]
    best_is_sharpe: float
    best_oos_sharpe: float
    best_overfit_score: float
    total_combinations: int
    elapsed_seconds: float


def generate_param_combinations(param_grid: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    """Generate parameter combinations from grid specification.

    Args:
        param_grid: Dict mapping param names to lists of values.

    Yields:
        Dict of parameter combinations.
    """
    if not param_grid:
        yield {}
        return

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def count_combinations(param_grid: Dict[str, List[Any]]) -> int:
    """Count total parameter combinations."""
    if not param_grid:
        return 1
    count = 1
    for values in param_grid.values():
        count *= len(values)
    return count


def split_dates_is_oos(
    dates: pd.DatetimeIndex,
    split_ratio: float = 0.7,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split dates into in-sample and out-of-sample periods.

    Args:
        dates: Full date index.
        split_ratio: Ratio of dates for IS (e.g., 0.7 = 70% IS, 30% OOS).

    Returns:
        (is_dates, oos_dates) tuple.
    """
    n = len(dates)
    split_idx = int(n * split_ratio)
    return dates[:split_idx], dates[split_idx:]


def _run_single_backtest(
    config: Dict[str, Any],
    run_dir: Path,
    signal_params: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Run a single backtest with given parameters.

    This function is designed to be called by ProcessPoolExecutor.
    Returns only scalar metrics to minimize serialization overhead.

    Args:
        config: Base backtest config.
        run_dir: Run directory.
        signal_params: Signal engine parameters to override.
        start_date: Start date for this backtest.
        end_date: End date for this backtest.

    Returns:
        Dict with metrics (sharpe, total_return, max_drawdown, etc.).
    """
    import importlib.util
    import sys

    # Load signal engine module
    signal_path = run_dir / "code" / "signal_engine.py"
    spec = importlib.util.spec_from_file_location("signal_engine_grid", signal_path)
    if spec is None or spec.loader is None:
        return {"error": f"Cannot load signal engine from {signal_path}"}
    module = importlib.util.module_from_spec(spec)
    sys.modules["signal_engine_grid"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"error": f"Failed to load signal engine: {e}"}

    engine_cls = getattr(module, "SignalEngine", None)
    if engine_cls is None:
        return {"error": "SignalEngine class not found"}

    # Create modified config
    test_config = config.copy()
    test_config["start_date"] = start_date
    test_config["end_date"] = end_date
    test_config["signal_params"] = signal_params

    # Create signal engine with params
    try:
        signal_engine = engine_cls(**signal_params) if signal_params else engine_cls()
    except Exception as e:
        return {"error": f"Failed to create signal engine: {e}"}

    # Create loader and engine
    from backtest.loaders.registry import resolve_loader
    from backtest.runner import _detect_market, _create_market_engine, _AutoLoader
    from backtest.metrics import calc_bars_per_year

    codes = config.get("codes", [])
    source = config.get("source", "tushare")
    interval = config.get("interval", "1D")

    # Load data
    try:
        if source == "auto":
            from backtest.runner import _fetch_auto
            data_map = _fetch_auto(codes, test_config, interval)
            loader = _AutoLoader(data_map)
        else:
            LoaderCls = resolve_loader(source)
            loader = LoaderCls()
            data_map = loader.fetch(
                codes,
                start_date,
                end_date,
                interval=interval,
            )
            if not data_map:
                return {"error": "No data fetched"}
    except Exception as e:
        return {"error": f"Data loading failed: {e}"}

    # Determine bars_per_year
    market_types = {_detect_market(c) for c in codes}
    if len(market_types) > 1:
        bars_per_year = None
    else:
        bars_per_year = calc_bars_per_year(interval, source)

    # Create engine
    try:
        effective_source = source if source != "auto" else "tushare"
        market_engine = _create_market_engine(effective_source, test_config, codes)
    except Exception as e:
        return {"error": f"Engine creation failed: {e}"}

    # Run backtest
    try:
        metrics = market_engine.run_backtest(
            test_config, loader, signal_engine, run_dir, bars_per_year=bars_per_year
        )
        # Return only scalar metrics
        return {
            "sharpe": metrics.get("sharpe", 0.0),
            "total_return": metrics.get("total_return", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "calmar": metrics.get("calmar", 0.0),
            "sortino": metrics.get("sortino", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "trade_count": metrics.get("trade_count", 0),
        }
    except Exception as e:
        return {"error": f"Backtest failed: {e}"}


def run_grid_search(
    config: Dict[str, Any],
    run_dir: Path,
    param_grid: Dict[str, List[Any]],
    split_ratio: float = 0.7,
    objective: str = "sharpe",
    max_workers: int = 4,
) -> GridSearchSummary:
    """Run parameter grid search with IS/OOS validation.

    Args:
        config: Base backtest config.
        run_dir: Run directory.
        param_grid: Parameter grid specification.
        split_ratio: IS/OOS split ratio.
        objective: Metric to optimize ("sharpe", "calmar", "sortino").
        max_workers: Number of parallel workers.

    Returns:
        GridSearchSummary with all results.
    """
    import time

    start_time = time.time()

    # Get full date range from config
    start_date = config.get("start_date", "")
    end_date = config.get("end_date", "")

    # Load data once to get date range
    from backtest.loaders.registry import resolve_loader
    from backtest.runner import _fetch_auto, _detect_market, _AutoLoader

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
        raise ValueError("No data fetched for grid search")

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
    is_dates, oos_dates = split_dates_is_oos(dates, split_ratio)

    if len(is_dates) < 30 or len(oos_dates) < 10:
        raise ValueError(
            f"Insufficient data for IS/OOS split: IS={len(is_dates)}, OOS={len(oos_dates)}. "
            "Need at least 30 IS bars and 10 OOS bars."
        )

    is_start = str(is_dates[0].date())
    is_end = str(is_dates[-1].date())
    oos_start = str(oos_dates[0].date())
    oos_end = str(oos_dates[-1].date())

    # Generate parameter combinations
    param_combos = list(generate_param_combinations(param_grid))
    n_combos = len(param_combos)

    logger.info(
        "Grid search: %d combinations, IS=%s to %s, OOS=%s to %s",
        n_combos, is_start, is_end, oos_start, oos_end
    )

    results: List[GridSearchResult] = []

    # Run backtests in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for params in param_combos:
            # IS backtest
            is_future = executor.submit(
                _run_single_backtest,
                config, run_dir, params, is_start, is_end
            )
            futures[is_future] = ("is", params)

            # OOS backtest
            oos_future = executor.submit(
                _run_single_backtest,
                config, run_dir, params, oos_start, oos_end
            )
            futures[oos_future] = ("oos", params)

        # Collect results
        is_results: Dict[str, Dict] = {}
        oos_results: Dict[str, Dict] = {}

        for future in as_completed(futures):
            split_type, params = futures[future]
            params_key = str(sorted(params.items()))

            try:
                result = future.result()
                if split_type == "is":
                    is_results[params_key] = result
                else:
                    oos_results[params_key] = result
            except Exception as e:
                logger.warning("Backtest failed for %s: %s", params, e)

    # Combine IS/OOS results
    for params in param_combos:
        params_key = str(sorted(params.items()))
        is_metrics = is_results.get(params_key, {})
        oos_metrics = oos_results.get(params_key, {})

        if "error" in is_metrics or "error" in oos_metrics:
            continue

        is_sharpe = is_metrics.get("sharpe", 0.0)
        oos_sharpe = oos_metrics.get("sharpe", 0.0)

        # Overfit score: 1 - oos/is (higher = more overfit)
        if abs(is_sharpe) > 1e-6:
            overfit_score = 1.0 - (oos_sharpe / is_sharpe)
        else:
            overfit_score = 0.0

        results.append(GridSearchResult(
            params=params,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            overfit_score=overfit_score,
        ))

    elapsed = time.time() - start_time

    # Find best params by IS objective
    if results:
        best_result = max(results, key=lambda r: r.is_metrics.get(objective, 0.0))
        best_params = best_result.params
        best_is_sharpe = best_result.is_metrics.get("sharpe", 0.0)
        best_oos_sharpe = best_result.oos_metrics.get("sharpe", 0.0)
        best_overfit = best_result.overfit_score
    else:
        best_params = {}
        best_is_sharpe = 0.0
        best_oos_sharpe = 0.0
        best_overfit = 0.0

    return GridSearchSummary(
        results=results,
        best_params=best_params,
        best_is_sharpe=best_is_sharpe,
        best_oos_sharpe=best_oos_sharpe,
        best_overfit_score=best_overfit,
        total_combinations=n_combos,
        elapsed_seconds=elapsed,
    )


def format_grid_search_report(summary: GridSearchSummary) -> str:
    """Format grid search results as a readable report.

    Args:
        summary: Grid search summary.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        "参数网格搜索结果",
        "=" * 60,
        f"总参数组合: {summary.total_combinations}",
        f"有效结果: {len(summary.results)}",
        f"耗时: {summary.elapsed_seconds:.1f}秒",
        "",
        "最优参数 (按样本内 Sharpe):",
        f"  参数: {summary.best_params}",
        f"  样本内 Sharpe: {summary.best_is_sharpe:.4f}",
        f"  样本外 Sharpe: {summary.best_oos_sharpe:.4f}",
        f"  过拟合分数: {summary.best_overfit_score:.2%}",
        "",
        "Top 10 参数组合:",
        "-" * 60,
    ]

    # Sort by IS sharpe
    sorted_results = sorted(
        summary.results,
        key=lambda r: r.is_metrics.get("sharpe", 0.0),
        reverse=True
    )[:10]

    for i, r in enumerate(sorted_results, 1):
        lines.append(f"{i}. 参数: {r.params}")
        lines.append(f"   IS Sharpe: {r.is_metrics.get('sharpe', 0.0):.4f}, "
                    f"OOS Sharpe: {r.oos_metrics.get('sharpe', 0.0):.4f}, "
                    f"过拟合: {r.overfit_score:.2%}")

    lines.append("=" * 60)
    return "\n".join(lines)


def main(run_dir: Path) -> None:
    """CLI entry point for grid search.

    Args:
        run_dir: Run directory with config.json.
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(json.dumps({"error": "config.json not found"}))
        sys.exit(1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    gs_cfg = config.get("grid_search", {})

    if not gs_cfg:
        print(json.dumps({"error": "No grid_search config found in config.json"}))
        sys.exit(1)

    param_grid = gs_cfg.get("param_grid", {})
    if not param_grid:
        print(json.dumps({"error": "No param_grid specified"}))
        sys.exit(1)

    split_ratio = gs_cfg.get("split_ratio", 0.7)
    objective = gs_cfg.get("objective", "sharpe")
    max_workers = gs_cfg.get("max_workers", 4)

    try:
        summary = run_grid_search(
            config=config,
            run_dir=run_dir,
            param_grid=param_grid,
            split_ratio=split_ratio,
            objective=objective,
            max_workers=max_workers,
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    # Write results
    out_dir = run_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    results_json = {
        "best_params": summary.best_params,
        "best_is_sharpe": summary.best_is_sharpe,
        "best_oos_sharpe": summary.best_oos_sharpe,
        "best_overfit_score": summary.best_overfit_score,
        "total_combinations": summary.total_combinations,
        "valid_results": len(summary.results),
        "elapsed_seconds": summary.elapsed_seconds,
        "all_results": [
            {
                "params": r.params,
                "is_sharpe": r.is_metrics.get("sharpe", 0.0),
                "oos_sharpe": r.oos_metrics.get("sharpe", 0.0),
                "is_return": r.is_metrics.get("total_return", 0.0),
                "oos_return": r.oos_metrics.get("total_return", 0.0),
                "overfit_score": r.overfit_score,
            }
            for r in summary.results
        ],
    }

    out_path = out_dir / "grid_search.json"
    out_path.write_text(json.dumps(results_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # Sensitivity analysis
    try:
        from backtest.param_sensitivity import compute_sensitivity
        sensitivity = compute_sensitivity(results_json["all_results"], "is_sharpe")
        if sensitivity:
            results_json["sensitivity"] = sensitivity
            out_path.write_text(json.dumps(results_json, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("Sensitivity analysis failed: %s", exc)

    # Print report
    print(format_grid_search_report(summary))

    # Print sensitivity summary
    try:
        from backtest.param_sensitivity import format_sensitivity_text
        if "sensitivity" in results_json:
            print("\n" + format_sensitivity_text(results_json["sensitivity"]))
    except Exception:
        pass

    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.grid_search <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser()
    main(run_dir)
