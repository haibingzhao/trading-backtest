"""Structured export for grid search results.

After grid search completes, generates a structured artifact directory with:
- Shared OHLCV data (one copy for all periods)
- Separate train/test/full_period subdirectories with equity, trades, positions
- manifest.json for DB ingestion

Usage:
    Called automatically by backtest.grid_search main() after optimization completes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def export_grid_search(config: Dict[str, Any], run_dir: Path, summary: Any) -> Path:
    """Generate structured artifacts after grid search completes.

    Args:
        config: Base backtest config dict (from config.json).
        run_dir: Run directory containing config.json and code/.
        summary: GridSearchSummary dataclass with best_params, date ranges, etc.

    Returns:
        Path to the structured output directory.
    """
    structured_dir = run_dir / "artifacts" / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)

    best_params = summary.best_params
    is_start = summary.is_start
    is_end = summary.is_end
    oos_start = summary.oos_start
    oos_end = summary.oos_end

    # Full period spans the entire date range
    full_start = is_start
    full_end = oos_end

    codes = config.get("codes", [])
    source = config.get("source", "tushare")
    interval = config.get("interval", "1D")

    # 1. Load full-period data and write shared OHLCV (one copy)
    data_map = _load_data(config, codes, source, interval, full_start, full_end)
    if data_map:
        _export_ohlcv(data_map, structured_dir)

    # 2. Run train period backtest → structured/train/
    train_dir = structured_dir / "train"
    train_metrics = _run_and_export(
        config, run_dir, best_params, is_start, is_end, train_dir
    )

    # 3. Run test period backtest → structured/test/
    test_dir = structured_dir / "test"
    test_metrics = _run_and_export(
        config, run_dir, best_params, oos_start, oos_end, test_dir
    )

    # 4. Run full period backtest → structured/full_period/ (with HTML report)
    full_dir = structured_dir / "full_period"
    full_metrics = _run_and_export(
        config, run_dir, best_params, full_start, full_end, full_dir
    )

    # 5. Write manifest.json
    manifest = _build_manifest(
        config=config,
        run_dir=run_dir,
        best_params=best_params,
        split_ratio=config.get("grid_search", {}).get("split_ratio", 0.7),
        is_start=is_start,
        is_end=is_end,
        oos_start=oos_start,
        oos_end=oos_end,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        full_metrics=full_metrics,
        overfit_score=summary.best_overfit_score,
        codes=codes,
    )
    _write_manifest(manifest, structured_dir)

    logger.info("Structured export complete: %s", structured_dir)
    return structured_dir


# ── Internal helpers ──


def _load_data(
    config: Dict[str, Any],
    codes: list,
    source: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame] | None:
    """Load market data for the full period."""
    try:
        if source == "auto":
            from backtest.runner import _fetch_auto
            return _fetch_auto(codes, config, interval)
        else:
            from backtest.loaders.registry import resolve_loader
            LoaderCls = resolve_loader(source)
            loader = LoaderCls()
            return loader.fetch(codes, start_date, end_date, interval=interval)
    except Exception as exc:
        logger.warning("Failed to load data for OHLCV export: %s", exc)
        return None


def _export_ohlcv(data_map: Dict[str, pd.DataFrame], structured_dir: Path) -> None:
    """Write shared OHLCV files (one per symbol) to structured directory."""
    for code, df in data_map.items():
        out_path = structured_dir / f"ohlcv_{code}.csv"
        df.to_csv(out_path)


def _run_and_export(
    config: Dict[str, Any],
    run_dir: Path,
    signal_params: Dict[str, Any],
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run a single backtest and write artifacts to output_dir.

    Returns:
        Metrics dict (scalar values only for manifest).
    """
    import importlib.util
    import sys

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load signal engine module
    signal_path = run_dir / "code" / "signal_engine.py"
    spec = importlib.util.spec_from_file_location("signal_engine_export", signal_path)
    if spec is None or spec.loader is None:
        logger.error("Cannot load signal engine from %s", signal_path)
        return {}
    module = importlib.util.module_from_spec(spec)
    sys.modules["signal_engine_export"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error("Failed to load signal engine: %s", e)
        return {}

    engine_cls = getattr(module, "SignalEngine", None)
    if engine_cls is None:
        logger.error("SignalEngine class not found")
        return {}

    # Create modified config
    test_config = config.copy()
    test_config["start_date"] = start_date
    test_config["end_date"] = end_date
    test_config["signal_params"] = signal_params

    # Create signal engine
    try:
        signal_engine = engine_cls(**signal_params) if signal_params else engine_cls()
    except Exception as e:
        logger.error("Failed to create signal engine: %s", e)
        return {}

    # Create loader and engine
    from backtest.loaders.registry import resolve_loader
    from backtest.metrics import calc_bars_per_year
    from backtest.runner import _AutoLoader, _create_market_engine, _detect_market

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
            data_map = loader.fetch(codes, start_date, end_date, interval=interval)
            if not data_map:
                logger.error("No data fetched for %s ~ %s", start_date, end_date)
                return {}
    except Exception as e:
        logger.error("Data loading failed: %s", e)
        return {}

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
        logger.error("Engine creation failed: %s", e)
        return {}

    # Run backtest with artifacts written to output_dir
    try:
        metrics = market_engine.run_backtest(
            test_config, loader, signal_engine, run_dir,
            bars_per_year=bars_per_year,
            write_artifacts=True,
            artifact_dir=output_dir,
        )
        # Return only scalar metrics for manifest
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
        logger.error("Backtest failed for %s ~ %s: %s", start_date, end_date, e)
        return {}


def _build_manifest(
    config: Dict[str, Any],
    run_dir: Path,
    best_params: Dict[str, Any],
    split_ratio: float,
    is_start: str,
    is_end: str,
    oos_start: str,
    oos_end: str,
    train_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    full_metrics: Dict[str, Any],
    overfit_score: float,
    codes: list,
) -> Dict[str, Any]:
    """Build manifest.json content."""
    # Collect OHLCV file names
    ohlcv_files = [f"ohlcv_{code}.csv" for code in codes]

    return {
        "version": "1.0",
        "tool": "grid_search",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "run_dir": str(run_dir),
        "config": {
            "symbols": codes,
            "source": config.get("source", "tushare"),
            "interval": config.get("interval", "1D"),
            "start_date": config.get("start_date", ""),
            "end_date": config.get("end_date", ""),
            "initial_capital": config.get("initial_capital", 1_000_000),
            "commission": config.get("commission", 0.0),
            "strategy_file": "code/signal_engine.py",
        },
        "best_params": best_params,
        "split_info": {
            "ratio": split_ratio,
            "train_start": is_start,
            "train_end": is_end,
            "test_start": oos_start,
            "test_end": oos_end,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "full_period_metrics": full_metrics,
        "overfit_score": overfit_score,
        "files": {
            "ohlcv": ohlcv_files,
            "train": {
                "equity": "train/equity.csv",
                "trades": "train/trades.csv",
                "positions": "train/positions.csv",
            },
            "test": {
                "equity": "test/equity.csv",
                "trades": "test/trades.csv",
                "positions": "test/positions.csv",
            },
            "full_period": {
                "equity": "full_period/equity.csv",
                "trades": "full_period/trades.csv",
                "positions": "full_period/positions.csv",
                "report": "full_period/report.html",
            },
        },
    }


def _write_manifest(manifest: Dict[str, Any], structured_dir: Path) -> None:
    """Write manifest.json to structured directory."""
    manifest_path = structured_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
