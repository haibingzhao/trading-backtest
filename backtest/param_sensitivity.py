"""Parameter sensitivity analysis for grid search results.

Analyzes how sensitive the objective metric is to each parameter dimension.
A "plateau" pattern (low CV) indicates robustness; a "peak" pattern (high CV)
indicates overfitting risk.

Usage::

    python -m backtest.param_sensitivity <run_dir>

Requires ``grid_search.json`` in ``<run_dir>/artifacts/`` (produced by
``python -m backtest.grid_search <run_dir>``).
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

def compute_sensitivity(
    all_results: List[Dict[str, Any]],
    objective: str = "is_sharpe",
) -> Dict[str, Dict[str, Any]]:
    """Compute per-parameter sensitivity from grid search results.

    For each parameter, marginalize over all other parameters and compute
    the mean/std/CV of the objective metric at each value of that parameter.

    Args:
        all_results: List of dicts from ``grid_search.json["all_results"]``.
            Each dict must contain the parameter keys and the *objective* key.
        objective: Metric key to analyze (e.g. ``"is_sharpe"``, ``"oos_sharpe"``).

    Returns:
        ``{param_name: {value: {"mean": ..., "std": ..., "cv": ..., "plateau_score": ...}}}``
    """
    if not all_results:
        return {}

    # Discover parameter names (all keys that are not the objective or metadata)
    skip_keys = {objective, "is_sharpe", "oos_sharpe", "overfit_score",
                 "is_return", "oos_return", "is_max_dd", "oos_max_dd",
                 "error", "params"}
    sample = all_results[0]
    param_names = [k for k in sample if k not in skip_keys and not isinstance(sample[k], dict)]

    # If params are nested under a "params" key, flatten
    if "params" in sample and isinstance(sample["params"], dict):
        param_names = list(sample["params"].keys())
        results_flat = []
        for r in all_results:
            flat = {**r.get("params", {}), objective: r.get(objective, 0.0)}
            results_flat.append(flat)
        all_results = results_flat

    sensitivity: Dict[str, Dict[str, Any]] = {}

    for param in param_names:
        if param in skip_keys:
            continue

        # Group results by this parameter's value
        groups: Dict[Any, List[float]] = {}
        for r in all_results:
            val = r.get(param)
            metric = r.get(objective, 0.0)
            if val is None or metric is None:
                continue
            key = str(val)
            groups.setdefault(key, []).append(float(metric))

        if not groups:
            continue

        value_stats: Dict[str, Dict[str, Any]] = {}
        all_means = []
        for val_key, metrics in sorted(groups.items()):
            mean = float(np.mean(metrics))
            std = float(np.std(metrics))
            cv = abs(std / mean) if abs(mean) > 1e-8 else float("inf")
            value_stats[val_key] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "cv": round(cv, 4),
                "n_samples": len(metrics),
            }
            all_means.append(mean)

        # Overall plateau score: inverse of CV across all values of this param
        if all_means:
            overall_mean = float(np.mean(all_means))
            overall_std = float(np.std(all_means))
            overall_cv = abs(overall_std / overall_mean) if abs(overall_mean) > 1e-8 else float("inf")
            plateau = round(1.0 / overall_cv, 2) if overall_cv > 1e-8 else float("inf")
        else:
            overall_cv = 0.0
            plateau = 0.0

        sensitivity[param] = {
            "values": value_stats,
            "overall_cv": round(overall_cv, 4),
            "plateau_score": plateau,
            "pattern": "plateau" if overall_cv < 0.5 else ("moderate" if overall_cv < 1.0 else "peak"),
        }

    return sensitivity


def format_sensitivity_text(sensitivity: Dict[str, Dict[str, Any]]) -> str:
    """Format sensitivity analysis as a readable text report."""
    if not sensitivity:
        return "无参数敏感度数据"

    lines = [
        "=" * 70,
        "参数敏感度分析",
        "=" * 70,
        "",
        "说明: 高原型(plateau)=稳健, 山峰型(peak)=过拟合风险",
        "",
    ]

    # Sort by plateau_score descending (most robust first)
    sorted_params = sorted(
        sensitivity.items(),
        key=lambda x: x[1].get("plateau_score", 0),
        reverse=True,
    )

    for param, info in sorted_params:
        pattern = info.get("pattern", "unknown")
        cv = info.get("overall_cv", 0)
        plateau = info.get("plateau_score", 0)

        pattern_label = {"plateau": "高原型(稳健)", "moderate": "中等", "peak": "山峰型(风险)"}
        label = pattern_label.get(pattern, pattern)

        lines.append(f"参数: {param}")
        lines.append(f"  模式: {label}  |  CV={cv:.2f}  |  Plateau Score={plateau}")
        lines.append(f"  {'值':>12} {'均值':>8} {'标准差':>8} {'样本数':>6}")
        lines.append(f"  {'-' * 40}")

        for val, stats in info.get("values", {}).items():
            lines.append(
                f"  {val:>12} {stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['n_samples']:>6}"
            )
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def sensitivity_to_json(sensitivity: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert sensitivity data to JSON-serializable dict (for HTML report)."""
    return json.loads(json.dumps(sensitivity, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(run_dir: Path) -> None:
    """CLI entry: read grid_search.json, output sensitivity report."""
    gs_path = run_dir / "artifacts" / "grid_search.json"
    if not gs_path.exists():
        print(json.dumps({"error": "grid_search.json not found. Run grid search first."}))
        sys.exit(1)

    gs_data = json.loads(gs_path.read_text(encoding="utf-8"))
    all_results = gs_data.get("all_results", [])

    if not all_results:
        print(json.dumps({"error": "No results found in grid_search.json"}))
        sys.exit(1)

    # Determine objective key
    sample = all_results[0]
    objective = "is_sharpe" if "is_sharpe" in sample else "sharpe"

    sensitivity = compute_sensitivity(all_results, objective)
    print(format_sensitivity_text(sensitivity))

    # Write JSON output
    out_path = run_dir / "artifacts" / "param_sensitivity.json"
    out_path.write_text(
        json.dumps(sensitivity, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.param_sensitivity <run_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]).expanduser())
