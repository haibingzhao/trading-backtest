"""Standalone path safety helpers for the backtest framework.

Replaces ``src.tools.path_utils`` from the full trading agent.
Only the ``safe_run_dir`` function is needed by ``backtest.runner``.
The full agent version enforces an allowlist of run roots; here we
provide a simpler but still safe implementation that validates the
run directory exists and is not a UNC path.
"""

from __future__ import annotations

import os
from pathlib import Path

_ALLOWED_RUN_ROOTS_ENV = "TRADING_BACKTEST_ALLOWED_RUN_ROOTS"


def _rejects_unc(p: str) -> None:
    """Raise ValueError if `p` starts with a UNC share prefix."""
    if p.startswith("\\\\") or p.startswith("//"):
        raise ValueError(f"UNC paths are not allowed: {p!r}")


def _default_run_roots() -> list[Path]:
    """Return default roots for generated backtest/tool run directories."""
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return [
        project_root / "runs",
        project_root / "backtest" / "runs",
        cwd / "runs",
        home / ".trading-backtest" / "runs",
    ]


def _allowed_run_roots() -> list[Path]:
    """Return all roots allowed for run_dir-based tools.

    Reads from the ``TRADING_BACKTEST_ALLOWED_RUN_ROOTS`` environment
    variable (comma-separated paths) and merges with defaults.
    """
    raw = os.getenv(_ALLOWED_RUN_ROOTS_ENV, "")
    configured: list[Path] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        _rejects_unc(item)
        configured.append(Path(item).expanduser().resolve())

    roots: list[Path] = []
    for root in [*_default_run_roots(), *configured]:
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def safe_run_dir(p: str) -> Path:
    """Validate a run directory used by generated-code tools.

    Args:
        p: User/LLM-supplied run directory. ``~`` expansion is supported.

    Returns:
        Absolute resolved path inside an allowed run root.

    Raises:
        ValueError: If `p` is a UNC share or resolves outside all allowed
            run roots.
    """
    _rejects_unc(p)
    resolved = Path(p).expanduser().resolve()

    for root in _allowed_run_roots():
        if resolved.is_relative_to(root):
            return resolved

    raise ValueError(
        f"run_dir {p!r} is outside allowed run roots. "
        f"Set {_ALLOWED_RUN_ROOTS_ENV} to add a run directory."
    )
