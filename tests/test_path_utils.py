"""Unit tests for backtest._compat.path_utils module."""
import os
from pathlib import Path

import pytest

from backtest._compat.path_utils import safe_run_dir


class TestSafeRunDir:
    def test_rejects_unc_path(self):
        with pytest.raises(ValueError, match="UNC paths are not allowed"):
            safe_run_dir("//server/share/runs")

    def test_rejects_outside_allowed_roots(self, tmp_path):
        # tmp_path is not in any allowed run root
        outside = tmp_path / "somewhere" / "else"
        outside.mkdir(parents=True)
        with pytest.raises(ValueError, match="outside allowed run roots"):
            safe_run_dir(str(outside))

    def test_accepts_path_in_allowed_root(self, tmp_path, monkeypatch):
        # Add tmp_path as an allowed run root via env var
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        monkeypatch.setenv("TRADING_BACKTEST_ALLOWED_RUN_ROOTS", str(tmp_path))
        result = safe_run_dir(str(run_dir))
        assert result == run_dir.resolve()

    def test_tilde_expansion(self, monkeypatch):
        # Ensure ~ expansion doesn't crash (even if path doesn't exist in roots)
        home = Path.home()
        run_root = home / ".trading-backtest" / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        result = safe_run_dir("~/.trading-backtest/runs")
        assert result == run_root.resolve()
