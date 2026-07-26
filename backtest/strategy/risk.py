"""风控管理层。

从 signal_engine.py 的风控逻辑 (L334-369) 提取。
管理三类风控: 回撤熔断、连续止损暂停、日亏损限制。
"""

from __future__ import annotations

import numpy as np


class RiskManager:
    """策略风控管理器。

    三类风控机制:
    1. 回撤熔断 — 净值回撤超过阈值时暂停交易，恢复到一半以下时解除
    2. 连续止损暂停 — 连续触发止损后暂停 N 根 bar
    3. 日亏损限制 — 单日亏损超过阈值时清空信号

    Args:
        max_drawdown_halt: 回撤熔断阈值（如 0.08 = 8%）。
        max_daily_loss: 日亏损限制（如 0.03 = 3%）。
        max_consecutive_stops: 触发暂停的连续止损次数。
        pause_bars: 暂停的 bar 数。
    """

    def __init__(
        self,
        max_drawdown_halt: float = 0.08,
        max_daily_loss: float = 0.03,
        max_consecutive_stops: int = 3,
        pause_bars: int = 3,
    ):
        self.max_drawdown_halt = max_drawdown_halt
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_stops = max_consecutive_stops
        self.pause_bars = pause_bars

        # 运行时状态
        self.peak_equity: float = 1.0
        self.halted: bool = False
        self.consecutive_stops: int = 0
        self.pause_bars_left: int = 0
        self.prev_day_equity: float = 1.0

    def reset(self) -> None:
        """重置所有运行时状态。"""
        self.peak_equity = 1.0
        self.halted = False
        self.consecutive_stops = 0
        self.pause_bars_left = 0
        self.prev_day_equity = 1.0

    def check(
        self,
        i: int,
        signal: float,
        close_arr: np.ndarray,
        equity_curve: np.ndarray,
        prev_signal: float = 0.0,
    ) -> tuple[bool, float]:
        """检查风控条件，返回是否允许交易及信号覆盖值。

        此方法同时更新内部净值曲线（模拟: signal × 日收益）。

        Args:
            i: 当前 bar 索引。
            signal: 当前信号权重。
            close_arr: 收盘价 numpy 数组。
            equity_curve: 净值曲线 numpy 数组（会被原地更新）。
            prev_signal: 上一 bar 的信号权重（优先使用）。

        Returns:
            (is_active, signal_override):
            - is_active: True 表示允许交易，False 表示被风控拦截。
            - signal_override: 风控覆盖后的信号值（通常为 0.0）。
        """
        self.update_equity_curve(i, signal, close_arr, equity_curve, prev_signal)
        return self.check_risk_conditions(i, signal, equity_curve)

    def update_equity_curve(
        self,
        i: int,
        signal: float,
        close_arr: np.ndarray,
        equity_curve: np.ndarray,
        prev_signal: float = 0.0,
    ) -> None:
        """更新模拟净值曲线（纯计算，不做风控判断）。

        使用上一 bar 的信号（prev_signal）计算净值变化，
        与执行引擎的 next-bar-open 语义对齐。

        Args:
            i: 当前 bar 索引。
            signal: 当前信号权重（用于向后兼容，未提供 prev_signal 时使用）。
            close_arr: 收盘价 numpy 数组。
            equity_curve: 净值曲线 numpy 数组（原地更新）。
            prev_signal: 上一 bar 的信号权重（优先使用）。
        """
        if i > 30:
            daily_ret = (
                (close_arr[i] - close_arr[i - 1]) / close_arr[i - 1]
                if close_arr[i - 1] > 0
                else 0
            )
            # 使用上一 bar 的信号（next-bar-open 语义）
            effective_signal = prev_signal if prev_signal != 0.0 else signal
            equity_curve[i] = equity_curve[i - 1] * (1 + effective_signal * daily_ret)
        else:
            equity_curve[i] = equity_curve[i - 1] if i > 0 else 1.0

        self.peak_equity = max(self.peak_equity, equity_curve[i])

    def check_risk_conditions(
        self,
        i: int,
        signal: float,
        equity_curve: np.ndarray,
    ) -> tuple[bool, float]:
        """检查风控条件（纯判断，不更新净值）。

        Args:
            i: 当前 bar 索引。
            signal: 当前信号权重。
            equity_curve: 净值曲线 numpy 数组。

        Returns:
            (is_active, signal_override):
            - is_active: True 表示允许交易。
            - signal_override: 风控覆盖后的信号值。
        """
        drawdown = (
            (self.peak_equity - equity_curve[i]) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )

        # 回撤熔断
        if drawdown > self.max_drawdown_halt:
            self.halted = True
        if self.halted and drawdown < self.max_drawdown_halt * 0.5:
            self.halted = False
            self.consecutive_stops = 0

        if self.halted:
            return False, 0.0

        # 连续止损暂停
        if self.pause_bars_left > 0:
            self.pause_bars_left -= 1
            return False, 0.0

        # 日亏损限制
        daily_pnl = (
            (equity_curve[i] - self.prev_day_equity) / self.prev_day_equity
            if self.prev_day_equity > 0
            else 0
        )
        if daily_pnl < -self.max_daily_loss and signal != 0:
            return False, 0.0

        return True, signal

    def record_stop(self) -> None:
        """记录一次止损事件。

        当连续止损达到阈值时，触发暂停。
        """
        self.consecutive_stops += 1
        if self.consecutive_stops >= self.max_consecutive_stops:
            self.pause_bars_left = self.pause_bars

    def reset_stops(self) -> None:
        """重置连续止损计数（成功入场时调用）。"""
        self.consecutive_stops = 0

    def update_equity(self, equity: float) -> None:
        """更新前日净值基准。"""
        self.prev_day_equity = equity
