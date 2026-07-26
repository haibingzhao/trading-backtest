"""Engine-level stop-loss and take-profit rules.

Provides a safety net independent from strategy-level risk management.
Checks all open positions each bar and force-closes those that breach
configured thresholds.

Usage: triggered by config["stop_rules"] in the backtest config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from backtest.models import Position


@dataclass(frozen=True)
class StopLossConfig:
    """止损配置。

    Attributes:
        enabled: 是否启用止损。
        pct: 固定百分比止损（相对入场价，如 0.08 = 8%）。
        trailing_pct: 移动止损百分比（从最高点回撤 N% 平仓，0 = 不使用）。
    """

    enabled: bool = False
    pct: float = 0.08
    trailing_pct: float = 0.0


@dataclass(frozen=True)
class TakeProfitConfig:
    """止盈配置。

    Attributes:
        enabled: 是否启用止盈。
        pct: 固定百分比止盈（如 0.20 = 20%）。
    """

    enabled: bool = False
    pct: float = 0.20


class EngineStopRules:
    """引擎层止损/止盈安全网。

    在 _execute_bars() 的每个 bar 执行前检查所有持仓，
    触发止损/止盈时强制平仓。exit_reason 标记为 "stop_loss" 或 "take_profit"。

    与策略层风控（RiskManager）独立：
    - 策略层风控：基于信号/净值的风控（回撤熔断、连续止损暂停等）
    - 引擎层止损止盈：基于持仓盈亏的最终防线

    Args:
        stop_loss: 止损配置。
        take_profit: 止盈配置。
    """

    def __init__(
        self,
        stop_loss: StopLossConfig,
        take_profit: TakeProfitConfig,
    ) -> None:
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # 移动止损/止盈追踪状态: symbol -> highest_price_since_entry
        self._trailing_state: Dict[str, float] = {}

    def check_positions(
        self,
        positions: Dict[str, Position],
        close_df: pd.DataFrame,
        ts: pd.Timestamp,
    ) -> List[tuple[str, str]]:
        """检查所有持仓是否触发止损/止盈。

        Args:
            positions: 当前持仓字典 {symbol: Position}。
            close_df: 收盘价 DataFrame（dates x symbols）。
            ts: 当前时间戳。

        Returns:
            需要平仓的 (symbol, reason) 列表。reason 为 "stop_loss" 或 "take_profit"。
        """
        force_close: List[tuple[str, str]] = []

        for symbol, pos in positions.items():
            if ts not in close_df.index:
                continue
            if symbol not in close_df.columns:
                continue

            current_price = float(close_df.at[ts, symbol])
            if current_price <= 0:
                continue

            # 更新移动止损追踪
            if self.stop_loss.trailing_pct > 0:
                if symbol not in self._trailing_state:
                    self._trailing_state[symbol] = current_price
                else:
                    self._trailing_state[symbol] = max(
                        self._trailing_state[symbol], current_price
                    )

            reason = self._check_single(pos, current_price, symbol)
            if reason:
                force_close.append((symbol, reason))
                # 清除追踪状态
                self._trailing_state.pop(symbol, None)

        return force_close

    def _check_single(
        self, pos: Position, current_price: float, symbol: str
    ) -> str | None:
        """检查单个持仓是否触发止损/止盈。

        Returns:
            触发原因 ("stop_loss" / "take_profit") 或 None。
        """
        entry_price = pos.entry_price
        direction = pos.direction

        # 计算盈亏百分比（考虑方向）
        if entry_price <= 0:
            return None

        pnl_pct = direction * (current_price - entry_price) / entry_price

        # 止损检查
        if self.stop_loss.enabled:
            # 固定百分比止损
            if pnl_pct <= -self.stop_loss.pct:
                return "stop_loss"

            # 移动止损
            if self.stop_loss.trailing_pct > 0 and symbol in self._trailing_state:
                highest = self._trailing_state[symbol]
                trailing_pnl = direction * (current_price - highest) / highest
                if trailing_pnl <= -self.stop_loss.trailing_pct:
                    return "stop_loss"

        # 止盈检查
        if self.take_profit.enabled:
            if pnl_pct >= self.take_profit.pct:
                return "take_profit"

        return None

    def reset(self) -> None:
        """重置所有追踪状态。"""
        self._trailing_state.clear()


def parse_stop_rules_config(config: Dict[str, Any]) -> EngineStopRules | None:
    """从 config 解析止损止盈配置。

    Args:
        config: 回测配置中的 "stop_rules" 部分。

    Returns:
        EngineStopRules 实例，或 None（如果未启用）。
    """
    if not config:
        return None

    sl_cfg = config.get("stop_loss", {})
    tp_cfg = config.get("take_profit", {})

    stop_loss = StopLossConfig(
        enabled=sl_cfg.get("enabled", False),
        pct=sl_cfg.get("pct", 0.08),
        trailing_pct=sl_cfg.get("trailing_pct", 0.0),
    )

    take_profit = TakeProfitConfig(
        enabled=tp_cfg.get("enabled", False),
        pct=tp_cfg.get("pct", 0.20),
    )

    if not stop_loss.enabled and not take_profit.enabled:
        return None

    return EngineStopRules(stop_loss=stop_loss, take_profit=take_profit)
