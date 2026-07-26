"""策略参数配置集合。

将 StrategyBase 的 30+ 个 __init__ 参数分组为 frozen dataclass，
提升可读性和可维护性。StrategyBase.__init__ 保持散参数接口不变，
内部自动构建这些 dataclass。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeParams:
    """Regime 检测参数。"""

    trend_strength_threshold: float = 25.0
    trend_strength_grid_max: float = 20.0
    mode_confirm_bars: int = 1
    mode_cooldown_bars: int = 1
    osc_confirm_bars: int = 2
    vol_high_thresh: float = 1.3
    vol_low_thresh: float = 0.7


@dataclass(frozen=True)
class GridParams:
    """网格策略参数。"""

    grid_reset_days: int = 5
    grid_stop_loss_pct: float = 0.10
    max_position_ratio: float = 0.8


@dataclass(frozen=True)
class TrendParams:
    """趋势做多策略参数。"""

    risk_per_trade: float = 0.03
    initial_stop_atr_mult: float = 3.0
    trailing_stop_atr_mult: float = 2.0
    max_pyramid: int = 4
    max_position_ratio: float = 0.8
    reversal_ema_gap_pct: float = 0.003
    min_hold_bars: int = 2


@dataclass(frozen=True)
class ShortParams:
    """做空策略参数。"""

    allow_short: bool = False
    max_short_ratio: float = 0.5
    short_stop_atr_mult: float = 2.5
    short_squeeze_rsi: float = 75.0


@dataclass(frozen=True)
class MeanReversionParams:
    """均值回归策略参数。"""

    zscore_entry: float = 2.0       # 入场 Z-Score 阈值（价格偏离均线几倍 ATR）
    zscore_exit: float = 0.5        # 出场 Z-Score 阈值（回归均线附近平仓）
    zscore_stop: float = 3.0        # 止损 Z-Score（继续偏离则止损）
    rsi_oversold: float = 30.0      # RSI 超卖阈值
    rsi_overbought: float = 70.0    # RSI 超买阈值
    max_position_ratio: float = 0.5  # 最大仓位（比趋势策略保守）


@dataclass(frozen=True)
class RiskParams:
    """风控参数。"""

    max_drawdown_halt: float = 0.08
    max_daily_loss: float = 0.03
    max_consecutive_stops: int = 3
    pause_bars: int = 3


@dataclass(frozen=True)
class PortfolioParams:
    """组合级风控参数（多标的回测时生效）。

    所有约束均为 opt-in：enabled=False 时不做任何后处理。
    """

    #: 是否启用组合级约束（多标的回测时设为 True 启用）
    enabled: bool = False
    #: 总敞口上限（所有标的绝对值之和）
    max_gross_exposure: float = 2.0
    #: 净敞口上限（所有标的信号之和的绝对值）
    max_net_exposure: float = 1.0
    #: 单标的信号上限
    max_single_weight: float = 0.3
    #: 板块/行业上限（需提供 sector_map）
    max_sector_weight: float = 0.5
    #: 相关性惩罚系数（高相关标的自动降权）
    correlation_penalty: float = 0.5
    #: 相关性回溯窗口（bar 数）
    correlation_lookback: int = 60
