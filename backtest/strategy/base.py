"""策略编排器基类。

提供 StrategyBase — 满足 SignalEngine 合约的模板方法基类。
负责编排: 特征计算 → Regime 检测 → 风控检查 → 子策略分派。
框架层不绑定任何特定技术指标，具体指标的使用在默认实现中。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from backtest.strategy.components import (
    EntrySignal,
    GridStrategy,
    MeanReversionStrategy,
    SARDirectionEntry,
    TrendLongStrategy,
    TrendShortStrategy,
    build_arrays_ns,
)
from backtest.strategy.indicators import (
    IndicatorParams,
    IndicatorPipeline,
    build_regime_features,
    build_regime_features_fast,
    compute_indicators,
    default_pipeline,
)
from backtest.strategy.market_rules import MarketRules
from backtest.strategy.params import (
    GridParams,
    MeanReversionParams,
    PortfolioParams,
    RegimeParams,
    RiskParams,
    ShortParams,
    TrendParams,
)
from backtest.strategy.regime import (
    DEFAULT_REGIME_CONFIGS,
    DefaultRegimeDetector,
    Regime,
    RegimeConfig,
    RegimeDetector,
)
from backtest.strategy.risk import RiskManager


# ==================================================================
# SignalTrace — 信号可追溯性日志
# ==================================================================


@dataclass
class SignalTrace:
    """单根 bar 的信号决策记录。

    用于回测后分析“为什么在某个时点做了这个决策”。
    通过 ``verbose=True`` 开启记录。
    """

    bar: int
    code: str
    regime: str
    sub_strategy: str
    action: str
    signal: float
    reasons: List[str] = field(default_factory=list)


class StrategyBase:
    """策略编排器基类，满足 SignalEngine 合约。

    所有参数均有默认值，可通过 ``StrategyBase()`` 或 ``StrategyBase(**signal_params)`` 实例化。
    子类可覆盖工厂方法以注入自定义组件:
      - create_regime_detector()
      - create_grid_strategy()
      - create_trend_long_strategy()
      - create_trend_short_strategy()
      - create_mean_reversion_strategy()

    也可覆盖钩子方法在关键节点插入自定义逻辑:
      - on_regime_change(old, new)
      - _build_regime_features(d, i)  — 使用不同指标构建特征
    """

    # ------------------------------------------------------------------
    # 类常量（原硬编码魔数，提取为可覆盖的类属性）
    # ------------------------------------------------------------------

    #: 数据量最小阈值，低于此值的标的直接返回全零信号
    MIN_BARS_REQUIRED: int = 30
    #: 熊市做多降权系数（BEAR_TREND 下做多信号乘以此系数）
    BEAR_LONG_WEIGHT_SCALE: float = 0.3
    #: 趋势入场风险缩放因子（risk_per_trade × confidence × 此系数）
    ENTRY_RISK_SCALE: float = 2.25

    def __init__(
        self,
        # 指标参数
        adx_period: int = 14,
        adx_smooth: int = 6,
        atr_period: int = 14,
        ema_fast: int = 12,
        ema_slow: int = 26,
        sar_accel: float = 0.02,
        sar_max_accel: float = 0.2,
        rsi_period: int = 14,
        ma_period: int = 20,
        atr_ma_period: int = 60,
        # Regime 参数
        trend_strength_threshold: float = 25,
        trend_strength_grid_max: float = 20,
        mode_confirm_bars: int = 1,
        mode_cooldown_bars: int = 1,
        osc_confirm_bars: int = 2,
        vol_high_thresh: float = 1.3,
        vol_low_thresh: float = 0.7,
        # 网格参数
        grid_levels: int = 5,  # deprecated: 未使用，保留以向后兼容
        grid_reset_days: int = 5,
        grid_stop_loss_pct: float = 0.10,
        # 趋势参数
        risk_per_trade: float = 0.03,
        initial_stop_atr_mult: float = 3.0,
        trailing_stop_atr_mult: float = 2.0,
        max_pyramid: int = 4,
        max_position_ratio: float = 0.8,
        reversal_ema_gap_pct: float = 0.003,
        min_hold_bars: int = 2,
        # 做空参数
        allow_short: bool = False,
        max_short_ratio: float = 0.5,
        short_stop_atr_mult: float = 2.5,
        short_squeeze_rsi: float = 75,
        # 均值回归参数
        mr_enabled: bool = False,
        mr_zscore_entry: float = 2.0,
        mr_zscore_exit: float = 0.5,
        mr_zscore_stop: float = 3.0,
        mr_rsi_oversold: float = 30.0,
        mr_rsi_overbought: float = 70.0,
        mr_max_position_ratio: float = 0.5,
        # 风控参数
        max_drawdown_halt: float = 0.08,
        max_daily_loss: float = 0.03,
        max_consecutive_stops: int = 3,
        # 追踪参数
        verbose: bool = False,
        # 入场信号工厂（可注入）
        create_entry_signal: Callable[[int], EntrySignal] | None = None,
        **kwargs: Any,
    ):
        # 指标参数
        self.indicator_params = IndicatorParams(
            atr_period=atr_period,
            adx_period=adx_period,
            adx_smooth=adx_smooth,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            sar_accel=sar_accel,
            sar_max_accel=sar_max_accel,
            rsi_period=rsi_period,
            ma_period=ma_period,
            atr_ma_period=atr_ma_period,
        )

        # Regime 参数
        self.trend_strength_threshold = trend_strength_threshold
        self.trend_strength_grid_max = trend_strength_grid_max
        self.mode_confirm_bars = mode_confirm_bars
        self.mode_cooldown_bars = mode_cooldown_bars
        self.osc_confirm_bars = osc_confirm_bars
        self.vol_high_thresh = vol_high_thresh
        self.vol_low_thresh = vol_low_thresh

        # 网格参数
        self.grid_reset_days = grid_reset_days
        self.grid_stop_loss_pct = grid_stop_loss_pct
        # grid_levels 已弃用：从未传递给 GridStrategy，保留参数仅为向后兼容
        if grid_levels != 5:
            warnings.warn(
                "grid_levels 参数已弃用，不会被使用。将在未来版本移除。",
                DeprecationWarning,
                stacklevel=2,
            )

        # 趋势参数
        self.risk_per_trade = risk_per_trade
        self.initial_stop_atr_mult = initial_stop_atr_mult
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.max_pyramid = max_pyramid
        self.max_position_ratio = max_position_ratio
        self.reversal_ema_gap_pct = reversal_ema_gap_pct
        self.min_hold_bars = min_hold_bars

        # 做空参数
        self.allow_short = allow_short
        self.max_short_ratio = max_short_ratio
        self.short_stop_atr_mult = short_stop_atr_mult
        self.short_squeeze_rsi = short_squeeze_rsi

        # 均值回归参数
        self.mr_enabled = mr_enabled
        self.mr_zscore_entry = mr_zscore_entry
        self.mr_zscore_exit = mr_zscore_exit
        self.mr_zscore_stop = mr_zscore_stop
        self.mr_rsi_oversold = mr_rsi_oversold
        self.mr_rsi_overbought = mr_rsi_overbought
        self.mr_max_position_ratio = mr_max_position_ratio

        # 风控参数
        self.max_drawdown_halt = max_drawdown_halt
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_stops = max_consecutive_stops

        # 追踪参数
        self.verbose = verbose
        self._signal_log: List[SignalTrace] = []

        # 入场信号工厂
        self._create_entry_signal_fn: Callable[[int], EntrySignal] = (
            create_entry_signal or self._default_create_entry_signal
        )

        # PortfolioParams 通过 signal_params 传入，默认为不启用
        # 提取 PortfolioParams 字段，避免被误报为未知参数
        _portfolio_fields = set(PortfolioParams.__dataclass_fields__)
        _portfolio_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _portfolio_fields}
        self.portfolio_params = PortfolioParams(**_portfolio_kwargs)

        # 未知参数告警（防止 config.json 中拼写错误被静默忽略）
        if kwargs:
            warnings.warn(
                f"Unknown signal_params ignored: {list(kwargs.keys())}",
                UserWarning,
                stacklevel=2,
            )

        # 分组参数 dataclass（内部使用，便于整体传递给子组件）
        self.regime_params = RegimeParams(
            trend_strength_threshold=trend_strength_threshold,
            trend_strength_grid_max=trend_strength_grid_max,
            mode_confirm_bars=mode_confirm_bars,
            mode_cooldown_bars=mode_cooldown_bars,
            osc_confirm_bars=osc_confirm_bars,
            vol_high_thresh=vol_high_thresh,
            vol_low_thresh=vol_low_thresh,
        )
        self.grid_params = GridParams(
            grid_reset_days=grid_reset_days,
            grid_stop_loss_pct=grid_stop_loss_pct,
            max_position_ratio=max_position_ratio,
        )
        self.trend_params = TrendParams(
            risk_per_trade=risk_per_trade,
            initial_stop_atr_mult=initial_stop_atr_mult,
            trailing_stop_atr_mult=trailing_stop_atr_mult,
            max_pyramid=max_pyramid,
            max_position_ratio=max_position_ratio,
            reversal_ema_gap_pct=reversal_ema_gap_pct,
            min_hold_bars=min_hold_bars,
        )
        self.short_params = ShortParams(
            allow_short=allow_short,
            max_short_ratio=max_short_ratio,
            short_stop_atr_mult=short_stop_atr_mult,
            short_squeeze_rsi=short_squeeze_rsi,
        )
        self.mr_params = MeanReversionParams(
            zscore_entry=mr_zscore_entry,
            zscore_exit=mr_zscore_exit,
            zscore_stop=mr_zscore_stop,
            rsi_oversold=mr_rsi_oversold,
            rsi_overbought=mr_rsi_overbought,
            max_position_ratio=mr_max_position_ratio,
        )
        self.risk_params = RiskParams(
            max_drawdown_halt=max_drawdown_halt,
            max_daily_loss=max_daily_loss,
            max_consecutive_stops=max_consecutive_stops,
        )

        # 金字塔衰减因子
        self.pyramid_decay = [
            (max_pyramid - j + 1) / (max_pyramid + 1)
            for j in range(1, max_pyramid + 1)
        ]

        # Regime 配置表（子类可覆盖）
        self.regime_configs: dict[int, RegimeConfig] = dict(DEFAULT_REGIME_CONFIGS)

    # ------------------------------------------------------------------
    # 工厂方法（子类可覆盖）
    # ------------------------------------------------------------------

    @staticmethod
    def _default_create_entry_signal(direction: int) -> EntrySignal:
        """默认入场信号工厂：SAR 方向 + DI 交叉。"""
        return SARDirectionEntry(direction=direction)

    def create_entry_signal(self, direction: int) -> EntrySignal:
        """创建入场信号判断器。子类可覆盖以使用不同入场逻辑。

        也可在构造 StrategyBase 时通过 ``create_entry_signal`` 参数注入。

        Args:
            direction: +1 做多, -1 做空。

        Returns:
            EntrySignal 实例。
        """
        return self._create_entry_signal_fn(direction)

    def create_regime_detector(self) -> RegimeDetector:
        """创建 Regime 检测器。"""
        return DefaultRegimeDetector(
            trend_strength_threshold=self.trend_strength_threshold,
            trend_strength_grid_max=self.trend_strength_grid_max,
            confirm_bars=self.mode_confirm_bars,
            cooldown_bars=self.mode_cooldown_bars,
            osc_confirm_bars=self.osc_confirm_bars,
            vol_high_thresh=self.vol_high_thresh,
            vol_low_thresh=self.vol_low_thresh,
        )

    def create_risk_manager(self) -> RiskManager:
        """创建风控管理器。"""
        return RiskManager(
            max_drawdown_halt=self.max_drawdown_halt,
            max_daily_loss=self.max_daily_loss,
            max_consecutive_stops=self.max_consecutive_stops,
        )

    def create_grid_strategy(self) -> GridStrategy:
        """创建网格子策略。"""
        return GridStrategy(
            grid_reset_days=self.grid_reset_days,
            grid_stop_loss_pct=self.grid_stop_loss_pct,
            max_position_ratio=self.max_position_ratio,
        )

    def create_trend_long_strategy(self) -> TrendLongStrategy:
        """创建趋势做多子策略。"""
        return TrendLongStrategy(
            risk_per_trade=self.risk_per_trade,
            initial_stop_atr_mult=self.initial_stop_atr_mult,
            trailing_stop_atr_mult=self.trailing_stop_atr_mult,
            max_pyramid=self.max_pyramid,
            max_position_ratio=self.max_position_ratio,
            reversal_ema_gap_pct=self.reversal_ema_gap_pct,
            min_hold_bars=self.min_hold_bars,
            pyramid_decay=self.pyramid_decay,
            entry_signal=self.create_entry_signal(direction=1),
        )

    def create_trend_short_strategy(self) -> TrendShortStrategy:
        """创建趋势做空子策略。"""
        return TrendShortStrategy(
            risk_per_trade=self.risk_per_trade,
            initial_stop_atr_mult=self.initial_stop_atr_mult,
            short_stop_atr_mult=self.short_stop_atr_mult,
            max_pyramid=self.max_pyramid,
            max_short_ratio=self.max_short_ratio,
            reversal_ema_gap_pct=self.reversal_ema_gap_pct,
            min_hold_bars=self.min_hold_bars,
            short_squeeze_rsi=self.short_squeeze_rsi,
            pyramid_decay=self.pyramid_decay,
            entry_signal=self.create_entry_signal(direction=-1),
        )

    def create_mean_reversion_strategy(self) -> MeanReversionStrategy:
        """创建均值回归子策略。"""
        return MeanReversionStrategy(
            zscore_entry=self.mr_zscore_entry,
            zscore_exit=self.mr_zscore_exit,
            zscore_stop=self.mr_zscore_stop,
            rsi_oversold=self.mr_rsi_oversold,
            rsi_overbought=self.mr_rsi_overbought,
            max_position_ratio=self.mr_max_position_ratio,
            allow_short=self.allow_short,
        )

    def create_indicator_pipeline(self) -> IndicatorPipeline:
        """创建指标管道。子类可覆盖以自定义指标集。"""
        return default_pipeline(self.indicator_params)

    # ------------------------------------------------------------------
    # 钩子方法（子类可覆盖）
    # ------------------------------------------------------------------

    def on_regime_change(self, old_regime: int, new_regime: int) -> None:
        """Regime 切换时的回调钩子。默认空操作。"""

    def _build_regime_features(self, a, i: int) -> dict[str, float]:
        """构建 Regime 检测所需的特征字典。

        默认实现使用 ADX/PDI/MDI/vol_level。
        子类可覆盖此方法来使用完全不同的指标。

        Args:
            a: build_arrays_ns() 返回的 numpy 数组命名空间。
            i: 当前 bar 索引。

        Returns:
            特征字典。
        """
        return build_regime_features_fast(a, i)

    @staticmethod
    def _detect_sub_strategy_label(regime: int, signal: float, allow_short: bool, mr_enabled: bool = False) -> str:
        """根据当前 Regime 和信号方向推断触发信号变化的子策略名称。"""
        if Regime.is_grid_like(regime) and mr_enabled:
            return "mean_reversion"
        if Regime.is_grid_like(regime):
            return "grid"
        if Regime.is_trend_like(regime):
            if signal < 0 and allow_short:
                return "trend_short"
            return "trend_long"
        return "unknown"

    # ------------------------------------------------------------------
    # Regime 配置应用
    # ------------------------------------------------------------------

    def _apply_regime_config(
        self,
        regime: int,
        grid: GridStrategy,
        trend_long: TrendLongStrategy,
        trend_short: TrendShortStrategy,
        mean_reversion: MeanReversionStrategy | None = None,
    ) -> None:
        """根据 Regime 配置调整子策略参数。"""
        cfg = self.regime_configs.get(regime, self.regime_configs.get(Regime.NONE, RegimeConfig()))

        # 网格参数调制
        grid.spacing_scale = cfg.grid_spacing_scale
        grid.max_position_ratio = cfg.max_position_ratio

        # 趋势止损调制
        trend_long.trailing_stop_atr_mult = self.trailing_stop_atr_mult * cfg.trend_stop_scale
        trend_short.short_stop_atr_mult = self.short_stop_atr_mult * cfg.trend_stop_scale

        # 均值回归参数调制
        if mean_reversion is not None:
            mean_reversion.max_position_ratio = cfg.max_position_ratio * 0.6  # 均值回归更保守

    # ------------------------------------------------------------------
    # SignalEngine 合约
    # ------------------------------------------------------------------

    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """生成交易信号（SignalEngine 合约方法）。

        Args:
            data_map: ``{code: DataFrame}``，DataFrame 包含 OHLCV 列。

        Returns:
            ``{code: Series}``，Series 值为目标仓位权重（-0.8~0.8）。
        """
        self._signal_log.clear()
        signals = {}
        for code, df in data_map.items():
            signals[code] = self._generate_single(df, code)

        # 组合级后处理（opt-in: 多标的 + portfolio_params.enabled）
        if (
            len(signals) > 1
            and self.portfolio_params.enabled
        ):
            signals = self._apply_portfolio_constraints(signals, data_map)

        return signals

    def get_signal_log(self) -> pd.DataFrame:
        """导出信号追踪日志为 DataFrame。

        仅在 verbose=True 时有数据。

        Returns:
            DataFrame，列: bar, code, regime, sub_strategy, action, signal, reasons。
        """
        if not self._signal_log:
            return pd.DataFrame(
                columns=["bar", "code", "regime", "sub_strategy", "action", "signal", "reasons"]
            )
        return pd.DataFrame([
            {
                "bar": t.bar,
                "code": t.code,
                "regime": t.regime,
                "sub_strategy": t.sub_strategy,
                "action": t.action,
                "signal": t.signal,
                "reasons": "; ".join(t.reasons),
            }
            for t in self._signal_log
        ])

    # ------------------------------------------------------------------
    # 单标的编排
    # ------------------------------------------------------------------

    def _generate_single(self, df: pd.DataFrame, code: str = "") -> pd.Series:
        """单标的信号生成编排。

        流程:
        1. 计算指标
        2. 判断市场规则（是否允许做空）
        3. 创建子策略实例（确保跨标的状态隔离）
        4. 逐 bar 循环: 风控 → Regime 检测 → 子策略分派 → 安全阀
        """
        n = len(df)
        if n < self.MIN_BARS_REQUIRED:
            return pd.Series(0.0, index=df.index)

        # 1. 计算指标
        pipeline = self.create_indicator_pipeline()
        d = pipeline.compute(df, self.indicator_params)

        # 2. 市场规则: 是否允许做空
        market_allow_short = MarketRules.can_short(code) if code else True
        effective_short = self.allow_short and market_allow_short

        # 3. 创建子策略实例（每次调用新实例，确保状态隔离）
        regime_det = self.create_regime_detector()
        risk_mgr = self.create_risk_manager()
        grid = self.create_grid_strategy()
        trend_long = self.create_trend_long_strategy()
        trend_short = self.create_trend_short_strategy()
        mean_reversion = self.create_mean_reversion_strategy() if self.mr_enabled else None

        # 预提取 numpy 数组
        a = build_arrays_ns(d)

        # 状态变量
        signal_out = np.zeros(n)
        signal = 0.0
        equity_curve = np.ones(n)

        for i in range(self.MIN_BARS_REQUIRED, n):
            # 安全检查 NaN
            if np.isnan(a.atr[i]) or np.isnan(a.adx[i]) or a.atr[i] <= 0:
                signal_out[i] = signal
                continue

            # === 风控检查 ===
            prev_signal = signal_out[i - 1] if i > 0 else 0.0
            is_active, signal_override = risk_mgr.check(i, signal, a.close, equity_curve, prev_signal)
            if not is_active:
                signal = signal_override
                signal_out[i] = signal_override
                if self.verbose:
                    self._signal_log.append(SignalTrace(
                        bar=i, code=code,
                        regime=Regime.label(regime_det.mode),
                        sub_strategy="risk_manager",
                        action="halted" if risk_mgr.halted else "paused",
                        signal=signal_override,
                        reasons=["drawdown_halt" if risk_mgr.halted else "consecutive_stops/daily_loss"],
                    ))
                continue

            # === Regime 检测（通过特征字典，指标无关） ===
            old_regime = regime_det.mode
            features = self._build_regime_features(a, i)
            new_regime = regime_det.detect(features)

            if old_regime != new_regime:
                self.on_regime_change(old_regime, new_regime)
                self._apply_regime_config(new_regime, grid, trend_long, trend_short, mean_reversion)
                if self.verbose:
                    self._signal_log.append(SignalTrace(
                        bar=i, code=code,
                        regime=f"{Regime.label(old_regime)}->{Regime.label(new_regime)}",
                        sub_strategy="regime_detector",
                        action="regime_change",
                        signal=signal,
                        reasons=[f"features={features}"],
                    ))

                if Regime.is_grid_like(new_regime):
                    # 切换到网格类: 重置网格状态
                    cur_close = a.close[i]
                    cur_ma20 = a.ma20[i] if not np.isnan(a.ma20[i]) else cur_close
                    grid.reset(
                        i, cur_close, a.atr[i], a.vol_ratio[i], a.vstd[i],
                        cur_ma20, a.rsi[i], a.sar_dir[i],
                    )
                    # 重置均值回归状态
                    if mean_reversion is not None:
                        mean_reversion.reset()

                elif Regime.is_trend_like(new_regime):
                    # 切换到趋势类: 重置趋势状态
                    trend_long.reset()
                    trend_short.reset()
                    # 趋势中禁用均值回归
                    if mean_reversion is not None:
                        mean_reversion.reset()

            # === 子策略分派 ===
            cfg = self.regime_configs.get(new_regime, self.regime_configs.get(Regime.NONE, RegimeConfig()))

            if Regime.is_grid_like(new_regime) and not cfg.trend_active:
                # 震荡模式
                if self.mr_enabled and cfg.mr_active and mean_reversion is not None:
                    # 均值回归优先（统计信号驱动）
                    signal = mean_reversion.on_bar(i, a, signal)
                    # 若均值回归无信号，fallback 到网格
                    if signal == 0.0:
                        signal = grid.on_bar(i, a, signal)
                else:
                    # 纯网格模式
                    signal = grid.on_bar(i, a, signal)

            elif Regime.is_grid_like(new_regime) and cfg.grid_active and cfg.trend_active:
                # 网格 + 趋势混合（不太常见，留给子类扩展）
                if self.mr_enabled and cfg.mr_active and mean_reversion is not None:
                    signal = mean_reversion.on_bar(i, a, signal)
                    if signal == 0.0:
                        signal = grid.on_bar(i, a, signal)
                else:
                    signal = grid.on_bar(i, a, signal)

            elif Regime.is_trend_like(new_regime):
                # 趋势模式
                if new_regime == Regime.BEAR_TREND:
                    # === 熊市: 做空优先，做多降权 ===
                    if effective_short:
                        short_signal, short_stopped = trend_short.on_bar(i, a, risk_mgr)
                        if short_stopped:
                            signal = 0.0
                            signal_out[i] = 0.0
                            risk_mgr.update_equity(equity_curve[i])
                            continue
                        signal = short_signal
                    else:
                        # 不允许做空 → 清仓观望
                        signal = 0.0

                    # 做多方向（低权重，用于捕捉短暂反弹）
                    if cfg.grid_active or new_regime != Regime.BEAR_TREND:
                        long_signal, long_stopped = trend_long.on_bar(i, a, risk_mgr)
                        if long_stopped:
                            risk_mgr.update_equity(equity_curve[i])
                            # 不停止，只是做多被止损
                        elif abs(long_signal) > abs(signal) and signal >= 0:
                            signal = long_signal * self.BEAR_LONG_WEIGHT_SCALE  # 大幅降权

                else:
                    # === 牛市 / 通用趋势: 做多优先 ===
                    long_signal, long_stopped = trend_long.on_bar(i, a, risk_mgr)
                    if long_stopped:
                        signal = 0.0
                        signal_out[i] = 0.0
                        risk_mgr.update_equity(equity_curve[i])
                        continue

                    signal = long_signal

                    # 做空方向
                    if effective_short:
                        short_signal, short_stopped = trend_short.on_bar(i, a, risk_mgr)
                        if short_stopped:
                            signal = 0.0
                            signal_out[i] = 0.0
                            risk_mgr.update_equity(equity_curve[i])
                            continue

                        # 做多和做空信号互斥: 取绝对值更大者
                        if abs(short_signal) > abs(signal):
                            signal = short_signal

            # 安全阀: 不做空时强制信号非负
            if not effective_short and signal < 0:
                signal = 0.0

            signal_out[i] = signal
            risk_mgr.update_equity(equity_curve[i])

            # 记录 verbose trace（仅当信号变化或有事件时）
            if self.verbose:
                prev_signal = signal_out[i - 1] if i > 0 else 0.0
                if abs(signal - prev_signal) > 1e-6:
                    self._signal_log.append(SignalTrace(
                        bar=i, code=code,
                        regime=Regime.label(new_regime),
                        sub_strategy=self._detect_sub_strategy_label(
                            new_regime, signal, effective_short, self.mr_enabled,
                        ),
                        action="signal_change",
                        signal=signal,
                        reasons=[f"prev={prev_signal:.4f}", f"new={signal:.4f}"],
                    ))

        return pd.Series(signal_out, index=df.index)

    # ------------------------------------------------------------------
    # 组合级约束（多标的后处理）
    # ------------------------------------------------------------------

    def _apply_portfolio_constraints(
        self,
        signals: Dict[str, pd.Series],
        data_map: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """组合级仓位约束后处理。

        在所有标的独立信号生成后执行，按以下顺序应用约束:
        1. 单标的信号裁剪
        2. 总敢口等比缩放
        3. 净敢口等比缩放

        Args:
            signals: 各标的独立信号。
            data_map: 原始数据（供未来相关性计算使用）。

        Returns:
            约束调整后的信号字典。
        """
        pp = self.portfolio_params
        combined = pd.DataFrame(signals)

        # 1. 单标的信号裁剪
        if pp.max_single_weight < 1.0:
            combined = combined.clip(-pp.max_single_weight, pp.max_single_weight)

        # 2. 总敢口约束: sum(|signal|) <= max_gross_exposure
        total_gross = combined.abs().sum(axis=1)
        gross_scale = (total_gross / pp.max_gross_exposure).clip(lower=1.0)
        combined = combined.div(gross_scale, axis=0)

        # 3. 净敢口约束: |sum(signal)| <= max_net_exposure
        total_net = combined.sum(axis=1).abs()
        net_exceeded = total_net > pp.max_net_exposure
        if net_exceeded.any():
            net_scale = (total_net / pp.max_net_exposure).clip(lower=1.0)
            combined = combined.div(net_scale, axis=0)

        return {code: combined[code] for code in signals}
