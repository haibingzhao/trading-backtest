"""子策略组件层。

包含四个独立子策略:
- GridStrategy: 网格交易（低买高卖，不做空）
- TrendLongStrategy: 趋势做多（入场/加仓/止损/反转）
- TrendShortStrategy: 趋势做空（做多镜像 + 逼空保护）
- MeanReversionStrategy: 均值回归（布林带 + RSI 超买超卖）

每个子策略维护独立状态，通过 on_bar 方法接收预提取的 numpy 数组。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Protocol, Union, runtime_checkable

import numpy as np

from backtest.strategy.risk import RiskManager

#: 指标数组类型: SimpleNamespace（属性访问）或 dict（键访问）
IndicatorArrays = Union[SimpleNamespace, dict]


def _arr(a: IndicatorArrays, key: str):
    """从指标数组中取值，兼容 SimpleNamespace 和 dict 两种模式。"""
    if isinstance(a, dict):
        return a[key]
    return getattr(a, key)


# ------------------------------------------------------------------
# EntrySignal 协议及内置实现
# ------------------------------------------------------------------


@runtime_checkable
class EntrySignal(Protocol):
    """入场信号协议。子类实现 should_enter() 方法。"""

    def should_enter(self, i: int, a: IndicatorArrays) -> bool:
        """判断是否应入场。

        Args:
            i: 当前 bar 索引。
            a: 指标数组（SimpleNamespace 或 dict）。

        Returns:
            True 表示满足入场条件。
        """
        ...


class SARDirectionEntry:
    """默认入场信号: SAR 方向 + DI 交叉。

    做多: sar_dir > 0 且 pdi > mdi
    做空: sar_dir < 0 且 mdi > pdi
    """

    def __init__(self, direction: int = 1):
        self._dir = direction

    def should_enter(self, i: int, a: IndicatorArrays) -> bool:
        sar_dir = _arr(a, "sar_dir")
        pdi = _arr(a, "pdi")
        mdi = _arr(a, "mdi")
        if self._dir > 0:
            return sar_dir[i] > 0 and pdi[i] > mdi[i]
        else:
            return sar_dir[i] < 0 and mdi[i] > pdi[i]


class SpreadCrossEntry:
    """价差阈值入场信号（如 EMA 交叉、均线突破等）。

    做多: fast 上穿 slow（金叉）且价差 >= spread_threshold
    做空: fast 下穿 slow（死叉）且价差 >= spread_threshold

    通过 ``fast_key`` / ``slow_key`` 指定使用的指标列名，
    例如默认 ``ema_fast`` / ``ema_slow``，也可切换为 ``ma5`` / ``ma20`` 等。
    """

    def __init__(
        self,
        direction: int = 1,
        spread_threshold: float = 0.0,
        fast_key: str = "ema_fast",
        slow_key: str = "ema_slow",
    ):
        self._dir = direction
        self.spread_threshold = spread_threshold
        self.fast_key = fast_key
        self.slow_key = slow_key

    def should_enter(self, i: int, a: IndicatorArrays) -> bool:
        if i < 1:
            return False
        fast_arr = _arr(a, self.fast_key)
        slow_arr = _arr(a, self.slow_key)
        cur_fast = fast_arr[i]
        cur_slow = slow_arr[i]
        prev_fast = fast_arr[i - 1]
        prev_slow = slow_arr[i - 1]

        if self._dir > 0:
            cross_up = prev_fast <= prev_slow and cur_fast > cur_slow
            spread = (cur_fast - cur_slow) / cur_slow if cur_slow > 0 else 0
            return cross_up and spread >= self.spread_threshold
        else:
            cross_down = prev_fast >= prev_slow and cur_fast < cur_slow
            spread = (cur_slow - cur_fast) / cur_fast if cur_fast > 0 else 0
            return cross_down and spread >= self.spread_threshold


# ------------------------------------------------------------------
# 模块级常量（原硬编码魔数）
# ------------------------------------------------------------------

#: 趋势入场风险缩放因子（risk_per_trade × confidence × 此系数）
ENTRY_RISK_SCALE: float = 2.25
#: 网格下跌保护：ADX 超过此阈值且 MDI > PDI 时阻止网格做多
GRID_BLOCK_ADX_THRESHOLD: float = 30.0


# ------------------------------------------------------------------
# 辅助: 指标数组命名空间（便于 on_bar 传参）
# ------------------------------------------------------------------


def build_arrays_ns(d, as_dict: bool = False) -> SimpleNamespace | dict:
    """从增强 DataFrame 预提取 numpy 数组，构建命名空间或字典。

    对于可能缺失的列（自定义管道），使用 NaN 或默认值填充。

    Args:
        d: compute_indicators() 或 IndicatorPipeline.compute() 返回的 DataFrame。
        as_dict: True 返回 dict[str, np.ndarray]，False 返回 SimpleNamespace。

    Returns:
        SimpleNamespace 或 dict，每个属性/键是一个 numpy 数组。
    """
    def _safe_col(name: str, default=0.0):
        if name in d.columns:
            return d[name].values
        return np.full(len(d), default)

    data = dict(
        close=d["close"].values,
        high=d["high"].values,
        low=d["low"].values,
        atr=_safe_col("atr", np.nan),
        adx=_safe_col("adx", np.nan),
        adxr=_safe_col("adxr", np.nan),
        pdi=_safe_col("pdi", 0.0),
        mdi=_safe_col("mdi", 0.0),
        sar_dir=_safe_col("sar_dir", 0.0),
        sar=_safe_col("sar", 0.0),
        sar_rev=_safe_col("sar_rev", 0.0),
        ema_fast=_safe_col("ema_fast", 0.0),
        ema_slow=_safe_col("ema_slow", 0.0),
        rsi=_safe_col("rsi", 50.0),
        ma20=_safe_col("ma20", np.nan),
        high_20=_safe_col("high_20", np.nan),
        low_20=_safe_col("low_20", np.nan),
        vol_ratio=_safe_col("vol_ratio", 1.0),
        vstd=_safe_col("vstd", 0.0),
        volume=d["volume"].values,
        vol_ma20=_safe_col("vol_ma20", 0.0),
        vol_level=_safe_col("vol_level", 1.0),
        bb_width=_safe_col("bb_width", 0.0),
        momentum_10=_safe_col("momentum_10", 0.0),
        momentum_20=_safe_col("momentum_20", 0.0),
        momentum_60=_safe_col("momentum_60", 0.0),
    )
    if as_dict:
        return data
    return SimpleNamespace(**data)


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------


def calc_grid_spacing(atr: float, vol_ratio: float, vstd: float) -> float:
    """根据 ATR + 量比 + VSTD 计算动态网格间距。"""
    if vol_ratio > 1.8:
        spacing_mult = 1.35
    elif vol_ratio > 1.2:
        spacing_mult = 1.15
    elif vol_ratio >= 0.8:
        spacing_mult = 1.0
    elif vol_ratio >= 0.6:
        spacing_mult = 0.85
    else:
        spacing_mult = 0.75

    if vstd > 0.35:
        spacing_mult *= 1.1
    elif vstd < 0.15:
        spacing_mult *= 0.95

    return atr * spacing_mult


def calc_bias(close: float, ma20: float, atr: float, rsi: float, sar_dir: float) -> float:
    """计算市场偏向 bias（-0.3~0.3）。"""
    bias = 0.0
    if atr > 0:
        price_bias = (close - ma20) / atr * 0.15
        bias += max(-0.3, min(price_bias, 0.3)) * 0.4
    if rsi > 65:
        bias -= 0.08 * 0.4
    elif rsi < 35:
        bias += 0.08 * 0.4
    if sar_dir > 0:
        bias += 0.05 * 0.2
    else:
        bias -= 0.05 * 0.2
    return max(-0.3, min(bias, 0.3))


def calc_confidence_mult(
    adx: float,
    prev_adx: float,
    ema_fast: float,
    prev_ema_fast: float,
    ema_slow: float,
    close: float,
    high20: float,
    vol: float,
    vol_ma20: float,
    rsi: float,
    sar_dir: float,
    direction: int = 1,
    low20: float = float("nan"),
) -> float:
    """计算趋势置信倍数（用于调整建仓规模）。

    Args:
        adx: 当前 ADX 值。
        prev_adx: 前一根 bar ADX 值。
        ema_fast: 当前快 EMA。
        prev_ema_fast: 前一根 bar 快 EMA。
        ema_slow: 当前慢 EMA。
        close: 当前收盘价。
        high20: 20 日高点。
        vol: 当前成交量。
        vol_ma20: 20 日成交量均线。
        rsi: 当前 RSI。
        sar_dir: 当前 SAR 方向（+1/-1）。
        direction: 交易方向，+1 做多，-1 做空。
        low20: 20 日低点（做空方向需要）。

    Returns:
        置信倍数，范围 [0.5, +inf)。
    """
    mult = 1.0
    if adx > 30 and adx > prev_adx:
        mult += 0.1
    if prev_ema_fast < ema_slow and ema_fast > ema_slow:
        mult += 0.15
    # 价格突破 20 日极值：方向感知
    if direction >= 1 and close > high20:
        mult += 0.15
    elif direction <= -1 and not np.isnan(low20) and close < low20:
        mult += 0.15
    if vol_ma20 > 0 and vol > vol_ma20 * 1.5:
        mult += 0.15
    if rsi >= 70:
        mult -= 0.2
    # SAR 方向一致：方向感知
    if direction >= 1 and sar_dir > 0:
        mult += 0.2
    elif direction <= -1 and sar_dir < 0:
        mult += 0.2
    return max(mult, 0.5)


# ==================================================================
# GridStrategy — 网格交易子策略
# ==================================================================


class GridStrategy:
    """网格交易策略（低买高卖，不做空）。

    Args:
        grid_reset_days: 网格重置间隔（bar 数）。
        grid_stop_loss_pct: 网格整体止损百分比。
        max_position_ratio: 最大仓位权重。
    """

    def __init__(
        self,
        grid_reset_days: int = 5,
        grid_stop_loss_pct: float = 0.10,
        max_position_ratio: float = 0.8,
    ):
        self.grid_reset_days = grid_reset_days
        self.grid_stop_loss_pct = grid_stop_loss_pct
        self.max_position_ratio = max_position_ratio
        self.spacing_scale: float = 1.0  # 由 RegimeConfig 动态设置

        # 运行时状态
        self.base_price: float = 0.0
        self.buy_price: float = 0.0
        self.sell_price: float = 0.0
        self.spacing: float = 0.0
        self.last_reset: int = -999

    def reset(
        self,
        bar_idx: int,
        close: float,
        atr: float,
        vol_ratio: float,
        vstd: float,
        ma20: float,
        rsi: float,
        sar_dir: float,
    ) -> None:
        """初始化/重置网格状态。"""
        self.base_price = close
        self.last_reset = bar_idx
        self.spacing = calc_grid_spacing(atr, vol_ratio, vstd) * self.spacing_scale
        bias = calc_bias(close, ma20, atr, rsi, sar_dir)
        buy_mult = max(0.6, min(1.0 - bias, 1.5))
        sell_mult = max(0.6, min(1.0 + bias, 1.5))
        self.buy_price = close - self.spacing * buy_mult
        self.sell_price = close + self.spacing * sell_mult

    def on_bar(
        self,
        i: int,
        a: SimpleNamespace,
        signal: float,
    ) -> float:
        """处理单根 bar，返回信号权重。

        Args:
            i: 当前 bar 索引。
            a: 指标数组命名空间。
            signal: 当前信号（用于判断持仓状态）。

        Returns:
            新的信号权重（永远 ≥ 0）。
        """
        cur_close = a.close[i]
        cur_atr = a.atr[i]
        cur_adx = a.adx[i]
        cur_mdi = a.mdi[i]
        cur_pdi = a.pdi[i]
        cur_vol_ratio = a.vol_ratio[i]
        cur_vstd = a.vstd[i]
        cur_ma20 = a.ma20[i] if not np.isnan(a.ma20[i]) else cur_close
        cur_rsi = a.rsi[i]
        cur_sar_dir = a.sar_dir[i]

        # 检查网格重置
        if i - self.last_reset >= self.grid_reset_days:
            self.reset(i, cur_close, cur_atr, cur_vol_ratio, cur_vstd, cur_ma20, cur_rsi, cur_sar_dir)

        # 网格下跌保护（ADX > 30 且 MDI > PDI 时阻止做多）
        grid_blocked = False
        if cur_adx > GRID_BLOCK_ADX_THRESHOLD and cur_mdi > cur_pdi:
            grid_blocked = True

        # 网格整体止损
        if self.base_price > 0 and (cur_close - self.base_price) / self.base_price < -self.grid_stop_loss_pct:
            return 0.0

        # 网格信号判断（仅做多，网格不触发做空）
        if cur_close <= self.buy_price and not grid_blocked:
            return self.max_position_ratio
        elif cur_close >= self.sell_price:
            return 0.0  # 仅平仓多头，不开空
        elif signal > 0.1 and cur_close > self.base_price + self.spacing * 0.5:
            return 0.0

        return signal


# ==================================================================
# BaseTrendStrategy — 趋势子策略抽象基类（方向参数化）
# ==================================================================


class BaseTrendStrategy:
    """趋势子策略抽象基类，通过 ``direction`` 参数统一做多/做空逻辑。

    子类仅需实例化并指定 direction (+1 做多, -1 做空)。

    Args:
        direction: +1 做多, -1 做空。
        risk_per_trade: 每笔交易风险比例。
        initial_stop_atr_mult: 初始止损 ATR 倍数。
        trailing_stop_atr_mult: 移动止损 ATR 倍数。
        max_pyramid: 最大加仓次数。
        max_position_ratio: 最大仓位权重。
        reversal_ema_gap_pct: 反转 EMA 间距阈值。
        min_hold_bars: 最小持仓 bar 数。
        squeeze_rsi: 逼空/逼多 RSI 阈值（仅 direction=-1 时使用）。
    """

    def __init__(
        self,
        direction: int,
        risk_per_trade: float = 0.03,
        initial_stop_atr_mult: float = 3.0,
        trailing_stop_atr_mult: float = 2.0,
        max_pyramid: int = 4,
        max_position_ratio: float = 0.8,
        reversal_ema_gap_pct: float = 0.003,
        min_hold_bars: int = 2,
        squeeze_rsi: float = 75.0,
        pyramid_decay: list | None = None,
        entry_signal: EntrySignal | None = None,
    ):
        self._dir = direction
        self.risk_per_trade = risk_per_trade
        self.initial_stop_atr_mult = initial_stop_atr_mult
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.max_pyramid = max_pyramid
        self.max_position_ratio = max_position_ratio
        self.reversal_ema_gap_pct = reversal_ema_gap_pct
        self.min_hold_bars = min_hold_bars
        self.squeeze_rsi = squeeze_rsi
        self.pyramid_decay = pyramid_decay or [
            (max_pyramid - j + 1) / (max_pyramid + 1)
            for j in range(1, max_pyramid + 1)
        ]
        self.entry_signal: EntrySignal = entry_signal or SARDirectionEntry(direction=direction)

        # 运行时状态
        self.weight: float = 0.0
        self.stop_price: float = 0.0 if direction > 0 else float("inf")
        self.extreme: float = 0.0 if direction > 0 else float("inf")
        self.add_count: int = 0
        self.entry_bar: int = -999
        self.initial_size: float = 0.0

    def reset(self) -> None:
        """重置所有运行时状态。"""
        self.weight = 0.0
        self.stop_price = 0.0 if self._dir > 0 else float("inf")
        self.extreme = 0.0 if self._dir > 0 else float("inf")
        self.add_count = 0
        self.entry_bar = -999
        self.initial_size = 0.0

    def on_bar(
        self,
        i: int,
        a: SimpleNamespace,
        risk_mgr: RiskManager,
    ) -> tuple[float, bool]:
        """处理单根 bar，返回 (信号权重, 是否被止损/反转)。

        Args:
            i: 当前 bar 索引。
            a: 指标数组命名空间。
            risk_mgr: 风控管理器。

        Returns:
            (signal, was_stopped):
            - signal: 信号权重（做多为正，做空为负，0 表示无仓位）。
            - was_stopped: True 表示本次触发了止损/反转/逼空。
        """
        cur_close = a.close[i]
        cur_atr = a.atr[i]
        cur_pdi = a.pdi[i]
        cur_mdi = a.mdi[i]
        cur_sar_dir = a.sar_dir[i]
        cur_ema_fast = a.ema_fast[i]
        cur_ema_slow = a.ema_slow[i]
        cur_high20 = a.high_20[i] if not np.isnan(a.high_20[i]) else cur_close
        cur_low20 = a.low_20[i] if not np.isnan(a.low_20[i]) else cur_close
        cur_vol = a.volume[i]
        cur_vol_ma20 = a.vol_ma20[i] if not np.isnan(a.vol_ma20[i]) else cur_vol
        cur_rsi = a.rsi[i]
        cur_adx = a.adx[i]

        prev_adx = a.adx[i - 1] if i > 0 else cur_adx
        prev_ema_fast = a.ema_fast[i - 1] if i > 0 else cur_ema_fast
        prev_sar_dir = a.sar_dir[i - 1] if i > 0 else cur_sar_dir

        # === 逼空/逼多保护（仅做空方向） ===
        if self._dir < 0 and self.weight > 0.1 and cur_rsi > self.squeeze_rsi:
            self.reset()
            risk_mgr.record_stop()
            return 0.0, True

        # === 更新极值 ===
        if self._dir > 0:
            self.extreme = max(self.extreme, a.high[i])
        else:
            if a.low[i] < self.extreme:
                self.extreme = a.low[i]

        # === 移动止损 ===
        atr_mult = self.initial_stop_atr_mult if self.weight < 0.1 else self.trailing_stop_atr_mult
        if self._dir > 0:
            new_stop = self.extreme - atr_mult * cur_atr
            if new_stop > self.stop_price:
                self.stop_price = new_stop
        else:
            new_stop = self.extreme + atr_mult * cur_atr
            if new_stop < self.stop_price:
                self.stop_price = new_stop

        # === 检查止损 ===
        if self.weight > 0.1:
            stopped = (self._dir > 0 and cur_close <= self.stop_price) or (
                self._dir < 0 and cur_close >= self.stop_price
            )
            if stopped:
                self.reset()
                risk_mgr.record_stop()
                return 0.0, True

        # === 检查反转 ===
        bars_held = i - self.entry_bar if self.entry_bar > 0 else 999
        reversal = False
        if bars_held >= self.min_hold_bars:
            if self._dir > 0:
                ema_gap = (cur_ema_slow - cur_ema_fast) / cur_ema_slow if cur_ema_slow > 0 else 0
                if cur_ema_fast < cur_ema_slow and ema_gap > self.reversal_ema_gap_pct:
                    reversal = True
                if cur_sar_dir == -1 and prev_sar_dir == 1:
                    reversal = True
            else:
                ema_gap = (cur_ema_fast - cur_ema_slow) / cur_ema_fast if cur_ema_fast > 0 else 0
                if cur_ema_fast > cur_ema_slow and ema_gap > self.reversal_ema_gap_pct:
                    reversal = True
                if cur_sar_dir == 1 and prev_sar_dir == -1:
                    reversal = True

        if reversal and self.weight > 0.1:
            self.reset()
            risk_mgr.record_stop()
            return 0.0, True

        # === 入场/加仓 ===
        trend_signal = self.entry_signal.should_enter(i, a)

        if self.weight < 0.1:
            if trend_signal:
                size_mult = calc_confidence_mult(
                    cur_adx, prev_adx, cur_ema_fast, prev_ema_fast,
                    cur_ema_slow, cur_close, cur_high20,
                    cur_vol, cur_vol_ma20, cur_rsi, cur_sar_dir,
                    direction=self._dir, low20=cur_low20,
                )
                risk_amount = self.risk_per_trade * size_mult * ENTRY_RISK_SCALE
                stop_dist = self.initial_stop_atr_mult * cur_atr
                if stop_dist > 0 and cur_close > 0:
                    w = min(risk_amount * cur_close / stop_dist, self.max_position_ratio)
                    self.weight = w
                    self.initial_size = w
                    self.entry_bar = i
                    self.extreme = a.high[i] if self._dir > 0 else a.low[i]
                    if self._dir > 0:
                        self.stop_price = self.extreme - self.initial_stop_atr_mult * cur_atr
                    else:
                        self.stop_price = self.extreme + self.initial_stop_atr_mult * cur_atr
                    self.add_count = 0
                    risk_mgr.reset_stops()
                    return self._dir * w, False
        else:
            if trend_signal and self.add_count < self.max_pyramid:
                decay = self.pyramid_decay[self.add_count]
                add_w = self.initial_size * decay
                new_w = min(self.weight + add_w, self.max_position_ratio)
                if new_w > self.weight + 0.01:
                    self.weight = new_w
                    self.add_count += 1
                    self.entry_bar = i
                    return self._dir * new_w, False

        return self._dir * self.weight, False


# ==================================================================
# TrendLongStrategy — 趋势做多子策略（BaseTrendStrategy 薄包装）
# ==================================================================


class TrendLongStrategy(BaseTrendStrategy):
    """趋势做多策略（入场/加仓/止损/反转）。

    等价于 ``BaseTrendStrategy(direction=+1, ...)``。
    """

    def __init__(
        self,
        risk_per_trade: float = 0.03,
        initial_stop_atr_mult: float = 3.0,
        trailing_stop_atr_mult: float = 2.0,
        max_pyramid: int = 4,
        max_position_ratio: float = 0.8,
        reversal_ema_gap_pct: float = 0.003,
        min_hold_bars: int = 2,
        pyramid_decay: list | None = None,
        entry_signal: EntrySignal | None = None,
    ):
        super().__init__(
            direction=1,
            risk_per_trade=risk_per_trade,
            initial_stop_atr_mult=initial_stop_atr_mult,
            trailing_stop_atr_mult=trailing_stop_atr_mult,
            max_pyramid=max_pyramid,
            max_position_ratio=max_position_ratio,
            reversal_ema_gap_pct=reversal_ema_gap_pct,
            min_hold_bars=min_hold_bars,
            pyramid_decay=pyramid_decay,
            entry_signal=entry_signal,
        )


# ==================================================================
# TrendShortStrategy — 趋势做空子策略（BaseTrendStrategy 薄包装）
# ==================================================================


class TrendShortStrategy(BaseTrendStrategy):
    """趋势做空策略（做多镜像 + 逼空保护）。

    等价于 ``BaseTrendStrategy(direction=-1, ...)``。
    """

    def __init__(
        self,
        risk_per_trade: float = 0.03,
        initial_stop_atr_mult: float = 3.0,
        short_stop_atr_mult: float = 2.5,
        max_pyramid: int = 4,
        max_short_ratio: float = 0.5,
        reversal_ema_gap_pct: float = 0.003,
        min_hold_bars: int = 2,
        short_squeeze_rsi: float = 75,
        pyramid_decay: list | None = None,
        entry_signal: EntrySignal | None = None,
    ):
        super().__init__(
            direction=-1,
            risk_per_trade=risk_per_trade,
            initial_stop_atr_mult=initial_stop_atr_mult,
            trailing_stop_atr_mult=short_stop_atr_mult,
            max_pyramid=max_pyramid,
            max_position_ratio=max_short_ratio,
            reversal_ema_gap_pct=reversal_ema_gap_pct,
            min_hold_bars=min_hold_bars,
            squeeze_rsi=short_squeeze_rsi,
            pyramid_decay=pyramid_decay,
            entry_signal=entry_signal,
        )


# ==================================================================
# MeanReversionStrategy — 均值回归子策略（布林带 + RSI）
# ==================================================================


class MeanReversionStrategy:
    """均值回归子策略（布林带 + RSI 超买超卖）。

    入场条件（做多）:
    - 价格跌破布林带下轨（close < ma20 - zscore_entry * atr）
    - RSI < rsi_oversold（默认 30）

    入场条件（做空）:
    - 价格涨破布林带上轨（close > ma20 + zscore_entry * atr）
    - RSI > rsi_overbought（默认 70）

    出场条件:
    - 价格回归到均线附近（close 回到 ma20 ± zscore_exit * atr）
    - 或止损触发（价格继续偏离超过 zscore_stop * atr）

    Args:
        zscore_entry: 入场 Z-Score 阈值（价格偏离均线几倍 ATR）。
        zscore_exit: 出场 Z-Score 阈值（回归均线附近平仓）。
        zscore_stop: 止损 Z-Score（继续偏离则止损）。
        rsi_oversold: RSI 超卖阈值。
        rsi_overbought: RSI 超买阈值。
        max_position_ratio: 最大仓位权重。
        allow_short: 是否允许做空。
    """

    def __init__(
        self,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        zscore_stop: float = 3.0,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        max_position_ratio: float = 0.5,
        allow_short: bool = False,
    ):
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.zscore_stop = zscore_stop
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.max_position_ratio = max_position_ratio
        self.allow_short = allow_short

        # 运行时状态
        self.weight: float = 0.0       # 当前仓位权重（正=多，负=空）
        self.direction: int = 0         # 当前持仓方向（+1/-1/0）
        self.entry_price: float = 0.0   # 入场价格

    def reset(self) -> None:
        """重置所有运行时状态。"""
        self.weight = 0.0
        self.direction = 0
        self.entry_price = 0.0

    def on_bar(
        self,
        i: int,
        a: SimpleNamespace,
        signal: float,
    ) -> float:
        """处理单根 bar，返回信号权重。

        Args:
            i: 当前 bar 索引。
            a: 指标数组命名空间。
            signal: 当前信号（来自其他子策略或上一 bar）。

        Returns:
            新的信号权重（做多为正，做空为负，0 表示无仓位）。
        """
        cur_close = a.close[i]
        cur_atr = a.atr[i]
        cur_ma20 = a.ma20[i] if not np.isnan(a.ma20[i]) else cur_close
        cur_rsi = a.rsi[i]

        # 跳过无效 ATR
        if cur_atr <= 0:
            return signal

        # 计算 Z-Score（价格偏离均线的标准化距离）
        zscore = (cur_close - cur_ma20) / cur_atr

        # === 持仓中：检查出场/止损 ===
        if self.direction != 0:
            # 止损检查：价格继续偏离超过 zscore_stop
            if self.direction > 0 and zscore <= -self.zscore_stop:
                self.reset()
                return 0.0
            if self.direction < 0 and zscore >= self.zscore_stop:
                self.reset()
                return 0.0

            # 出场检查：价格回归均线附近
            if self.direction > 0 and zscore >= -self.zscore_exit:
                self.reset()
                return 0.0
            if self.direction < 0 and zscore <= self.zscore_exit:
                self.reset()
                return 0.0

            # 继续持仓
            return self.direction * self.weight

        # === 无仓位：检查入场条件 ===

        # 做多入场：价格跌破下轨 + RSI 超卖
        if zscore <= -self.zscore_entry and cur_rsi < self.rsi_oversold:
            self.direction = 1
            self.weight = self.max_position_ratio
            self.entry_price = cur_close
            return self.weight

        # 做空入场：价格涨破上轨 + RSI 超买
        if self.allow_short and zscore >= self.zscore_entry and cur_rsi > self.rsi_overbought:
            self.direction = -1
            self.weight = self.max_position_ratio
            self.entry_price = cur_close
            return -self.weight

        return signal
