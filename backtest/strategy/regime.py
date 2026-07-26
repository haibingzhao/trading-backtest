"""市场 Regime 检测层。

提供:
- Regime: 市场状态常量（指标无关）
- RegimeDetector: 检测器抽象协议（指标无关）
- DefaultRegimeDetector: 基于 ADX+DI+vol_level 的默认实现
- RegimeConfig: 每种 Regime 的策略参数调制配置
"""

from __future__ import annotations

from dataclasses import dataclass


# ==================================================================
# Regime 常量（指标无关，纯状态标签）
# ==================================================================


class Regime:
    """市场 Regime 常量。"""

    NONE = 0          # 初始 / 未确定
    BULL_TREND = 1    # 单边牛市
    BEAR_TREND = 2    # 单边熊市
    HIGH_VOL_OSC = 3  # 高波动震荡
    LOW_VOL_CONV = 4  # 低波动收敛
    NORMAL_OSC = 5    # 普通震荡

    _GRID_LIKE = frozenset({3, 4, 5})
    _TREND_LIKE = frozenset({1, 2})

    @classmethod
    def is_grid_like(cls, regime: int) -> bool:
        """判断是否为震荡 / 网格类 Regime。"""
        return regime in cls._GRID_LIKE

    @classmethod
    def is_trend_like(cls, regime: int) -> bool:
        """判断是否为趋势类 Regime。"""
        return regime in cls._TREND_LIKE

    @classmethod
    def label(cls, regime: int) -> str:
        """返回 Regime 的可读标签。"""
        _LABELS = {
            0: "NONE",
            1: "BULL_TREND", 2: "BEAR_TREND",
            3: "HIGH_VOL_OSC", 4: "LOW_VOL_CONV", 5: "NORMAL_OSC",
        }
        return _LABELS.get(regime, f"UNKNOWN({regime})")


# ==================================================================
# RegimeConfig — 每种 Regime 的策略参数调制
# ==================================================================


@dataclass(frozen=True)
class RegimeConfig:
    """Regime 策略参数调制配置。

    用于在 Regime 切换时调整子策略的行为参数。
    """

    grid_spacing_scale: float = 1.0    # 网格间距乘数
    max_position_ratio: float = 0.8    # 最大仓位
    trend_stop_scale: float = 1.0      # 趋势止损 ATR 倍数乘数
    trend_active: bool = False         # 是否启用趋势策略
    grid_active: bool = True           # 是否启用网格策略
    mr_active: bool = False            # 是否启用均值回归策略


DEFAULT_REGIME_CONFIGS: dict[int, RegimeConfig] = {
    Regime.NONE: RegimeConfig(),
    Regime.BULL_TREND: RegimeConfig(
        grid_spacing_scale=1.3,
        trend_stop_scale=1.2,
        max_position_ratio=0.8,
        trend_active=True,
        grid_active=True,   # 网格辅助
        mr_active=False,    # 趋势中不启用均值回归
    ),
    Regime.BEAR_TREND: RegimeConfig(
        max_position_ratio=0.3,
        trend_stop_scale=0.9,
        trend_active=True,
        grid_active=False,
        mr_active=False,    # 趋势中不启用均值回归
    ),
    Regime.HIGH_VOL_OSC: RegimeConfig(
        grid_spacing_scale=1.5,
        max_position_ratio=0.5,
        mr_active=True,     # 高波动震荡启用均值回归
    ),
    Regime.LOW_VOL_CONV: RegimeConfig(
        grid_spacing_scale=0.7,
        max_position_ratio=0.4,
        mr_active=True,     # 低波动收敛启用均值回归
    ),
    Regime.NORMAL_OSC: RegimeConfig(
        mr_active=True,     # 普通震荡启用均值回归
    ),
}


# ==================================================================
# RegimeDetector — 抽象协议（指标无关）
# ==================================================================


class RegimeDetector:
    """市场 Regime 检测器协议（指标无关）。

    子类通过 ``detect(features)`` 接受一个通用特征字典，
    不绑定任何特定技术指标。具体指标的使用是实现细节。

    Attributes:
        mode: 当前 Regime 值。
    """

    def __init__(
        self,
        confirm_bars: int = 3,
        cooldown_bars: int = 2,
    ) -> None:
        self.mode: int = Regime.NONE
        self._confirm_bars = confirm_bars
        self._cooldown_bars = cooldown_bars
        self._pending_regime: int = Regime.NONE
        self._confirm_count: int = 0
        self._cooldown: int = 0

    def detect(self, features: dict[str, float]) -> int:
        """根据特征字典检测并返回当前 Regime。

        Args:
            features: 特征字典，键为特征名，值为浮点数。
                具体键名由实现类定义。

        Returns:
            Regime 常量值。
        """
        raise NotImplementedError

    def reset(self) -> None:
        """重置所有运行时状态。"""
        self.mode = Regime.NONE
        self._pending_regime = Regime.NONE
        self._confirm_count = 0
        self._cooldown = 0

    def _apply_debounce(self, target_regime: int) -> int:
        """通用防抖: confirm + cooldown。

        子类在确定目标 Regime 后调用此方法，由基类统一管理
        确认计数和冷却逻辑。

        Args:
            target_regime: 原始检测结果（未经防抖）。

        Returns:
            经防抖后的实际 Regime（可能尚未切换）。
        """
        if self._cooldown > 0:
            self._cooldown -= 1

        if target_regime != self.mode:
            if target_regime == self._pending_regime:
                self._confirm_count += 1
            else:
                self._pending_regime = target_regime
                self._confirm_count = 1

            if self._confirm_count >= self._confirm_bars and self._cooldown <= 0:
                self.mode = target_regime
                self._cooldown = self._cooldown_bars
                self._confirm_count = 0
                self._pending_regime = Regime.NONE
        else:
            # 目标与当前一致，重置确认计数
            self._confirm_count = 0

        return self.mode


# ==================================================================
# DefaultRegimeDetector — 基于趋势强度 + 方向 + 波动率的默认实现
# ==================================================================


class DefaultRegimeDetector(RegimeDetector):
    """默认 Regime 检测器。

    使用三个特征维度:
    - ``trend_strength``: 趋势强度（默认实现中使用 ADX）。
    - ``trend_up``: 趋势方向（默认实现中使用 PDI > MDI）。
    - ``vol_level``: 波动率水平（默认实现中使用 ATR/MA60）。

    状态机:
    - 趋势判定: trend_strength >= threshold + direction → BULL/BEAR_TREND
    - 震荡细分: 按 vol_level 分为 HIGH_VOL_OSC / LOW_VOL_CONV / NORMAL_OSC
    - 死区: 保持当前 mode
    - confirm/cooldown 机制防止频繁切换

    Args:
        trend_strength_threshold: 趋势强度阈值（默认 25，对应 ADX）。
        trend_strength_grid_max: 震荡判定上限（默认 20，低于此为震荡）。
        confirm_bars: 趋势切换所需连续确认 bar 数。
        cooldown_bars: 切换后冷却 bar 数。
        osc_confirm_bars: 震荡子状态切换所需确认 bar 数（防抖）。
        vol_high_thresh: 高波动阈值（vol_level >= 此值为 HIGH_VOL_OSC）。
        vol_low_thresh: 低波动阈值（vol_level <= 此值为 LOW_VOL_CONV）。
    """

    def __init__(
        self,
        trend_strength_threshold: float = 25,
        trend_strength_grid_max: float = 20,
        confirm_bars: int = 3,
        cooldown_bars: int = 2,
        osc_confirm_bars: int = 2,
        vol_high_thresh: float = 1.3,
        vol_low_thresh: float = 0.7,
    ):
        super().__init__()
        self.trend_strength_threshold = trend_strength_threshold
        self.trend_strength_grid_max = trend_strength_grid_max
        self.confirm_bars = confirm_bars
        self.cooldown_bars = cooldown_bars
        self.osc_confirm_bars = osc_confirm_bars
        self.vol_high_thresh = vol_high_thresh
        self.vol_low_thresh = vol_low_thresh

        # 运行时状态
        self.confirm_count: int = 0   # 正=向震荡方向，负=向趋势方向
        self.cooldown: int = 0
        self.osc_confirm_count: int = 0
        self._pending_osc: int = Regime.NORMAL_OSC

    def reset(self) -> None:
        """重置所有运行时状态。"""
        super().reset()
        self.confirm_count = 0
        self.cooldown = 0
        self.osc_confirm_count = 0
        self._pending_osc = Regime.NORMAL_OSC

    def detect(self, features: dict[str, float]) -> int:
        """根据特征字典检测 Regime。

        Args:
            features: 至少包含 ``trend_strength`` 的字典。
                可选键: ``trend_up``（bool/float）, ``vol_level``（float, 默认 1.0）。

        Returns:
            Regime 常量值。
        """
        trend_strength = features.get("trend_strength", 0.0)
        trend_up = bool(features.get("trend_up", True))
        vol_level = features.get("vol_level", 1.0)

        if self.cooldown > 0:
            self.cooldown -= 1

        new_mode = self.mode

        if trend_strength >= self.trend_strength_threshold:
            # --- 向趋势方向确认 ---
            self.confirm_count = self.confirm_count - 1 if self.confirm_count <= 0 else -1
            if self.confirm_count <= -self.confirm_bars and self.cooldown <= 0:
                target = Regime.BULL_TREND if trend_up else Regime.BEAR_TREND
                if self.mode != target:
                    new_mode = target
                    self.cooldown = self.cooldown_bars
                    self.confirm_count = 0
                    self.osc_confirm_count = 0

        elif trend_strength <= self.trend_strength_grid_max:
            # --- 向震荡方向确认 ---
            self.confirm_count = self.confirm_count + 1 if self.confirm_count >= 0 else 1
            if self.confirm_count >= self.confirm_bars and self.cooldown <= 0:
                # 确定目标震荡子状态
                if vol_level >= self.vol_high_thresh:
                    target_osc = Regime.HIGH_VOL_OSC
                elif vol_level <= self.vol_low_thresh:
                    target_osc = Regime.LOW_VOL_CONV
                else:
                    target_osc = Regime.NORMAL_OSC

                if Regime.is_grid_like(self.mode):
                    # 已在震荡中 → 子状态切换需要额外确认
                    if target_osc != self.mode:
                        if target_osc == self._pending_osc:
                            self.osc_confirm_count += 1
                        else:
                            self._pending_osc = target_osc
                            self.osc_confirm_count = 1

                        if self.osc_confirm_count >= self.osc_confirm_bars:
                            new_mode = target_osc
                            self.osc_confirm_count = 0
                            self.cooldown = self.cooldown_bars
                    else:
                        self.osc_confirm_count = 0
                else:
                    # 从趋势切换到震荡
                    new_mode = target_osc
                    self.cooldown = self.cooldown_bars
                    self.confirm_count = 0
                    self.osc_confirm_count = 0

        # 死区（grid_max < strength < threshold）：保持当前 mode

        self.mode = new_mode
        return self.mode


# ==================================================================
# MACrossoverRegimeDetector — 基于 EMA 交叉 + ADX 过滤
# ==================================================================


class MACrossoverRegimeDetector(RegimeDetector):
    """基于 EMA 快/慢线交叉的 Regime 检测器。

    金叉（EMA_fast 上穿 EMA_slow）且 ADX 足够高 → 趋势，
    否则按波动率细分为震荡子状态。

    特征需求: ema_fast, ema_slow, trend_strength(ADX), vol_level
    """

    def __init__(
        self,
        gap_pct_threshold: float = 0.02,
        adx_filter: float = 20.0,
        vol_high_thresh: float = 1.3,
        vol_low_thresh: float = 0.7,
        confirm_bars: int = 3,
        cooldown_bars: int = 2,
    ):
        super().__init__()
        self.gap_pct_threshold = gap_pct_threshold
        self.adx_filter = adx_filter
        self.vol_high_thresh = vol_high_thresh
        self.vol_low_thresh = vol_low_thresh
        self.confirm_bars = confirm_bars
        self.cooldown_bars = cooldown_bars
        self._confirm = 0
        self._cooldown = 0

    def reset(self) -> None:
        super().reset()
        self._confirm = 0
        self._cooldown = 0

    def detect(self, features: dict) -> int:
        ema_fast = features.get("ema_fast", 0.0)
        ema_slow = features.get("ema_slow", 0.0)
        adx = features.get("trend_strength", 0.0)
        vol_level = features.get("vol_level", 1.0)

        if self._cooldown > 0:
            self._cooldown -= 1

        gap_pct = (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0.0

        new_mode = self.mode

        if adx >= self.adx_filter:
            if gap_pct > self.gap_pct_threshold:
                # 金叉 + ADX 足够高 → 牛市
                self._confirm = self._confirm - 1 if self._confirm <= 0 else -1
                if self._confirm <= -self.confirm_bars and self._cooldown <= 0:
                    new_mode = Regime.BULL_TREND
                    self._cooldown = self.cooldown_bars
                    self._confirm = 0
            elif gap_pct < -self.gap_pct_threshold:
                # 死叉 + ADX 足够高 → 熊市
                self._confirm = self._confirm + 1 if self._confirm >= 0 else 1
                if self._confirm >= self.confirm_bars and self._cooldown <= 0:
                    new_mode = Regime.BEAR_TREND
                    self._cooldown = self.cooldown_bars
                    self._confirm = 0
        else:
            # ADX 不足 → 震荡，按波动率细分
            self._confirm = 0
            if vol_level >= self.vol_high_thresh:
                new_mode = Regime.HIGH_VOL_OSC
            elif vol_level <= self.vol_low_thresh:
                new_mode = Regime.LOW_VOL_CONV
            else:
                new_mode = Regime.NORMAL_OSC

        self.mode = new_mode
        return self.mode


# ==================================================================
# BollingerSqueezeRegimeDetector — 基于布林带宽度收敛/扩张
# ==================================================================


class BollingerSqueezeRegimeDetector(RegimeDetector):
    """基于布林带宽度（bb_width）的 Regime 检测器。

    - bb_width 持续收窄 → LOW_VOL_CONV（挤压，潜在突破）
    - bb_width 扩张 + 价格方向 → BULL/BEAR_TREND
    - bb_width 持续高位 → HIGH_VOL_OSC
    - 其他 → NORMAL_OSC

    特征需求: bb_width, close, ema_fast, ema_slow, trend_up, vol_level
    """

    def __init__(
        self,
        squeeze_threshold: float = 0.04,
        expand_threshold: float = 0.08,
        high_vol_threshold: float = 0.15,
        vol_high_thresh: float = 1.3,
        vol_low_thresh: float = 0.7,
        trend_strength_fallback: float = 25.0,
        confirm_bars: int = 3,
        cooldown_bars: int = 2,
    ):
        super().__init__(confirm_bars=confirm_bars, cooldown_bars=cooldown_bars)
        self.squeeze_threshold = squeeze_threshold
        self.expand_threshold = expand_threshold
        self.high_vol_threshold = high_vol_threshold
        self.vol_high_thresh = vol_high_thresh
        self.vol_low_thresh = vol_low_thresh
        self.trend_strength_fallback = trend_strength_fallback
        self._prev_bb_width = 0.0

    def reset(self) -> None:
        super().reset()
        self._prev_bb_width = 0.0

    def detect(self, features: dict) -> int:
        bb_width = features.get("bb_width", 0.0)
        trend_up = bool(features.get("trend_up", True))
        vol_level = features.get("vol_level", 1.0)
        trend_strength = features.get("trend_strength", 0.0)

        new_mode = self.mode
        bb_expanding = bb_width > self._prev_bb_width
        bb_contracting = bb_width < self._prev_bb_width

        if bb_width < self.squeeze_threshold and bb_contracting:
            # 布林带挤压 → 低波动收敛
            new_mode = Regime.LOW_VOL_CONV
        elif bb_width > self.expand_threshold and bb_expanding:
            # 布林带扩张 + ADX 确认 → 趋势
            if trend_strength >= self.trend_strength_fallback:
                new_mode = Regime.BULL_TREND if trend_up else Regime.BEAR_TREND
            else:
                new_mode = Regime.HIGH_VOL_OSC
        elif bb_width > self.high_vol_threshold:
            # 布林带持续宽幅 → 高波动震荡
            new_mode = Regime.HIGH_VOL_OSC
        elif vol_level <= self.vol_low_thresh:
            new_mode = Regime.LOW_VOL_CONV
        elif vol_level >= self.vol_high_thresh:
            new_mode = Regime.HIGH_VOL_OSC
        else:
            new_mode = Regime.NORMAL_OSC

        self._prev_bb_width = bb_width
        return self._apply_debounce(new_mode)


# ==================================================================
# MomentumRegimeDetector — 基于多时间框架动量一致性
# ==================================================================


class MomentumRegimeDetector(RegimeDetector):
    """基于多时间框架动量共振的 Regime 检测器。

    三个动量指标（10/20/60日）方向一致时为强趋势，
    方向分歧时为震荡。

    特征需求: momentum_10, momentum_20, momentum_60, vol_level, trend_strength
    """

    def __init__(
        self,
        vol_high_thresh: float = 1.3,
        vol_low_thresh: float = 0.7,
        divergence_vol_thresh: float = 1.3,
        confirm_bars: int = 3,
        cooldown_bars: int = 2,
    ):
        super().__init__(confirm_bars=confirm_bars, cooldown_bars=cooldown_bars)
        self.vol_high_thresh = vol_high_thresh
        self.vol_low_thresh = vol_low_thresh
        self.divergence_vol_thresh = divergence_vol_thresh

    def detect(self, features: dict) -> int:
        mom_short = features.get("momentum_10", 0.0)
        mom_mid = features.get("momentum_20", 0.0)
        mom_long = features.get("momentum_60", 0.0)
        vol_level = features.get("vol_level", 1.0)

        # 三时间框架动量共振 → 强趋势
        if mom_short > 0 and mom_mid > 0 and mom_long > 0:
            new_mode = Regime.BULL_TREND
        elif mom_short < 0 and mom_mid < 0 and mom_long < 0:
            new_mode = Regime.BEAR_TREND
        # 动量分歧 + 高波动 → 高波动震荡
        elif vol_level > self.divergence_vol_thresh or (mom_short * mom_long < 0):
            new_mode = Regime.HIGH_VOL_OSC
        elif vol_level < self.vol_low_thresh:
            new_mode = Regime.LOW_VOL_CONV
        else:
            new_mode = Regime.NORMAL_OSC

        return self._apply_debounce(new_mode)


# ==================================================================
# VolatilityRegimeDetector — 基于波动率趋势 + ADX 联合判断
# ==================================================================


class VolatilityRegimeDetector(RegimeDetector):
    """基于波动率水平及其趋势的 Regime 检测器。

    高波动 + 强 ADX → 波动驱动的趋势，
    高波动 + 弱 ADX → 高波动震荡，
    低波动 → 低波动收敛。

    特征需求: vol_level, trend_strength, trend_up
    """

    def __init__(
        self,
        high_vol_thresh: float = 1.5,
        med_vol_thresh: float = 1.3,
        low_vol_thresh: float = 0.7,
        trend_strength_threshold: float = 25.0,
        confirm_bars: int = 3,
        cooldown_bars: int = 2,
    ):
        super().__init__(confirm_bars=confirm_bars, cooldown_bars=cooldown_bars)
        self.high_vol_thresh = high_vol_thresh
        self.med_vol_thresh = med_vol_thresh
        self.low_vol_thresh = low_vol_thresh
        self.trend_strength_threshold = trend_strength_threshold

    def detect(self, features: dict) -> int:
        vol_level = features.get("vol_level", 1.0)
        trend_strength = features.get("trend_strength", 0.0)
        trend_up = bool(features.get("trend_up", True))

        # 高波动 + 强 ADX → 波动驱动的趋势
        if vol_level > self.high_vol_thresh and trend_strength > self.trend_strength_threshold:
            new_mode = Regime.BULL_TREND if trend_up else Regime.BEAR_TREND
        # 高波动 + 无趋势 → 高波动震荡
        elif vol_level > self.med_vol_thresh:
            new_mode = Regime.HIGH_VOL_OSC
        # 低波动 → 低波动收敛
        elif vol_level < self.low_vol_thresh:
            new_mode = Regime.LOW_VOL_CONV
        # 中等波动 + 强 ADX → 趋势
        elif trend_strength > self.trend_strength_threshold:
            new_mode = Regime.BULL_TREND if trend_up else Regime.BEAR_TREND
        else:
            new_mode = Regime.NORMAL_OSC

        return self._apply_debounce(new_mode)


# ==================================================================
# CompositeRegimeDetector — 多检测器投票组合
# ==================================================================


class CompositeRegimeDetector(RegimeDetector):
    """多检测器投票组合检测器。

    运行所有子检测器，取加权投票结果。
    当最高票数的 Regime 达到 min_agreement 阈值时才切换，
    否则保持当前 mode，有效抑制噪声。

    用法::

        det = CompositeRegimeDetector([
            DefaultRegimeDetector(),
            BollingerSqueezeRegimeDetector(),
            MomentumRegimeDetector(),
        ], min_agreement=2)
    """

    def __init__(
        self,
        detectors: list,
        weights: list[float] | None = None,
        min_agreement: int = 2,
    ):
        super().__init__(confirm_bars=1, cooldown_bars=0)
        if not detectors:
            raise ValueError("detectors must not be empty")
        self.detectors = detectors
        self.weights = weights or [1.0] * len(detectors)
        if len(self.weights) != len(self.detectors):
            raise ValueError("weights length must match detectors length")
        self.min_agreement = min_agreement

    def reset(self) -> None:
        super().reset()
        for d in self.detectors:
            d.reset()

    def detect(self, features: dict) -> int:
        votes: dict[int, float] = {}
        for det, w in zip(self.detectors, self.weights):
            regime = det.detect(features)
            votes[regime] = votes.get(regime, 0.0) + w

        # 找出最高票数
        best_regime = max(votes, key=lambda k: votes[k])
        best_score = votes[best_regime]

        # 统计达到 min_agreement 的候选
        if best_score >= self.min_agreement and best_regime != self.mode:
            self.mode = best_regime

        return self.mode

