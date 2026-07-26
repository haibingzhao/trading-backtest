"""动量 Regime 策略模板。

使用 MomentumRegimeDetector 替代默认 Regime 检测器，
基于多时间框架动量一致性判断市场状态。

策略逻辑:
    - Regime 检测: MomentumRegimeDetector（10/20/60 日动量共振）
      - 三时间框架动量同向 → 强趋势（BULL/BEAR_TREND）
      - 动量分歧 + 高波动 → 高波动震荡
      - 动量分歧 + 低波动 → 低波动收敛
    - 入场信号: 框架默认 SAR+DI（SARDirectionEntry）
    - 风控: 框架标准 RiskManager

可自定义参数（通过 config.json 的 signal_params 传入）:
    - confirm_bars: Regime 切换确认 bar 数（默认 3）
    - cooldown_bars: Regime 切换冷却 bar 数（默认 2）
    - vol_high_thresh / vol_low_thresh: 波动率阈值
    - risk_per_trade: 每笔交易风险比例
    - trailing_stop_atr_mult: 移动止损 ATR 倍数
    - trend_strength_threshold: 趋势强度阈值

使用步骤:
    1. 复制本文件到 runs/<your_run>/code/signal_engine.py
    2. 修改 config.json 中的 signal_params 调整参数
    3. 运行: ./run_backtest.sh runs/<your_run>
"""
from __future__ import annotations

from backtest.strategy import (
    MomentumRegimeDetector,
    StrategyBase,
)


class SignalEngine(StrategyBase):
    """动量 Regime + 框架标准子策略。

    自定义点:
    - Regime 检测: MomentumRegimeDetector（多时间框架动量共振）
    - 入场信号: 使用框架默认 SAR+DI
    - 风控: 使用框架标准 RiskManager

    框架自动处理:
    - 指标计算（IndicatorPipeline: ATR / ADX / SAR / EMA / RSI / 动量等）
    - 趋势 Regime 下: SAR+DI 入场 + 金字塔加仓 + 移动止损
    - 震荡 Regime 下: 网格低买高卖
    - 回撤熔断、连续止损暂停、日亏损限制
    """

    def create_regime_detector(self):
        """使用动量 Regime 检测器。

        MomentumRegimeDetector 基于三个时间框架的动量方向一致性:
        - momentum_10 > 0 且 momentum_20 > 0 且 momentum_60 > 0 → BULL_TREND
        - 三者均 < 0 → BEAR_TREND
        - 方向分歧 → 按波动率细分为震荡子状态

        Returns:
            MomentumRegimeDetector 实例。
        """
        return MomentumRegimeDetector(
            confirm_bars=3,            # Regime 切换确认 bar 数
            cooldown_bars=2,           # 切换后冷却 bar 数
            vol_high_thresh=1.3,       # 高波动阈值
            vol_low_thresh=0.7,        # 低波动阈值
            divergence_vol_thresh=1.3, # 动量分歧时的高波动阈值
        )
