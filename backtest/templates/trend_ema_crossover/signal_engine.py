"""EMA 交叉趋势策略模板。

继承 StrategyBase，使用 EMA 快慢线交叉（SpreadCrossEntry）替代默认的 SAR+DI 入场。
适用于趋势跟踪型策略，通过 EMA 金叉/死叉触发入场。

策略逻辑:
    - 入场信号: EMA 快线上穿慢线（金叉）做多，下穿（死叉）做空
    - Regime 检测: 框架默认 DefaultRegimeDetector（ADX + DI + 波动率）
    - 风控: 框架标准 RiskManager（回撤熔断 + 连续止损暂停 + 日亏损限制）
    - 子策略: 趋势 Regime 下使用 EMA 交叉入场 + 金字塔加仓 + 移动止损
             震荡 Regime 下使用网格低买高卖

可自定义参数（通过 config.json 的 signal_params 传入）:
    - ema_fast / ema_slow: EMA 快慢线周期（默认 12/26）
    - spread_threshold: EMA 间距阈值，过滤噪声交叉（默认 0.003）
    - risk_per_trade: 每笔交易风险比例（默认 0.03）
    - trailing_stop_atr_mult: 移动止损 ATR 倍数（默认 2.0）
    - max_drawdown_halt: 回撤熔断阈值（默认 0.08）
    - trend_strength_threshold: 趋势强度阈值 / ADX（默认 25）

使用步骤:
    1. 复制本文件到 runs/<your_run>/code/signal_engine.py
    2. 修改 config.json 中的 signal_params 调整参数
    3. 运行: ./run_backtest.sh runs/<your_run>
"""
from __future__ import annotations

from backtest.strategy import StrategyBase, SpreadCrossEntry


class SignalEngine(StrategyBase):
    """EMA 交叉入场 + 框架标准 Regime / 风控。

    自定义点:
    - 入场信号: EMA 快慢线交叉（替代默认 SAR+DI）
    - Regime 检测: 使用框架默认 DefaultRegimeDetector
    - 风控: 使用框架标准 RiskManager

    框架自动处理:
    - 指标计算（IndicatorPipeline: ATR / ADX / SAR / EMA / RSI / 布林带 / 动量等）
    - Regime 切换时子策略分派（趋势→趋势跟踪，震荡→网格 / 均值回归）
    - 金字塔加仓、移动止损、反转平仓
    - 回撤熔断、连续止损暂停、日亏损限制
    """

    def create_entry_signal(self, direction: int):
        """使用 EMA 交叉作为入场信号。

        SpreadCrossEntry 检测 EMA 快慢线的交叉事件:
        - 做多: ema_fast 上穿 ema_slow（金叉），且间距 >= spread_threshold
        - 做空: ema_fast 下穿 ema_slow（死叉），且间距 >= spread_threshold

        Args:
            direction: +1 做多, -1 做空。

        Returns:
            SpreadCrossEntry 实例。
        """
        return SpreadCrossEntry(
            direction=direction,
            spread_threshold=0.003,  # EMA 间距阈值，可调整
            fast_key="ema_fast",     # 使用框架计算的 EMA 快线
            slow_key="ema_slow",     # 使用框架计算的 EMA 慢线
        )
