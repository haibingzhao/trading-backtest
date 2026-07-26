"""布林带均值回归策略模板。

启用框架内置的 MeanReversionStrategy，
在震荡 Regime 下基于布林带 + RSI 超买超卖进行交易。

策略逻辑:
    - 启用均值回归子策略（mr_enabled=True）
    - 震荡 Regime 下:
      - 做多: 价格跌破布林带下轨（Z-Score <= -2.0）且 RSI < 30（超卖）
      - 做空: 价格涨破布林带上轨（Z-Score >= 2.0）且 RSI > 70（超买）
      - 出场: 价格回归均线附近（Z-Score 回到 ±0.5）
      - 止损: 价格继续偏离超过 Z-Score 3.0
    - 趋势 Regime 下: 自动切换到趋势跟踪子策略（SAR+DI 入场）

可自定义参数（通过 config.json 的 signal_params 传入）:
    - mr_zscore_entry: 入场 Z-Score 阈值（默认 2.0）
    - mr_zscore_exit: 出场 Z-Score 阈值（默认 0.5）
    - mr_zscore_stop: 止损 Z-Score（默认 3.0）
    - mr_rsi_oversold: RSI 超卖阈值（默认 30）
    - mr_rsi_overbought: RSI 超买阈值（默认 70）
    - mr_max_position_ratio: 均值回归最大仓位（默认 0.5）
    - allow_short: 是否允许做空（默认 False）

使用步骤:
    1. 复制本文件到 runs/<your_run>/code/signal_engine.py
    2. 修改 config.json 中的 signal_params 调整参数
    3. 运行: ./run_backtest.sh runs/<your_run>
"""
from __future__ import annotations

from backtest.strategy import StrategyBase


class SignalEngine(StrategyBase):
    """布林带均值回归 + 框架标准 Regime / 风控。

    自定义点:
    - 启用均值回归子策略（mr_enabled=True）
    - 在震荡 Regime 下自动启用布林带+RSI 均值回归
    - 在趋势 Regime 下自动切换到趋势跟踪

    框架自动处理:
    - 指标计算（IndicatorPipeline: ATR / ADX / SAR / EMA / RSI / 布林带等）
    - Regime 检测（DefaultRegimeDetector: ADX + DI + 波动率）
    - 趋势 Regime 下: SAR+DI 入场 + 金字塔加仓 + 移动止损
    - 回撤熔断、连续止损暂停、日亏损限制
    """

    def __init__(self, **kwargs):
        # 默认启用均值回归
        kwargs.setdefault("mr_enabled", True)
        super().__init__(**kwargs)
