"""策略框架 — 可复用的多模式交易策略引擎。

提供:
- StrategyBase: 编排器基类（模板方法，指标无关）
- DefaultSignalEngine: 开箱即用的默认实现
- Regime / RegimeDetector: 市场 Regime 检测协议
- DefaultRegimeDetector: 基于 ADX+DI+vol_level 的默认检测器
- MACrossoverRegimeDetector: 基于 EMA 交叉 + ADX 过滤
- BollingerSqueezeRegimeDetector: 基于布林带宽度收敛/扩张
- MomentumRegimeDetector: 基于多时间框架动量一致性
- VolatilityRegimeDetector: 基于波动率趋势 + ADX 联合
- CompositeRegimeDetector: 多检测器投票组合
- RegimeConfig / DEFAULT_REGIME_CONFIGS: Regime 策略参数调制
- RiskManager: 风控管理
- MarketRules: 市场规则感知（做空许可）
- GridStrategy / TrendLongStrategy / TrendShortStrategy: 子策略组件
- MeanReversionStrategy: 均值回归子策略（布林带 + RSI）
- BaseTrendStrategy: 趋势子策略抽象基类（方向参数化）
- EntrySignal / SARDirectionEntry / SpreadCrossEntry: 入场信号协议及内置实现
- SignalTrace: 信号可追溯性日志
- IndicatorCalculator / IndicatorPipeline: 插件式指标管道
- compute_indicators / build_regime_features / build_regime_features_fast: 向量化技术指标计算
- RegimeParams / GridParams / TrendParams / ShortParams / RiskParams / PortfolioParams / MeanReversionParams: 参数分组

用法::

    from backtest.strategy import DefaultSignalEngine as SignalEngine
"""

from backtest.strategy.base import SignalTrace, StrategyBase
from backtest.strategy.components import (
    BaseTrendStrategy,
    EntrySignal,
    GridStrategy,
    MeanReversionStrategy,
    SARDirectionEntry,
    SpreadCrossEntry,
    TrendLongStrategy,
    TrendShortStrategy,
)
from backtest.strategy.defaults import DefaultSignalEngine
from backtest.strategy.indicators import (
    ATRCalculator,
    ADXCalculator,
    BollingerBandCalculator,
    EMACalculator,
    IndicatorCalculator,
    IndicatorParams,
    IndicatorPipeline,
    MAHighLowCalculator,
    MomentumCalculator,
    RSICalculator,
    SARCalculator,
    VolLevelCalculator,
    VolumeCalculator,
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
    BollingerSqueezeRegimeDetector,
    CompositeRegimeDetector,
    DefaultRegimeDetector,
    MACrossoverRegimeDetector,
    MomentumRegimeDetector,
    Regime,
    RegimeConfig,
    RegimeDetector,
    VolatilityRegimeDetector,
)
from backtest.strategy.risk import RiskManager

__all__ = [
    "StrategyBase",
    "DefaultSignalEngine",
    "SignalTrace",
    "Regime",
    "RegimeDetector",
    "DefaultRegimeDetector",
    "MACrossoverRegimeDetector",
    "BollingerSqueezeRegimeDetector",
    "MomentumRegimeDetector",
    "VolatilityRegimeDetector",
    "CompositeRegimeDetector",
    "RegimeConfig",
    "DEFAULT_REGIME_CONFIGS",
    "RiskManager",
    "MarketRules",
    "GridStrategy",
    "BaseTrendStrategy",
    "EntrySignal",
    "SARDirectionEntry",
    "SpreadCrossEntry",
    "MeanReversionStrategy",
    "TrendLongStrategy",
    "TrendShortStrategy",
    "IndicatorCalculator",
    "IndicatorPipeline",
    "IndicatorParams",
    "RegimeParams",
    "GridParams",
    "MeanReversionParams",
    "TrendParams",
    "ShortParams",
    "RiskParams",
    "PortfolioParams",
    "compute_indicators",
    "default_pipeline",
    "build_regime_features",
    "build_regime_features_fast",
    "ATRCalculator",
    "ADXCalculator",
    "SARCalculator",
    "EMACalculator",
    "RSICalculator",
    "MAHighLowCalculator",
    "VolumeCalculator",
    "VolLevelCalculator",
    "BollingerBandCalculator",
    "MomentumCalculator",
]
