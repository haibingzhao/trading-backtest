# Development Guide — trading-backtest

## 环境配置

### 虚拟环境

项目使用 Python 虚拟环境管理依赖，位于 `trading-backtest/.venv/`。

**创建虚拟环境（首次设置）：**

```bash
python3 -m venv trading-backtest/.venv
```

**安装依赖：**

```bash
# 安装核心依赖 + 所有可选依赖（A股、美股、加密货币、富途、CLI）
trading-backtest/.venv/bin/pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e "trading-backtest[all]"

# 或仅安装核心依赖
trading-backtest/.venv/bin/pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e trading-backtest
```

> **注意**：如遇 SSL 证书错误，需添加 `--trusted-host pypi.org --trusted-host files.pythonhosted.org` 参数。

**激活虚拟环境（可选）：**

```bash
source trading-backtest/.venv/bin/activate
```

激活后直接使用 `python` 命令即可，无需指定完整路径。

### Python 解释器

未激活虚拟环境时，使用完整路径运行：

```bash
trading-backtest/.venv/bin/python <脚本>
```

## 运行回测

### 基本用法

```bash
# 未激活虚拟环境
trading-backtest/.venv/bin/python -m backtest.runner <run_dir>

# 已激活虚拟环境
python -m backtest.runner <run_dir>
```

`<run_dir>` 是包含 `config.json` 和 `code/signal_engine.py` 的回测运行目录。

### 运行目录结构

每个回测运行需要一个目录，结构如下：

```
runs/<策略名>/
├── config.json          # 回测配置（必填）
├── code/
│   └── signal_engine.py # 信号引擎策略代码（必填）
└── artifacts/           # 输出目录（自动生成）
    ├── equity.csv       # 净值曲线
    ├── metrics.csv      # 绩效指标
    ├── trades.csv       # 交易记录
    └── report.html      # HTML 报告（如启用）
```

### config.json 配置说明

```json
{
  "codes": ["BABA.US"],           // 标的代码列表（必填）
  "start_date": "2025-01-01",     // 开始日期 YYYY-MM-DD（必填）
  "end_date": "2025-12-31",       // 结束日期 YYYY-MM-DD（必填）
  "source": "auto",               // 数据源：auto/tushare/yfinance/akshare/okx/ccxt/futu
  "interval": "1D",               // K线周期：1m/5m/15m/30m/1H/4H/1D
  "engine": "daily",              // 引擎类型：daily/options
  "initial_cash": 1000000,        // 初始资金
  "signal_params": {},            // 传递给 SignalEngine 的参数
  "benchmark": "SPY",             // 基准标的（可选）
  "leverage": 1.0,                // 杠杆倍数
  "validation": {                 // 统计验证（可选）
    "monte_carlo": {"n_simulations": 1000, "confidence": 0.95},
    "walk_forward": {"n_splits": 5}
  }
}
```

### signal_engine.py 编写方式

回测引擎支持两种方式编写信号引擎：

#### 方式一：手动编写（从零实现）

在 `code/signal_engine.py` 中手动实现 `SignalEngine` 类，完全自定义逻辑：

```python
import pandas as pd

class SignalEngine:
    """信号引擎：生成交易信号。"""

    def __init__(self, **params):
        # 所有参数必须有默认值
        self.params = params

    def generate(self, data_map: dict) -> dict:
        """
        Args:
            data_map: {code: DataFrame}，DataFrame 包含 OHLCV 列
        Returns:
            {code: Series}，Series 值为 -1(卖)/0(持有)/1(买)
        """
        signals = {}
        for code, df in data_map.items():
            signals[code] = pd.Series(0, index=df.index)
        return signals
```

#### 方式二：使用内置 strategy 模块（推荐）

`backtest.strategy` 模块提供开箱即用的多模式策略框架，内置 Regime 检测、网格/趋势子策略、风控管理和市场规则感知。在 `code/signal_engine.py` 中一行代码即可启用：

```python
# runs/my_strategy/code/signal_engine.py
from backtest.strategy import DefaultSignalEngine as SignalEngine
```

如需自定义参数，通过 `config.json` 的 `signal_params` 字段传入（所有参数均有默认值）：

```json
{
  "codes": ["09988.HK"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "signal_params": {
    "allow_short": true,
    "risk_per_trade": 0.05,
    "trend_strength_threshold": 22,
    "max_drawdown_halt": 0.10
  }
}
```

**strategy 模块架构：**

```
strategy/
├── base.py          # StrategyBase 编排器基类（模板方法）
├── defaults.py      # DefaultSignalEngine 开箱即用实现
├── components.py    # 子策略：GridStrategy / TrendLongStrategy / TrendShortStrategy
├── regime.py        # Regime 检测：6种市场状态 + 5种检测器
├── risk.py          # RiskManager：回撤熔断 / 连续止损 / 日亏损
├── market_rules.py  # MarketRules：做空许可（A股禁做空）
├── indicators.py    # 向量化技术指标计算（ATR/ADX/SAR/EMA/RSI/布林带/动量）
└── params.py        # 参数分组 dataclass
```

**信号编排流程：** 指标计算 → Regime 检测 → 风控检查 → 子策略分派（网格/趋势做多/趋势做空）

**Regime 检测器（5种可选）：**
- `DefaultRegimeDetector` — 基于 ADX+DI+波动率（默认）
- `MACrossoverRegimeDetector` — EMA 交叉 + ADX 过滤
- `BollingerSqueezeRegimeDetector` — 布林带宽度收敛/扩张
- `MomentumRegimeDetector` — 多时间框架动量一致性
- `VolatilityRegimeDetector` — 波动率趋势 + ADX 联合
- `CompositeRegimeDetector` — 多检测器投票组合

**自定义扩展：** 继承 `StrategyBase` 并覆盖工厂方法或钩子方法：

```python
from backtest.strategy import StrategyBase, CompositeRegimeDetector, MomentumRegimeDetector

class SignalEngine(StrategyBase):
    """自定义策略：替换 Regime 检测器，使用布林带+动量投票。"""

    def create_regime_detector(self):
        return CompositeRegimeDetector([
            MomentumRegimeDetector(),
            BollingerSqueezeRegimeDetector(),
        ], min_agreement=2)

    def on_regime_change(self, old_regime, new_regime):
        # 自定义 Regime 切换回调
        pass
```

### 数据源说明

| 数据源 | 适用市场 | 可选依赖 |
|--------|---------|---------|
| `tushare` | A股、期货、基金 | `tushare` |
| `akshare` | A股、宏观、外汇 | `akshare` |
| `yfinance` | 美股、港股 | `yfinance` |
| `okx` | 加密货币 | — |
| `ccxt` | 加密货币（多交易所） | `ccxt` |
| `futu` | 港股、A股 | `futu-api` |
| `auto` | 自动识别路由 | 按标的自动选择 |

使用 `source="auto"` 时，系统根据标的代码格式自动选择数据源，并在主数据源失败时自动 fallback。

### 数据缓存

回测引擎内置了基于本地磁盘的 K 线数据缓存机制，**默认关闭**，需通过环境变量显式启用：

```bash
export VIBE_TRADING_DATA_CACHE=1   # 可选值：1 / true / yes / on
```

启用后，所有数据加载器（akshare、eastmoney、yfinance、tushare、okx、ccxt 等）在拉取 K 线数据时会自动走本地 Parquet 缓存，避免重复网络请求。

**缓存存储路径：**

```
~/.vibe-trading/cache/loaders/{source_name}/{sha256_hash}.parquet
```

**缓存 key 由以下参数组合做 SHA256 哈希生成：**
- `source`（数据源名称）
- `symbol`（股票代码）
- `timeframe`（K 线周期）
- `start_date` / `end_date`（日期范围）
- `fields`（附加字段）

**缓存命中条件：**
1. 环境变量 `VIBE_TRADING_DATA_CACHE` 已启用
2. 请求参数组合与某次已缓存的请求完全一致
3. `end_date` 严格早于当天（未结算的当日数据不会被缓存，避免盘中不完整 K 线污染缓存）

**注意事项：**
- 缓存读写失败不会导致回测中断，会自动 fallback 到网络请求
- 如需清除缓存，直接删除 `~/.vibe-trading/cache/loaders/` 目录即可
- 不同数据源的同标的同时间范围会产生不同缓存条目（source 参与 key 计算）

### 示例：运行完整回测

```bash
# 1. 进入项目目录
cd trading-backtest

# 2. 使用虚拟环境运行
.venv/bin/python -m backtest.runner runs/grid_trend_strategy
```

输出产物在 `runs/grid_trend_strategy/artifacts/` 目录下。

## 项目结构

```
trading-backtest/
├── .venv/                 # Python 虚拟环境
├── backtest/              # 核心代码包
│   ├── engines/           # 回测引擎（7个市场引擎 + 组合引擎）
│   ├── loaders/           # 数据加载器（6个数据源 + 自动fallback）
│   ├── strategy/          # 策略框架（Regime检测、网格/趋势子策略、风控）
│   ├── optimizers/        # 组合优化器
│   ├── templates/         # HTML 报告模板
│   ├── runner.py          # 主入口：配置解析、引擎路由
│   ├── metrics.py         # 绩效指标计算
│   ├── models.py          # 数据模型
│   ├── benchmark.py       # 基准对比
│   ├── correlation.py     # 相关性分析
│   ├── validation.py      # Monte Carlo / Walk-Forward 验证
│   └── run_card.py        # 可复现运行记录
├── runs/                  # 回测运行目录
├── pyproject.toml         # 项目配置与依赖
└── AGENTS.md              # 本文件
```

## 注意事项

- **必须使用 `.venv` 中的 Python**：不要使用系统 Python 运行回测
- **不要绕过 `runner.py`**：它是唯一入口，负责配置验证、引擎选择、产物收集
- **不要硬编码数据源**：使用 `loaders/registry.py` 的 fallback 链
- **运行目录路径必须合规**：`safe_run_dir()` 会验证路径安全性
- **信号引擎限制**：`SignalEngine.__init__()` 所有参数必须有默认值，不能有顶层可执行语句
- **strategy 模块扩展**：继承 `StrategyBase` 时通过覆盖工厂方法（`create_regime_detector()` 等）注入自定义组件，不要覆盖 `generate()` 方法
