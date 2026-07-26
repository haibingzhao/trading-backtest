# trading-backtest

[English](README.md)

多市场回测引擎，支持股票、期货、外汇、加密货币、期权等 7 种市场类型。

## 特性

- **多市场支持**：A股、美股/港股、国内期货、国际期货、外汇、加密货币、期权
- **15+ 数据源**：tushare、yfinance、akshare、ccxt、futu 等，内置自动 fallback
- **灵活策略接口**：实现 `SignalEngine.generate()` 即可接入任意自定义策略
- **内置策略框架**：Regime 检测、网格/趋势子策略、风控管理
- **组合优化器**：均值方差、风险平价、最大分散化、等波动率
- **统计验证**：Monte Carlo 模拟、Walk-Forward 分析、时间序列交叉验证
- **自动报告生成**：HTML 回测报告，含权益曲线、交易记录、绩效指标

## 安装

```bash
# 核心安装
pip install trading-backtest

# 安装特定数据源依赖
pip install "trading-backtest[a-share]"       # A股（tushare, akshare, baostock）
pip install "trading-backtest[global-equity]" # 美股/港股（yfinance）
pip install "trading-backtest[crypto]"        # 加密货币（ccxt）
pip install "trading-backtest[futu]"          # 富途 API
pip install "trading-backtest[all]"           # 全部依赖
```

## 快速开始

### 1. 安装

```bash
pip install "trading-backtest[global-equity]"  # 美股/港股，使用 yfinance
```

### 2. 创建策略目录

```bash
mkdir -p runs/my_strategy/code
```

### 3. 编写配置（`runs/my_strategy/config.json`）

```json
{
  "codes": ["AAPL", "MSFT"],
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "source": "auto",
  "interval": "1D",
  "engine": "daily",
  "initial_cash": 1000000,
  "signal_params": {
    "ema_fast": 12,
    "ema_slow": 26
  }
}
```

### 4. 实现信号引擎（`runs/my_strategy/code/signal_engine.py`）

```python
import pandas as pd

class SignalEngine:
    def __init__(self, ema_fast=12, ema_slow=26, **kwargs):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def generate(self, data_map: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        """
        输入: data_map = {"AAPL": DataFrame, "MSFT": DataFrame}
              每个 DataFrame 包含 OHLCV 列，index 为 DatetimeIndex
        输出: signal_map = {"AAPL": Series, "MSFT": Series}
              信号值: +1 (做多), -1 (做空), 0 (空仓)
        """
        signals = {}
        for code, df in data_map.items():
            fast = df["close"].ewm(span=self.ema_fast).mean()
            slow = df["close"].ewm(span=self.ema_slow).mean()
            signal = (fast > slow).astype(int) * 2 - 1
            signals[code] = signal
        return signals
```

或使用**内置策略**（一行代码，无需自定义）：

```python
# runs/my_strategy/code/signal_engine.py
from backtest.strategy import DefaultSignalEngine as SignalEngine
```

### 5. 运行回测

```bash
python -m backtest.runner runs/my_strategy
```

### 6. 查看结果

输出在 `runs/my_strategy/artifacts/` 目录下：
- `report.html` — 可视化 HTML 报告（浏览器打开）
- `equity.csv` — 净值曲线
- `trades.csv` — 交易记录
- `metrics.csv` — 绩效指标

## 使用示例

### A股

```json
{
  "codes": ["600519.SH"],
  "start_date": "2023-01-01",
  "end_date": "2025-01-01",
  "source": "auto",
  "commission": 0.005,
  "signal_params": {"ema_fast": 8, "ema_slow": 21}
}
```

> A股需安装 `pip install "trading-backtest[a-share]"`。在 `.env` 中设置 `TUSHARE_TOKEN` 可获得最佳数据质量，也可使用免费数据源（akshare/baostock）无需 API Key。

### 港股

```json
{
  "codes": ["09988.HK"],
  "start_date": "2024-01-01",
  "end_date": "2025-07-01",
  "source": "auto",
  "hk_commission": 0.005,
  "hk_stamp_tax": 0.001,
  "signal_params": {"allow_short": true, "risk_per_trade": 0.05}
}
```

### 加密货币（24/7）

```json
{
  "codes": ["BTC-USDT", "ETH-USDT"],
  "start_date": "2024-01-01",
  "end_date": "2025-01-01",
  "source": "okx",
  "interval": "4H",
  "initial_cash": 100000,
  "leverage": 2.0
}
```

> 加密货币需安装 `pip install "trading-backtest[crypto]"` 以支持多交易所（ccxt）。内置 `okx` 数据源无需额外依赖。

### Walk-Forward 参数优化

在 `config.json` 中添加以下配置验证参数鲁棒性：

```json
{
  "walk_forward_opt": {
    "param_grid": {
      "ema_fast": [5, 8, 12],
      "ema_slow": [13, 21, 30],
      "risk_per_trade": [0.03, 0.05]
    },
    "n_splits": 3,
    "objective": "sharpe"
  }
}
```

```bash
python -m backtest.walk_forward_opt runs/my_strategy
# 输出: runs/my_strategy/artifacts/walk_forward_opt.json
```

### 统计验证

```json
{
  "validation": {
    "monte_carlo": {"n_simulations": 1000, "confidence": 0.95},
    "walk_forward": {"n_splits": 5}
  }
}
```

### 更多示例

参见 [`examples/`](examples/) 目录：
- `examples/ema_crossover/` — 自定义 EMA 交叉策略（手写 SignalEngine）
- `examples/builtin_strategy/` — 一行代码使用内置策略框架

## 配置说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `codes` | ✅ | 标的代码列表 |
| `start_date` | ✅ | 开始日期 (YYYY-MM-DD) |
| `end_date` | ✅ | 结束日期 (YYYY-MM-DD) |
| `source` | ❌ | 数据源，默认 `auto` |
| `interval` | ❌ | K线周期：`1m/5m/15m/30m/1H/4H/1D` |
| `engine` | ❌ | 引擎类型：`daily/options` |
| `initial_cash` | ❌ | 初始资金（默认 1000000） |
| `signal_params` | ❌ | 传递给 SignalEngine 的参数 |
| `benchmark` | ❌ | 基准标的（如 `SPY`） |

### 标的代码格式

| 市场 | 格式示例 |
|------|----------|
| A股 | `000001.SZ`, `600519.SH` |
| 美股 | `AAPL`, `MSFT` |
| 港股 | `00700.HK`, `09988.HK` |
| 期货 | `IF2312`, `CU2401` |
| 加密货币 | `BTC-USDT`, `ETH-USDT` |
| 外汇 | `USD/CNY`, `EUR/USD` |

## 引擎架构

```
BaseEngine (base.py)
├── ChinaAEngine          # A股（T+1, 涨跌停）
├── GlobalEquityEngine    # 美股/港股（T+0）
├── CryptoEngine          # 加密货币（24/7）
├── ChinaFuturesEngine    # 国内期货
├── GlobalFuturesEngine   # 国际期货
├── ForexEngine           # 外汇
├── CompositeEngine       # 跨市场组合
└── OptionsPortfolioEngine # 期权组合
```

## 环境变量

复制 `.env.example` 为 `.env` 并配置 API Key：

```bash
cp .env.example .env
```

主要变量：
- `TUSHARE_TOKEN` — Tushare API token（A股数据）
- `TIINGO_API_KEY` — Tiingo API key（美股）
- `TRADING_BACKTEST_DATA_CACHE=1` — 启用本地 Parquet 缓存

完整列表见 `.env.example`。

## 开发

```bash
# 环境搭建
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pip install pytest ruff mypy

# 测试
pytest tests/ -v

# 代码检查
ruff check backtest/

# 类型检查
mypy backtest/
```

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 项目结构

```
trading-backtest/
├── backtest/
│   ├── engines/          # 市场引擎
│   ├── loaders/          # 数据源适配器
│   ├── optimizers/       # 组合优化器
│   ├── strategy/         # 内置策略框架
│   ├── templates/        # 报告与策略模板
│   ├── runner.py         # CLI 入口
│   ├── metrics.py        # 绩效指标计算
│   └── models.py         # 数据模型
├── examples/             # 示例策略
├── tests/                # 单元测试
├── .github/workflows/    # CI/CD
└── pyproject.toml        # 项目配置
```

## 致谢

本项目 fork 自 [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading)，感谢原作者的优秀工作。

## License

[MIT](LICENSE)
