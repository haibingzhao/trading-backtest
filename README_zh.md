# trading-backtest

多市场回测引擎，支持股票、期货、外汇、加密货币等多种资产类别。

## 特性

- **多市场支持**：A股、美股、港股、期货、外汇、加密货币
- **多数据源**：tushare、yfinance、akshare、ccxt 等 15+ 数据源，内置 fallback 机制
- **灵活策略接口**：实现 `SignalEngine` 即可接入自定义策略
- **组合优化**：内置均值方差、风险平价、最大分散化等优化器
- **统计验证**：Monte Carlo 模拟 + Walk-Forward 分析
- **自动报告生成**：HTML 回测报告，包含绩效指标、交易记录、权益曲线

## 安装

```bash
# 基础安装
pip install -e .

# 安装特定数据源依赖
pip install -e ".[a-share]"       # A股（tushare, akshare, baostock）
pip install -e ".[global-equity]" # 美股/港股（yfinance）
pip install -e ".[crypto]"        # 加密货币（ccxt）
pip install -e ".[futu]"          # 富途 API
pip install -e ".[all]"           # 全部依赖
```

## 快速开始

### 1. 创建运行目录

```bash
mkdir -p runs/my_strategy/code
```

### 2. 编写配置文件 `runs/my_strategy/config.json`

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

### 3. 实现信号引擎 `runs/my_strategy/code/signal_engine.py`

```python
import pandas as pd

class SignalEngine:
    """自定义信号引擎，必须实现 generate() 方法"""
    
    def __init__(self, ema_fast=12, ema_slow=26, **kwargs):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
    
    def generate(self, data_map: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        """
        输入: data_map = {"AAPL": DataFrame, "MSFT": DataFrame}
              每个 DataFrame 包含 OHLCV 列，index 为日期
        输出: signal_map = {"AAPL": Series, "MSFT": Series}
              信号值: +1 (做多), -1 (做空), 0 (空仓)
        """
        signals = {}
        for code, df in data_map.items():
            fast_ema = df["close"].ewm(span=self.ema_fast).mean()
            slow_ema = df["close"].ewm(span=self.ema_slow).mean()
            signal = (fast_ema > slow_ema).astype(int) * 2 - 1  # +1/-1
            signals[code] = signal
        return signals
```

### 4. 运行回测

```bash
python -m backtest.runner runs/my_strategy
```

### 5. 查看结果

回测完成后，结果输出到 `runs/my_strategy/artifacts/`：

- `report.html` — 可视化 HTML 报告
- `trades.json` — 交易记录
- `equity_curve.json` — 权益曲线
- `metrics.json` — 绩效指标

## 配置规范

### config.json 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `codes` | ✅ | 标的代码列表 |
| `start_date` | ✅ | 开始日期 (YYYY-MM-DD) |
| `end_date` | ✅ | 结束日期 (YYYY-MM-DD) |
| `source` | ❌ | 数据源，默认 `auto` |
| `interval` | ❌ | K线周期：`1m/5m/15m/30m/1H/4H/1D`，默认 `1D` |
| `engine` | ❌ | 引擎类型：`daily/options`，默认 `daily` |
| `initial_cash` | ❌ | 初始资金，默认 1000000 |
| `signal_params` | ❌ | 传递给 SignalEngine 的参数 |
| `benchmark` | ❌ | 基准标的（如 `SPY`） |
| `validation` | ❌ | 统计验证配置 |

### 数据源选择

| source | 适用市场 | 说明 |
|--------|----------|------|
| `auto` | 全市场 | 根据代码自动路由（推荐） |
| `tushare` | A股/期货 | 需设置 TUSHARE_TOKEN 环境变量 |
| `yfinance` | 美股/港股 | 免费，但有限流 |
| `akshare` | A股/宏观 | 免费，数据丰富 |
| `ccxt` | 加密货币 | 支持 100+ 交易所 |
| `okx` | 加密货币 | OKX 专用 |
| `futu` | A股/港股 | 需启动 OpenD 网关，见下方说明 |

### 富途数据源配置

使用 `source: "futu"` 获取行情数据前，需要确保 **FutuOpenD 网关程序** 已启动。

**检查 OpenD 是否运行：**

```bash
# 检查进程是否存在
pgrep -f FutuOpenD || echo "OpenD 未运行"
```

**启动 OpenD（macOS）：**

```bash
# 如果未运行，启动 FutuOpenD
~/software/futu10/cli/FutuOpenD.app/Contents/MacOS/FutuOpenD &
```

> **注意**：启动前需确保已登录富途牛牛账号，且 OpenD 配置正确（API 端口、登录信息等）。首次使用请在 FutuOpenD GUI 中完成配置。

### 标的代码格式

| 市场 | 格式示例 |
|------|----------|
| A股 | `000001.SZ`, `600519.SH` |
| 美股 | `AAPL`, `MSFT` |
| 港股 | `00700.HK` |
| 期货 | `IF2312`, `CU2401` |
| 加密货币 | `BTC-USDT`, `ETH-USDT` |
| 外汇 | `USD/CNY`, `EUR/USD` |

## 信号引擎规范

### SignalEngine 接口

```python
class SignalEngine:
    def __init__(self, **kwargs):
        """所有参数必须有默认值，或通过 signal_params 传入"""
        pass
    
    def generate(self, data_map: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data_map: {code: DataFrame}，DataFrame 包含 OHLCV 列
                      - open, high, low, close, volume
                      - index 为 DatetimeIndex
        
        Returns:
            signal_map: {code: Series}
                        - 信号值: +1 (做多), -1 (做空), 0 (空仓)
                        - index 为 DatetimeIndex
        """
        pass
```

### 信号语义

- **+1**：全仓做多
- **-1**：全仓做空（A股不支持）
- **0**：空仓
- **0.5**：半仓做多（支持连续仓位）

### 约束

- `__init__()` 所有参数必须有默认值
- 不允许有顶层可执行语句（安全沙箱限制）
- `generate()` 中只能使用传入的 `data_map`，不能访问外部数据

## 引擎架构

```
BaseEngine (base.py)
├── ChinaAEngine      # A股（T+1, 涨跌停）
├── GlobalEquityEngine # 美股/港股（T+0, 无限制）
├── CryptoEngine       # 加密货币（24/7）
├── ChinaFuturesEngine # 国内期货
├── GlobalFuturesEngine # 国际期货
├── ForexEngine        # 外汇
├── CompositeEngine    # 跨市场组合
└── OptionsPortfolioEngine # 期权组合
```

## 数据加载器

所有数据加载器实现 `DataLoaderProtocol`：

```python
class DataLoaderProtocol(Protocol):
    def fetch(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
        interval: str = "1D",
    ) -> dict[str, pd.DataFrame]:
        """返回 {code: DataFrame} 映射"""
        ...
```

### 自定义数据加载器

```python
from backtest.loaders.base import cached_loader_fetch
from backtest.loaders.registry import register

@register
class MyDataLoader:
    name = "my_source"
    
    def fetch(self, codes, start_date, end_date, fields=None, interval="1D"):
        # 实现数据获取逻辑
        pass
```

## 绩效指标

回测自动计算以下指标：

| 指标 | 说明 |
|------|------|
| 总收益率 | 策略期间总回报 |
| 年化收益率 | 年化后的收益率 |
| 最大回撤 | 峰值到谷底的最大跌幅 |
| 夏普比率 | 风险调整后收益 |
| 索提诺比率 | 只考虑下行风险的调整收益 |
| 卡尔玛比率 | 年化收益/最大回撤 |
| 胜率 | 盈利交易占比 |
| 盈亏比 | 平均盈利/平均亏损 |

## 统计验证

在 `config.json` 中启用：

```json
{
  "validation": {
    "monte_carlo": {
      "n_simulations": 1000,
      "confidence": 0.95
    },
    "walk_forward": {
      "n_splits": 5
    }
  }
}
```

- **Monte Carlo**：随机打乱交易顺序，评估策略稳健性
- **Walk-Forward**：滚动窗口回测，验证样本外表现

## 目录结构

```
trading-backtest/
├── backtest/
│   ├── engines/          # 回测引擎
│   ├── loaders/          # 数据加载器
│   ├── optimizers/       # 组合优化器
│   ├── templates/        # 报告模板
│   ├── runner.py         # 入口脚本
│   ├── metrics.py        # 绩效计算
│   └── models.py         # 数据模型
├── runs/                 # 回测运行目录
│   └── my_strategy/
│       ├── config.json
│       ├── code/
│       │   └── signal_engine.py
│       └── artifacts/    # 输出结果
└── pyproject.toml
```

## 常见问题

### Q: 数据源限流怎么办？

A: 使用 `source: "auto"` 会自动 fallback 到备用数据源。也可以临时移除 `benchmark` 配置绕过限流。

### Q: A股如何支持做空信号？

A: A股引擎会自动将 `-1` 信号转换为 `0`（空仓），因为 A股不支持融券做空。

### Q: 如何回测分钟级数据？

A: 设置 `interval: "1m"` 或 `"5m"` 等，注意数据源是否支持分钟级数据。

### Q: 信号引擎参数如何传递？

A: 通过 `config.json` 的 `signal_params` 字段，会作为 `**kwargs` 传给 `SignalEngine.__init__()`。

## 致谢

本项目 fork 自 [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading)，感谢原作者的优秀工作。

## License

MIT
