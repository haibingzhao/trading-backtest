# trading-backtest

[中文文档](README_zh.md)

A multi-market backtesting engine supporting stocks, futures, forex, cryptocurrencies, and options across 7 market types.

## Features

- **Multi-market**: A-share, US/HK equity, China futures, global futures, forex, crypto, options
- **15+ data sources**: tushare, yfinance, akshare, ccxt, futu, and more — with automatic fallback
- **Flexible strategy interface**: Implement `SignalEngine.generate()` to plug in any custom strategy
- **Built-in strategy framework**: Regime detection, grid/trend sub-strategies, risk management
- **Portfolio optimizers**: Mean-variance, risk parity, max diversification, equal volatility
- **Statistical validation**: Monte Carlo simulation, Walk-Forward analysis, Time Series CV
- **Auto report generation**: HTML report with equity curve, trade log, and performance metrics

## Installation

```bash
# Core install
pip install trading-backtest

# With specific data source dependencies
pip install "trading-backtest[a-share]"       # China A-share (tushare, akshare, baostock)
pip install "trading-backtest[global-equity]" # US/HK equity (yfinance)
pip install "trading-backtest[crypto]"        # Crypto (ccxt)
pip install "trading-backtest[futu]"          # Futu API
pip install "trading-backtest[all]"           # All dependencies
```

## Quick Start

### 1. Create a run directory

```bash
mkdir -p runs/my_strategy/code
```

### 2. Write config (`runs/my_strategy/config.json`)

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

### 3. Implement signal engine (`runs/my_strategy/code/signal_engine.py`)

```python
import pandas as pd

class SignalEngine:
    def __init__(self, ema_fast=12, ema_slow=26, **kwargs):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def generate(self, data_map: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        """
        Input:  data_map = {"AAPL": DataFrame, "MSFT": DataFrame}
                Each DataFrame has OHLCV columns with DatetimeIndex.
        Output: signal_map = {"AAPL": Series, "MSFT": Series}
                Signal values: +1 (long), -1 (short), 0 (flat)
        """
        signals = {}
        for code, df in data_map.items():
            fast = df["close"].ewm(span=self.ema_fast).mean()
            slow = df["close"].ewm(span=self.ema_slow).mean()
            signal = (fast > slow).astype(int) * 2 - 1
            signals[code] = signal
        return signals
```

### 4. Run backtest

```bash
python -m backtest.runner runs/my_strategy
```

### 5. View results

Output in `runs/my_strategy/artifacts/`:
- `report.html` — Visual HTML report
- `equity.csv` — Equity curve
- `trades.csv` — Trade log
- `metrics.csv` — Performance metrics

## Using the Built-in Strategy Framework

For a batteries-included approach, use the built-in strategy with one line:

```python
# runs/my_strategy/code/signal_engine.py
from backtest.strategy import DefaultSignalEngine as SignalEngine
```

Configure via `signal_params` in `config.json`:

```json
{
  "signal_params": {
    "allow_short": true,
    "risk_per_trade": 0.05,
    "trend_strength_threshold": 22,
    "max_drawdown_halt": 0.10
  }
}
```

## Configuration Reference

| Field | Required | Description |
|-------|----------|-------------|
| `codes` | ✅ | List of instrument codes |
| `start_date` | ✅ | Start date (YYYY-MM-DD) |
| `end_date` | ✅ | End date (YYYY-MM-DD) |
| `source` | ❌ | Data source, default `auto` |
| `interval` | ❌ | Bar interval: `1m/5m/15m/30m/1H/4H/1D` |
| `engine` | ❌ | Engine type: `daily/options` |
| `initial_cash` | ❌ | Starting capital (default: 1000000) |
| `signal_params` | ❌ | Parameters passed to SignalEngine |
| `benchmark` | ❌ | Benchmark symbol (e.g. `SPY`) |

### Symbol Format

| Market | Examples |
|--------|----------|
| A-share | `000001.SZ`, `600519.SH` |
| US equity | `AAPL`, `MSFT` |
| HK equity | `00700.HK`, `09988.HK` |
| Futures | `IF2312`, `CU2401` |
| Crypto | `BTC-USDT`, `ETH-USDT` |
| Forex | `USD/CNY`, `EUR/USD` |

## Engine Architecture

```
BaseEngine (base.py)
├── ChinaAEngine          # A-share (T+1, price limits)
├── GlobalEquityEngine    # US/HK equity (T+0)
├── CryptoEngine          # Crypto (24/7)
├── ChinaFuturesEngine    # China futures
├── GlobalFuturesEngine   # Global futures
├── ForexEngine           # Forex
├── CompositeEngine       # Cross-market portfolio
└── OptionsPortfolioEngine # Options portfolio
```

## Environment Variables

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

Key variables:
- `TUSHARE_TOKEN` — Tushare API token (A-share data)
- `TIINGO_API_KEY` — Tiingo API key (US equity)
- `TRADING_BACKTEST_DATA_CACHE=1` — Enable local Parquet cache

See `.env.example` for the full list.

## Development

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pip install pytest ruff mypy

# Test
pytest tests/ -v

# Lint
ruff check backtest/

# Type check
mypy backtest/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Project Structure

```
trading-backtest/
├── backtest/
│   ├── engines/          # Market-specific engines
│   ├── loaders/          # Data source adapters
│   ├── optimizers/       # Portfolio optimizers
│   ├── strategy/         # Built-in strategy framework
│   ├── templates/        # Report & strategy templates
│   ├── runner.py         # CLI entry point
│   ├── metrics.py        # Performance metrics
│   └── models.py         # Data models
├── examples/             # Example strategies
├── tests/                # Unit tests
├── .github/workflows/    # CI/CD
└── pyproject.toml        # Project config
```

## Acknowledgements

This project is forked from [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading). We thank the original authors for their excellent work.

## License

[MIT](LICENSE)
