# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-25

### Added

- Multi-market backtest engines: A-share, US/HK equity, crypto, China futures, global futures, forex, options portfolio, and composite cross-market engine
- 15+ data loaders with automatic fallback: tushare, akshare, baostock, mootdx, yfinance, yahoo, stooq, sina, eastmoney, tencent, ccxt, okx, futu, tiingo, finnhub, alphavantage, FMP, SEC EDGAR, local file loader
- Built-in strategy framework (`backtest.strategy`): Regime detection (6 states, 5 detectors), grid/trend sub-strategies, risk management
- Portfolio optimizers: mean-variance, equal volatility, max diversification, risk parity, momentum rank
- Statistical validation: Monte Carlo simulation, Bootstrap CI, Walk-Forward analysis
- Walk-Forward Optimization (WFO) and Time Series Cross-Validation for parameter robustness
- HTML report generation with embedded charts and Chinese localization
- Run card system for reproducible research
- Local Parquet data cache (opt-in via `TRADING_BACKTEST_DATA_CACHE=1`)
- Strategy templates: EMA crossover, momentum regime, mean reversion (Bollinger Band)
- Comprehensive metrics: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, etc.
- Benchmark comparison and rolling correlation analysis
- Parameter sensitivity analysis and grid search
- Lot size analysis for HK/A-share markets
- Fill price stress testing
- Distribution analysis and periodic metrics breakdown
