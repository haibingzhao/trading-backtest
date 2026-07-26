# Contributing to trading-backtest

Thank you for your interest in contributing to trading-backtest! This document provides guidelines and information for contributors.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/haibingzhao/trading-backtest.git
cd trading-backtest

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode with all optional dependencies
pip install -e ".[all]"

# Install dev tools
pip install pytest pytest-cov ruff mypy
```

## Project Structure

```
backtest/
├── engines/       # Market-specific backtest engines (inherit BaseEngine)
├── loaders/       # Data source adapters (implement DataLoaderProtocol)
├── optimizers/    # Portfolio weight optimizers
├── strategy/      # Built-in strategy framework (Regime, Grid, Trend)
├── templates/     # HTML report template + strategy templates
├── runner.py      # CLI entry point
├── metrics.py     # Performance metrics calculation
├── models.py      # Data models (frozen dataclass)
└── validation.py  # Monte Carlo / Walk-Forward validation
```

## Coding Conventions

- **Python version**: >= 3.10
- **Data models**: Use `frozen dataclass` for immutable data structures
- **Type hints**: Required for all public APIs
- **Docstrings**: Google-style for all public classes/functions
- **Line length**: 100 characters max
- **Imports**: Standard library → third-party → local (isort compatible)

## Adding a New Data Loader

1. Create `backtest/loaders/<name>_loader.py`
2. Implement the `DataLoaderProtocol` interface:

```python
from backtest.loaders.base import cached_loader_fetch
from backtest.loaders.registry import register

@register
class MyDataLoader:
    name = "my_source"
    markets = {"us_equity"}  # Supported market types

    def is_available(self) -> bool:
        """Return True if this loader can be used (e.g., API key configured)."""
        return True

    def fetch(self, codes, start_date, end_date, fields=None, interval="1D"):
        """Return {code: DataFrame} with OHLCV columns."""
        ...
```

3. Add the source to the appropriate fallback chain in `loaders/registry.py`

## Adding a New Engine

1. Create `backtest/engines/<name>.py`
2. Inherit from `BaseEngine` and implement the bar-by-bar execution loop
3. Register in `engines/__init__.py`

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backtest --cov-report=html

# Run specific test file
pytest tests/test_metrics.py -v
```

## Code Quality

```bash
# Lint
ruff check backtest/

# Type check
mypy backtest/

# Format
ruff format backtest/
```

## Pull Request Process

1. Fork the repository and create a feature branch (`git checkout -b feature/my-feature`)
2. Make your changes following the coding conventions above
3. Add tests for new functionality
4. Ensure all tests pass and linting is clean
5. Update documentation if needed
6. Submit a pull request with a clear description of changes

## Reporting Issues

- Use the GitHub issue tracker
- Include: Python version, OS, minimal reproduction steps, expected vs actual behavior
- For data source issues, specify which source and whether API keys are configured

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
