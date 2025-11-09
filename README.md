# OptiBack

Options pricing & backtesting engine:
- Pricing: Black–Scholes, Binomial, Monte Carlo
- Greeks & implied vol
- Backtests: delta-hedge, mispricing with realistic costs

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
optiback version
pytest
```

## Overview
OptiBack is a Python toolkit for options research. It focuses on fast, reproducible pricing and
backtesting workflows with realistic frictions (transaction costs, slippage, discrete hedging).

Core goals:
- Accurate reference pricing models and Greeks
- Composable backtests for hedging and mispricing strategies
- Ergonomic CLI for quick experiments; Python API for notebooks and research

## Installation
Requires Python >= 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install optiback
```

For development (recommended if contributing):

```bash
pip install -e ".[dev]"
```

## CLI usage
The CLI is provided via the `optiback` entrypoint.

```bash
optiback --help
```

Available commands today:

- `optiback version` — print the installed version.

Planned CLI commands (roadmap):

- `optiback price ...` — price options using Black–Scholes, Binomial, or Monte Carlo
- `optiback greeks ...` — compute Greeks for a contract/spec
- `optiback iv ...` — compute implied volatility
- `optiback backtest ...` — run delta-hedge and mispricing strategies

## Python API
The initial public API is under active development. Early access will include:

- Pricing primitives (Black–Scholes, binomial tree, Monte Carlo)
- Greeks (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility solvers
- Backtesting utilities for delta hedging and mispricing

Example (subject to change as the API stabilizes):

```python
from optiback import pricing

price = pricing.black_scholes_call(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    dividend_yield=0.0,
)
```

## Data
Data helpers will support fetching spot/vol data sources for quick experiments (e.g., `yfinance`).
You can also bring your own data via CSV/Parquet.

## Development
This repository uses `ruff`, `black`, `mypy`, and `pytest`.

```bash
pip install -e ".[dev]"
ruff check src tests
black src tests
mypy src
pytest
```

Pre-commit is available:

```bash
pre-commit install
```

## Testing
Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=optiback --cov-report=term-missing
```

## Roadmap
- CLI subcommands for pricing, greeks, implied vol, and backtests
- Fast vectorized/Numba implementations for pricing and Greeks
- Backtest modules with cost models, discrete hedging, and slippage
- Example notebooks and docs site

## License
MIT
