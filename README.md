# OptiBack

Options pricing & backtesting engine:
- Pricing: Black–Scholes, Binomial, Monte Carlo
- Greeks & implied vol
- Backtests: delta-hedge, mispricing with realistic costs

## Prerequisites

- **Python 3.11 or higher** (Python 3.12 recommended)
- **pip** (usually included with Python)
- **Git** (to clone the repository)

### Installing Python 3.11+

**macOS (Homebrew):**
```bash
brew install python@3.12
# Or
brew install python@3.11
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
# Or python3.11
```

**Windows:**
- Download from https://www.python.org/downloads/
- Or use: `winget install Python.Python.3.12`

Verify your Python version:
```bash
python3 --version  # Should show 3.11.x or 3.12.x
```

## Quickstart

```bash
# 1. Clone the repository
git clone <repository-url>
cd optiback-1

# 2. Create virtual environment with Python 3.11+
python3.12 -m venv .venv  # Or python3.11, or python3 if it's 3.11+
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install the package
pip install -e ".[dev]"

# 5. Verify installation
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

### For Development (Recommended)

If you're contributing or want to run tests:

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

This installs:
- The package in editable mode (code changes reflected immediately)
- All runtime dependencies (numpy, scipy, pandas, etc.)
- Development tools (pytest, black, ruff, mypy, etc.)

### For Use Only

If you just want to use the package:

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .
```

### Troubleshooting

**Python version error:**
- Ensure you have Python 3.11 or higher installed
- Use `python3.12 -m venv .venv` to explicitly specify the version

**ModuleNotFoundError:**
- Make sure you've run `pip install -e .` or `pip install -e ".[dev]"`
- Verify you're in the virtual environment: `which python` should show `.venv/bin/python`

**Scipy/NumPy compatibility issues:**
- The project pins numpy to `<2.3` to avoid compatibility issues
- If you encounter errors, reinstall: `pip install "numpy>=1.26,<2.3" --upgrade`

## CLI usage
The CLI is provided via the `optiback` entrypoint.

```bash
optiback --help
```

### Available commands

- `optiback version` — print the installed version.
- `optiback price` — price options using Black–Scholes model

### Price command

Price European call or put options using the Black–Scholes model:

```bash
# Price a call option
optiback price --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call

# Price a put option
optiback price --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type put

# Price with dividend yield
optiback price --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call --dividend 0.01
```

**Parameters:**
- `--spot` (required): Current spot price of the underlying asset
- `--strike` (required): Strike price of the option
- `--rate` (required): Risk-free interest rate (annualized, as decimal, e.g., 0.02 for 2%)
- `--vol` (required): Volatility of the underlying asset (annualized, as decimal, e.g., 0.25 for 25%)
- `--time` (required): Time to expiration in years (e.g., 0.5 for 6 months)
- `--type` (required): Option type: `call` or `put`
- `--dividend` (optional): Dividend yield (annualized, as decimal, default: 0.0)

**Example output:**
```
     Option Pricing Results      
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Parameter      ┃        Value ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Option Type    │         Call │
│ Spot Price     │       100.00 │
│ Strike Price   │       100.00 │
│ Risk-Free Rate │        2.00% │
│ Volatility     │       25.00% │
│ Time to Expiry │ 0.5000 years │
│                │              │
│ Option Price   │       7.5168 │
└────────────────┴──────────────┘
```

### Planned CLI commands (roadmap)

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
from optiback import black_scholes_call

price = black_scholes_call(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    dividend_yield=0.0,
)
print(f"Option price: {price:.4f}")  # Output: Option price: 7.5168
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
