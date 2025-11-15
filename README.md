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
- `optiback price-binomial` — price American options using Binomial Tree model (CRR)
- `optiback greeks` — calculate all option Greeks (Delta, Gamma, Vega, Theta, Rho)
- `optiback implied-vol` — calculate implied volatility from market prices

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

### Greeks command

Calculate all option Greeks (Delta, Gamma, Vega, Theta, Rho) using the Black-Scholes model:

```bash
# Calculate Greeks for a call option
optiback greeks --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call

# Calculate Greeks for a put option
optiback greeks --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type put

# Calculate Greeks with dividend yield
optiback greeks --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call --dividend 0.01
```

**Parameters:**
- Same as `price` command (see above)

**Example output:**
```
        Option Greeks          
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
│ Delta (Δ)      │       0.5576 │
│ Gamma (Γ)      │       0.0223 │
│ Vega (ν)       │       0.2791 │
│ Theta (Θ)      │      -0.0218 │
│ Rho (ρ)        │       0.2412 │
└────────────────┴──────────────┘
```

### Implied-vol command

Calculate implied volatility from market price using the Black-Scholes model:

```bash
# Calculate implied volatility for a call option
optiback implied-vol --spot 100 --strike 100 --rate 0.02 --time 0.5 --price 7.5168 --type call

# Calculate implied volatility for a put option
optiback implied-vol --spot 100 --strike 100 --rate 0.02 --time 0.5 --price 6.5218 --type put

# Calculate implied volatility with dividend yield
optiback implied-vol --spot 100 --strike 100 --rate 0.02 --time 0.5 --price 7.5168 --type call --dividend 0.01
```

**Parameters:**
- `--spot` (required): Current spot price of the underlying asset
- `--strike` (required): Strike price of the option
- `--rate` (required): Risk-free interest rate (annualized, as decimal, e.g., 0.02 for 2%)
- `--time` (required): Time to expiration in years (e.g., 0.5 for 6 months)
- `--price` (required): Observed market price of the option
- `--type` (required): Option type: `call` or `put`
- `--dividend` (optional): Dividend yield (annualized, as decimal, default: 0.0)

**Example output:**
```
       Implied Volatility           
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Parameter          ┃           Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Option Type        │            Call │
│ Spot Price         │          100.00 │
│ Strike Price       │          100.00 │
│ Risk-Free Rate     │           2.00% │
│ Time to Expiry     │    0.5000 years │
│ Market Price       │          7.5168 │
│                    │                 │
│ Implied Volatility │ 0.2500 (25.00%) │
└────────────────────┴─────────────────┘
```

### Price-binomial command

Price American call or put options using the Binomial Tree model (CRR):

```bash
# Price an American call option
optiback price-binomial --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call

# Price an American put option
optiback price-binomial --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type put

# Price with dividend yield and custom steps
optiback price-binomial --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call --dividend 0.01 --steps 200
```

**Parameters:**
- `--spot` (required): Current spot price of the underlying asset
- `--strike` (required): Strike price of the option
- `--rate` (required): Risk-free interest rate (annualized, as decimal, e.g., 0.02 for 2%)
- `--vol` (required): Volatility of the underlying asset (annualized, as decimal, e.g., 0.25 for 25%)
- `--time` (required): Time to expiration in years (e.g., 0.5 for 6 months)
- `--type` (required): Option type: `call` or `put`
- `--dividend` (optional): Dividend yield (annualized, as decimal, default: 0.0)
- `--steps` (optional): Number of time steps in the binomial tree (default: 100)

**Example output:**
```
  Binomial Tree Option Pricing   
             Results             
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Parameter      ┃        Value ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Option Type    │         Call │
│ Spot Price     │       100.00 │
│ Strike Price   │       100.00 │
│ Risk-Free Rate │        2.00% │
│ Volatility     │       25.00% │
│ Time to Expiry │ 0.5000 years │
│ Steps          │          100 │
│                │              │
│ Option Price   │       7.4993 │
└────────────────┴──────────────┘
```

### Planned CLI commands (roadmap)

- `optiback backtest ...` — run delta-hedge and mispricing strategies

## Python API

OptiBack provides a comprehensive Python API for options pricing, Greeks, and implied volatility calculations.

### Pricing

#### Black-Scholes Model

```python
from optiback.pricing import black_scholes_call, black_scholes_put

# Price a call option
call_price = black_scholes_call(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    dividend_yield=0.0,
)
print(f"Call price: {call_price:.4f}")  # Output: Call price: 7.5168

# Price a put option
put_price = black_scholes_put(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
)
print(f"Put price: {put_price:.4f}")  # Output: Put price: 6.5218
```

#### Binomial Tree Model (American Options)

Price American options using the Cox-Ross-Rubinstein (CRR) binomial tree model:

```python
from optiback.pricing import binomial_tree_call, binomial_tree_put

# Price an American call option
call_price = binomial_tree_call(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    dividend_yield=0.0,
    steps=100,  # Number of time steps (default: 100)
)
print(f"American call price: {call_price:.4f}")  # Output: American call price: 7.4993

# Price an American put option
put_price = binomial_tree_put(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    steps=100,
)
print(f"American put price: {put_price:.4f}")  # Output: American put price: 6.5857

# Higher steps for better accuracy (converges to Black-Scholes for European options)
call_price_precise = binomial_tree_call(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    steps=500,  # More steps = higher accuracy
)
```

### Greeks

```python
from optiback.pricing import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_theta,
    black_scholes_rho,
    black_scholes_greeks,
)

# Calculate individual Greeks
delta = black_scholes_delta(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    option_type="call",
)
print(f"Delta: {delta:.4f}")  # Output: Delta: 0.5576

# Calculate all Greeks at once
greeks = black_scholes_greeks(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    option_type="call",
)
print(f"All Greeks: {greeks}")
# Output: {'delta': 0.5576, 'gamma': 0.0223, 'vega': 0.2791, 'theta': -0.0218, 'rho': 0.2412}
```

### Implied Volatility

```python
from optiback.pricing import black_scholes_implied_volatility

# Calculate implied volatility from market price
implied_vol = black_scholes_implied_volatility(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    time_to_expiry=0.5,
    market_price=7.5168,  # Observed market price
    option_type="call",
)
print(f"Implied volatility: {implied_vol:.4f}")  # Output: Implied volatility: 0.2500
```

### Array Support

All pricing functions support both scalar and NumPy array inputs:

```python
import numpy as np
from optiback.pricing import black_scholes_call, binomial_tree_call

# Price multiple options at once with Black-Scholes
spots = np.array([90.0, 100.0, 110.0])
strikes = np.array([100.0, 100.0, 100.0])
prices = black_scholes_call(
    spot=spots,
    strike=strikes,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
)
print(prices)  # Array of prices

# Price multiple American options with Binomial Tree
american_prices = binomial_tree_call(
    spot=spots,
    strike=strikes,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    steps=100,
)
print(american_prices)  # Array of American option prices
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
- Fast vectorized/Numba implementations for pricing and Greeks
- Additional pricing models (Monte Carlo)
- Backtest modules with cost models, discrete hedging, and slippage
- Example notebooks and docs site

## License
MIT
