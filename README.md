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
cd optiback

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

**Features:**
- **Vectorized operations**: All pricing functions support both scalar and array inputs for efficient batch processing

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

**`optiback: command not found`:**
- Activate the virtual environment: `source .venv/bin/activate`
- Reinstall in editable mode: `pip install -e ".[dev]"`
- If the venv was created at a different path, recreate it: `rm -rf .venv && python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`

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
- `optiback price-montecarlo` — price European options using Monte Carlo simulation
- `optiback greeks` — calculate all option Greeks (Delta, Gamma, Vega, Theta, Rho)
- `optiback implied-vol` — calculate implied volatility from market prices
- `optiback backtest-delta-hedge` — backtest a delta-hedged option position
- `optiback backtest-mispricing` — backtest trading on theoretical vs market mispricing

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

### Price-montecarlo command

Price European call or put options using Monte Carlo simulation:

```bash
# Price an European call option
optiback price-montecarlo --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call

# Price an European put option
optiback price-montecarlo --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type put

# Price with dividend yield, custom simulations, and seed
optiback price-montecarlo --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call --dividend 0.01 --simulations 200000 --seed 42
```

**Parameters:**
- `--spot` (required): Current spot price of the underlying asset
- `--strike` (required): Strike price of the option
- `--rate` (required): Risk-free interest rate (annualized, as decimal, e.g., 0.02 for 2%)
- `--vol` (required): Volatility of the underlying asset (annualized, as decimal, e.g., 0.25 for 25%)
- `--time` (required): Time to expiration in years (e.g., 0.5 for 6 months)
- `--type` (required): Option type: `call` or `put`
- `--dividend` (optional): Dividend yield (annualized, as decimal, default: 0.0)
- `--simulations` (optional): Number of Monte Carlo simulations (default: 100000)
- `--seed` (optional): Random seed for reproducibility (integer, default: None)

**Example output:**
```
   Monte Carlo Option Pricing
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
│ Simulations    │      100,000 │
│ Seed           │           42 │
│                │              │
│ Option Price   │       7.5288 │
└────────────────┴──────────────┘
```

### Backtest-delta-hedge command

Backtest a delta-hedged option position with transaction costs and slippage:

```bash
# From a CSV/Parquet file
optiback backtest-delta-hedge --spot-file prices.csv --strike 100 --rate 0.02 --vol 0.25 --time 0.25 --type call

# Fetch spot history via yfinance
optiback backtest-delta-hedge --ticker SPY --strike 450 --rate 0.04 --vol 0.20 --time 0.25 --type call

# Save equity curve and use weekly rebalancing
optiback backtest-delta-hedge --ticker SPY --strike 450 --rate 0.04 --vol 0.20 --time 0.25 --type call \
  --rebalance-frequency weekly --output-csv equity.csv
```

**Parameters (in addition to pricing params):**
- `--spot-file` or `--ticker` (one required): price series from file or yfinance
- `--start` / `--end`: date range for `--ticker` (`YYYY-MM-DD`)
- `--period`: yfinance period when dates not set (default: `3mo`)
- `--option-position`: option size, negative = short (default: `-1.0`)
- `--rebalance-frequency`: `daily`, `weekly`, or `monthly` (default: `daily`)
- `--output-csv`: optional path to save equity curve

**Output:** summary table with P&L, costs, trade count, and annualized Sharpe ratio. Use `--output-csv` to persist the equity curve (`equity` column).

### Backtest-mispricing command

Backtest buying/selling when market option prices diverge from theoretical value:

```bash
optiback backtest-mispricing --spot-file spots.csv --market-price-file prices.csv \
  --strike 100 --rate 0.02 --vol 0.25 --time 0.25 --type call --model black_scholes

optiback backtest-mispricing --ticker SPY --market-price-file prices.csv \
  --strike 450 --rate 0.04 --vol 0.20 --time 0.25 --type call --threshold 0.05 --steps 200
```

**Parameters (in addition to pricing params):**
- `--spot-file` or `--ticker` (one required)
- `--market-price-file` (required): observed option prices, same length as spot series
- `--model`: `black_scholes`, `binomial`, or `monte_carlo` (default: `black_scholes`)
- `--threshold`: mispricing trigger as decimal (default: `0.05` = 5%)
- `--steps`: binomial tree steps when using `--model binomial` (default: `100`)
- `--output-csv`: optional path to save equity curve

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

#### Monte Carlo Model (European Options)

Price European options using Monte Carlo simulation with geometric Brownian motion:

```python
from optiback.pricing import monte_carlo_call, monte_carlo_put

# Price an European call option
call_price = monte_carlo_call(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    dividend_yield=0.0,
    simulations=100000,  # Number of simulations (default: 100000)
    seed=42,  # Random seed for reproducibility (optional)
)
print(f"European call price: {call_price:.4f}")  # Output: European call price: 7.5288

# Price an European put option
put_price = monte_carlo_put(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    simulations=100000,
    seed=42,
)
print(f"European put price: {put_price:.4f}")  # Output: European put price: 6.5255

# More simulations for better accuracy (converges to Black-Scholes)
call_price_precise = monte_carlo_call(
    spot=100.0,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    simulations=500000,  # More simulations = higher accuracy
    seed=42,
)
```

**Note:** Monte Carlo simulation includes antithetic variates for variance reduction, improving accuracy with fewer simulations.

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
from optiback.pricing import black_scholes_call, binomial_tree_call, monte_carlo_call

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

# Price multiple European options with Monte Carlo
mc_prices = monte_carlo_call(
    spot=spots,
    strike=strikes,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.5,
    simulations=100000,
    seed=42,
)
print(mc_prices)  # Array of European option prices
```

### Backtesting

```python
import numpy as np
from optiback.backtest import backtest_delta_hedge, backtest_mispricing
from optiback.data import fetch_spot_history, load_prices

# Delta-hedge with file or fetched data
spots = load_prices("prices.csv").to_numpy()
# spots = fetch_spot_history("SPY", period="3mo").to_numpy()

result = backtest_delta_hedge(
    spot_prices=spots,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.25,
    option_type="call",
    rebalance_frequency="daily",
)
print(result.summary())
print(f"Sharpe: {result.sharpe_ratio:.4f}")
print(result.equity_curve)  # portfolio value per period

# Mispricing strategy
market_prices = np.array([7.0, 7.5, 6.5, 8.0, 7.0])
mispricing = backtest_mispricing(
    spot_prices=spots[:5],
    market_option_prices=market_prices,
    strike=100.0,
    rate=0.02,
    vol=0.25,
    time_to_expiry=0.25,
    option_type="call",
    theoretical_model="black_scholes",
    steps=100,
)
```

## Data

Load prices from CSV/Parquet or fetch spot history via yfinance:

```python
from optiback.data import fetch_spot_history, load_prices, save_prices

# Load from file (auto-detects Close/price column or single-column CSV)
series = load_prices("prices.csv")

# Fetch market data
spots = fetch_spot_history("SPY", period="3mo")
spots = fetch_spot_history("AAPL", start="2024-01-01", end="2024-06-01")

# Save for reuse
save_prices(spots, "spy_close.parquet")
```

Missing or incomplete bars (e.g. yfinance trailing NaN on the current day) are dropped automatically. Backtests reject price series that still contain NaN after cleaning.

## Project structure

```
optiback/
├── examples/                  # End-to-end walkthrough scripts
├── src/optiback/
│   ├── backtest/              # Delta-hedge and mispricing engines
│   ├── cli/                   # Typer CLI (`optiback` entrypoint)
│   ├── data/                  # Price load/save and yfinance fetch
│   └── pricing/               # BS, binomial, Monte Carlo, Greeks, IV
└── tests/
```

## Examples

End-to-end walkthrough (fetch or synthetic data → backtest → plot):

```bash
python examples/delta_hedge_walkthrough.py --synthetic
python examples/delta_hedge_walkthrough.py --ticker SPY --output equity_curve.png
```

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
- Docs site and additional example notebooks
- Expanded data helpers (vol surfaces, options chains)
- Additional backtest strategies and portfolio-level analytics

## License
MIT
