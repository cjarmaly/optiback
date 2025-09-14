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
