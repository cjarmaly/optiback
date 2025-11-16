"""Backtesting module for options trading strategies."""

from optiback.backtest.costs import (
    apply_slippage,
    calculate_total_execution_cost,
    calculate_transaction_cost,
)
from optiback.backtest.delta_hedge import backtest_delta_hedge
from optiback.backtest.engine import BacktestResult, calculate_sharpe_ratio
from optiback.backtest.mispricing import backtest_mispricing

__all__ = [
    "apply_slippage",
    "backtest_delta_hedge",
    "backtest_mispricing",
    "BacktestResult",
    "calculate_sharpe_ratio",
    "calculate_total_execution_cost",
    "calculate_transaction_cost",
]
