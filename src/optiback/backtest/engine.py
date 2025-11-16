"""Core backtesting engine and result structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class BacktestResult:
    """
    Results from a backtest.

    Attributes:
        total_pnl: Total profit and loss from the strategy
        transaction_costs: Total transaction costs paid
        slippage_costs: Total slippage costs (difference between reference and execution prices)
        num_trades: Number of trades executed
        initial_value: Initial portfolio value
        final_value: Final portfolio value
        returns: Total return as a decimal (e.g., 0.05 for 5%)
        strategy_type: Type of strategy ("delta_hedge" or "mispricing")
    """

    total_pnl: float
    transaction_costs: float
    slippage_costs: float
    num_trades: int
    initial_value: float
    final_value: float
    returns: float
    strategy_type: Literal["delta_hedge", "mispricing"]

    def __post_init__(self) -> None:
        """Calculate returns from initial and final values."""
        if self.initial_value != 0:
            self.returns = (self.final_value - self.initial_value) / self.initial_value
        else:
            self.returns = 0.0

    def summary(self) -> dict[str, float | int | str]:
        """Return a summary dictionary of backtest results."""
        return {
            "strategy": self.strategy_type,
            "total_pnl": self.total_pnl,
            "transaction_costs": self.transaction_costs,
            "slippage_costs": self.slippage_costs,
            "net_pnl": self.total_pnl - self.transaction_costs - self.slippage_costs,
            "num_trades": self.num_trades,
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "returns": self.returns,
            "returns_pct": self.returns * 100,
        }


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio from returns array.

    Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Deviation of Returns

    Args:
        returns: Array of period returns
        risk_free_rate: Risk-free rate (annualized, default: 0.0)

    Returns:
        Sharpe ratio (annualized if returns are daily, adjust accordingly)
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / len(
        returns
    )  # Approximate per-period risk-free rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(returns)

    if std_excess_return == 0:
        return 0.0

    # Annualize if needed (assuming daily returns, multiply by sqrt(252))
    annualized_sharpe = (mean_excess_return / std_excess_return) * np.sqrt(252)
    return float(annualized_sharpe)
