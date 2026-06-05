"""Core backtesting engine and result structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


def rebalance_periods(rebalance_frequency: str) -> int:
    """
    Map a rebalance frequency label to a number of price bars between rebalances.

    Assumes each row in ``spot_prices`` is one bar (e.g. daily). Supported values:
    ``daily`` (1), ``weekly`` (5), ``monthly`` (21).
    """
    mapping = {
        "daily": 1,
        "hourly": 1,
        "weekly": 5,
        "monthly": 21,
    }
    key = rebalance_frequency.lower()
    if key not in mapping:
        raise ValueError(
            f"rebalance_frequency must be one of {list(mapping.keys())}, got '{rebalance_frequency}'"
        )
    return mapping[key]


def compute_period_returns(equity_curve: np.ndarray) -> np.ndarray:
    """Compute period-over-period returns from an equity curve."""
    if len(equity_curve) < 2:
        return np.array([], dtype=np.float64)

    prev = equity_curve[:-1]
    safe_prev = np.where(prev != 0, prev, np.nan)
    returns: np.ndarray = np.diff(equity_curve) / safe_prev
    cleaned: np.ndarray = np.nan_to_num(returns, nan=0.0).astype(np.float64)
    return cleaned


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
        equity_curve: Portfolio value at each period (optional)
        period_returns: Period-over-period returns derived from equity_curve (optional)
        sharpe_ratio: Annualized Sharpe ratio from period_returns (optional)
    """

    total_pnl: float
    transaction_costs: float
    slippage_costs: float
    num_trades: int
    initial_value: float
    final_value: float
    returns: float
    strategy_type: Literal["delta_hedge", "mispricing"]
    equity_curve: np.ndarray | None = field(default=None, repr=False)
    period_returns: np.ndarray | None = field(default=None, repr=False)
    sharpe_ratio: float | None = None

    def __post_init__(self) -> None:
        """Calculate returns and optional analytics from equity curve."""
        if self.initial_value != 0:
            self.returns = (self.final_value - self.initial_value) / self.initial_value
        else:
            self.returns = 0.0

        if self.equity_curve is not None and len(self.equity_curve) >= 2:
            if self.period_returns is None:
                self.period_returns = compute_period_returns(self.equity_curve)
            if self.sharpe_ratio is None and len(self.period_returns) > 0:
                self.sharpe_ratio = calculate_sharpe_ratio(self.period_returns)

    def summary(self) -> dict[str, float | int | str | None]:
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
            "sharpe_ratio": self.sharpe_ratio,
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
