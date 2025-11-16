"""Mispricing detection and trading backtesting strategy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optiback.backtest.costs import calculate_total_execution_cost
from optiback.backtest.engine import BacktestResult
from optiback.pricing import (
    binomial_tree_call,
    binomial_tree_put,
    black_scholes_call,
    black_scholes_put,
    monte_carlo_call,
    monte_carlo_put,
)


def _get_theoretical_price(
    price_func: Callable,
    current_spot: float,
    strike: float,
    rate: float,
    vol: float,
    remaining_time: float,
    dividend_yield: float,
    theoretical_model: str,
    steps: int | None = None,
    simulations: int | None = None,
    seed: int | None = None,
) -> float:
    """Calculate theoretical option price based on model."""
    if theoretical_model == "binomial":
        price = price_func(
            spot=current_spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=remaining_time,
            dividend_yield=dividend_yield,
            steps=steps or 100,
        )
        return float(price) if not isinstance(price, float) else price
    elif theoretical_model == "monte_carlo":
        price = price_func(
            spot=current_spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=remaining_time,
            dividend_yield=dividend_yield,
            simulations=simulations or 100000,
            seed=seed,
        )
        return float(price) if not isinstance(price, float) else price
    else:  # black_scholes
        price = price_func(
            spot=current_spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=remaining_time,
            dividend_yield=dividend_yield,
        )
        return float(price) if not isinstance(price, float) else price


def _determine_trade_size(
    mispricing_pct: float,
    mispricing_threshold: float,
    option_position: float,
) -> float:
    """Determine trade size based on mispricing percentage."""
    if mispricing_pct > mispricing_threshold:
        # Undervalued: buy options
        return 1.0
    elif mispricing_pct < -mispricing_threshold:
        # Overvalued: sell options
        if option_position > 0:
            return -option_position  # Sell all options
        return -1.0  # Short 1 option
    return 0.0


def _execute_trade(
    trade_size: float,
    market_price: float,
    option_position: float,
    cash: float,
    cost_per_share: float,
    bid_ask_spread_bps: float,
    slippage_bps: float,
    impact_factor: float,
) -> tuple[float, float, float, float, float]:
    """Execute a trade and return updated positions and costs."""
    exec_price, cost = calculate_total_execution_cost(
        trade_size=trade_size,
        reference_price=market_price,
        cost_per_share=cost_per_share,
        bid_ask_spread_bps=bid_ask_spread_bps,
        slippage_bps=slippage_bps,
        impact_factor=impact_factor,
    )

    cash -= trade_size * exec_price  # Buy = negative cash
    cash -= cost
    option_position += trade_size
    transaction_cost = cost
    slippage_cost = abs(trade_size) * (exec_price - market_price)

    return option_position, cash, transaction_cost, slippage_cost, exec_price


def _settle_final_position(
    option_position: float,
    spot_prices: np.ndarray,
    market_option_prices: np.ndarray,
    strike: float,
    option_type: str,
    remaining_time: float,
    dt: float,
    cost_per_share: float,
    bid_ask_spread_bps: float,
    slippage_bps: float,
    impact_factor: float,
) -> tuple[float, float, float]:
    """Settle final option position and return final value and costs."""
    final_transaction_cost = 0.0
    final_slippage_cost = 0.0

    if option_position != 0 and remaining_time <= dt:
        # Option expired - calculate final value
        final_spot = spot_prices[-1]
        if option_type == "call":
            final_value = max(final_spot - strike, 0.0)
        else:  # put
            final_value = max(strike - final_spot, 0.0)
        return final_value, final_transaction_cost, final_slippage_cost

    if option_position != 0:
        # Close position at final market price
        final_market_price = market_option_prices[-1]
        exec_price, cost = calculate_total_execution_cost(
            trade_size=-option_position,
            reference_price=final_market_price,
            cost_per_share=cost_per_share,
            bid_ask_spread_bps=bid_ask_spread_bps,
            slippage_bps=slippage_bps,
            impact_factor=impact_factor,
        )
        final_transaction_cost = cost
        final_slippage_cost = abs(option_position) * (exec_price - final_market_price)
        # Return cash from closing position (negative because we're closing)
        cash_from_close = option_position * exec_price - cost
        return cash_from_close, final_transaction_cost, final_slippage_cost

    return 0.0, final_transaction_cost, final_slippage_cost


def backtest_mispricing(
    spot_prices: np.ndarray,
    market_option_prices: np.ndarray,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str = "call",
    theoretical_model: str = "black_scholes",
    dividend_yield: float = 0.0,
    mispricing_threshold: float = 0.05,  # 5% mispricing threshold
    cost_per_share: float = 0.01,
    bid_ask_spread_bps: float = 5.0,
    slippage_bps: float = 5.0,
    impact_factor: float = 0.1,
    simulations: int = 100000,
    seed: int | None = None,
) -> BacktestResult:
    """
    Backtest buying/selling options when they're mispriced relative to theoretical value.

    Compares market prices to theoretical prices and executes trades when mispricing
    exceeds threshold. Accounts for transaction costs and slippage.

    Args:
        spot_prices: Array of spot prices over time
        market_option_prices: Array of market option prices (same length as spot_prices)
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized)
        vol: Volatility (annualized)
        time_to_expiry: Initial time to expiry in years
        option_type: Option type ("call" or "put")
        theoretical_model: Pricing model to use ("black_scholes", "binomial", "monte_carlo")
        dividend_yield: Dividend yield (annualized, default: 0.0)
        mispricing_threshold: Minimum mispricing percentage to trigger trade (default: 0.05 = 5%)
        cost_per_share: Transaction cost per share for option (default: $0.01)
        bid_ask_spread_bps: Bid-ask spread in basis points (default: 5 bps)
        slippage_bps: Slippage in basis points (default: 5 bps)
        impact_factor: Market impact factor (default: 0.1)
        simulations: Number of simulations for Monte Carlo (default: 100000)
        seed: Random seed for Monte Carlo (default: None)

    Returns:
        BacktestResult with P&L, costs, and performance metrics

    Examples:
        >>> import numpy as np
        >>> spots = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        >>> market_prices = np.array([7.0, 7.5, 6.5, 8.0, 7.0])
        >>> result = backtest_mispricing(
        ...     spot_prices=spots,
        ...     market_option_prices=market_prices,
        ...     strike=100.0,
        ...     rate=0.02,
        ...     vol=0.25,
        ...     time_to_expiry=0.25,
        ...     option_type="call"
        ... )
        >>> result.num_trades >= 0
        True
    """
    if len(spot_prices) != len(market_option_prices):
        raise ValueError("spot_prices and market_option_prices must have same length")

    if len(spot_prices) < 1:
        raise ValueError("spot_prices must have at least 1 value")

    num_periods = len(spot_prices)
    dt = time_to_expiry / num_periods

    # Initialize positions
    option_position = 0.0  # Current option position
    cash = 0.0  # Cash position

    # Track costs
    total_transaction_costs = 0.0
    total_slippage_costs = 0.0
    num_trades = 0

    # Pricing function based on model
    if theoretical_model == "black_scholes":
        price_func = black_scholes_call if option_type == "call" else black_scholes_put
    elif theoretical_model == "binomial":
        price_func = binomial_tree_call if option_type == "call" else binomial_tree_put
        # Binomial needs steps parameter
        steps = 100
    elif theoretical_model == "monte_carlo":
        price_func = monte_carlo_call if option_type == "call" else monte_carlo_put
    else:
        raise ValueError(
            f"theoretical_model must be 'black_scholes', 'binomial', or 'monte_carlo', got '{theoretical_model}'"
        )

    # Simulate trading
    for i in range(num_periods):
        current_spot = spot_prices[i]
        market_price = market_option_prices[i]
        remaining_time = time_to_expiry - i * dt

        if remaining_time <= 0:
            break

        # Calculate theoretical price
        theoretical_price = _get_theoretical_price(
            price_func=price_func,
            current_spot=current_spot,
            strike=strike,
            rate=rate,
            vol=vol,
            remaining_time=remaining_time,
            dividend_yield=dividend_yield,
            theoretical_model=theoretical_model,
            steps=steps if theoretical_model == "binomial" else None,
            simulations=simulations if theoretical_model == "monte_carlo" else None,
            seed=seed if theoretical_model == "monte_carlo" else None,
        )

        # Check for mispricing
        mispricing_pct = (
            (theoretical_price - market_price) / theoretical_price if theoretical_price > 0 else 0.0
        )

        # Determine trade size
        trade_size = _determine_trade_size(mispricing_pct, mispricing_threshold, option_position)

        # Execute trade if needed
        if abs(trade_size) > 1e-10:
            option_position, cash, cost, slippage_cost, _exec_price = _execute_trade(
                trade_size=trade_size,
                market_price=market_price,
                option_position=option_position,
                cash=cash,
                cost_per_share=cost_per_share,
                bid_ask_spread_bps=bid_ask_spread_bps,
                slippage_bps=slippage_bps,
                impact_factor=impact_factor,
            )

            total_transaction_costs += cost
            total_slippage_costs += slippage_cost
            num_trades += 1

    # Final settlement
    settlement_value, final_cost, final_slippage = _settle_final_position(
        option_position=option_position,
        spot_prices=spot_prices,
        market_option_prices=market_option_prices,
        strike=strike,
        option_type=option_type,
        remaining_time=remaining_time if remaining_time > 0 else dt,
        dt=dt,
        cost_per_share=cost_per_share,
        bid_ask_spread_bps=bid_ask_spread_bps,
        slippage_bps=slippage_bps,
        impact_factor=impact_factor,
    )

    total_transaction_costs += final_cost
    total_slippage_costs += final_slippage
    if final_cost > 0 or final_slippage > 0:
        num_trades += 1

    final_value = cash + settlement_value

    # Calculate P&L
    initial_value = 0.0  # Start with no capital (or can modify)
    total_pnl = final_value - initial_value

    return BacktestResult(
        total_pnl=total_pnl,
        transaction_costs=total_transaction_costs,
        slippage_costs=total_slippage_costs,
        num_trades=num_trades,
        initial_value=initial_value,
        final_value=final_value,
        returns=0.0,  # Calculated in __post_init__
        strategy_type="mispricing",
    )
