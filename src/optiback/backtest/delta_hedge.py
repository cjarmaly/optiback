"""Delta-hedge backtesting strategy."""

from __future__ import annotations

import numpy as np

from optiback.backtest.costs import calculate_total_execution_cost
from optiback.backtest.engine import BacktestResult
from optiback.pricing import black_scholes_call, black_scholes_delta, black_scholes_put


def backtest_delta_hedge(
    spot_prices: np.ndarray,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str = "call",
    option_position: float = -1.0,  # Negative = short option (typical for delta-hedge)
    rebalance_frequency: str = "daily",
    dividend_yield: float = 0.0,
    cost_per_share: float = 0.01,
    bid_ask_spread_bps: float = 5.0,
    slippage_bps: float = 5.0,
    impact_factor: float = 0.1,
) -> BacktestResult:
    """
    Backtest a delta-hedged option position.

    Simulates maintaining a delta-neutral portfolio by rebalancing stock positions
    as the option's delta changes over time. Accounts for transaction costs,
    slippage, and discrete rebalancing.

    Args:
        spot_prices: Array of spot prices over time (e.g., daily prices)
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized)
        vol: Volatility (annualized)
        time_to_expiry: Initial time to expiry in years
        option_type: Option type ("call" or "put")
        option_position: Option position size (negative = short, positive = long)
        rebalance_frequency: Rebalancing frequency ("daily", "hourly", etc.)
        dividend_yield: Dividend yield (annualized, default: 0.0)
        cost_per_share: Transaction cost per share (default: $0.01)
        bid_ask_spread_bps: Bid-ask spread in basis points (default: 5 bps)
        slippage_bps: Slippage in basis points (default: 5 bps)
        impact_factor: Market impact factor (default: 0.1)

    Returns:
        BacktestResult with P&L, costs, and performance metrics

    Examples:
        >>> import numpy as np
        >>> spots = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
        >>> result = backtest_delta_hedge(
        ...     spot_prices=spots,
        ...     strike=100.0,
        ...     rate=0.02,
        ...     vol=0.25,
        ...     time_to_expiry=0.25,
        ...     option_type="call"
        ... )
        >>> result.num_trades >= 0
        True
    """
    if len(spot_prices) < 2:
        raise ValueError("spot_prices must have at least 2 values")

    num_periods = len(spot_prices)
    dt = time_to_expiry / num_periods  # Time step per period

    # Initialize positions
    stock_position = 0.0  # Current stock position (shares)
    cash = 0.0  # Cash position (dollars)

    # Track costs
    total_transaction_costs = 0.0
    total_slippage_costs = 0.0
    num_trades = 0

    # Calculate initial option price and delta
    initial_spot = spot_prices[0]
    if option_type == "call":
        initial_option_price = black_scholes_call(
            spot=initial_spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            dividend_yield=dividend_yield,
        )
    else:  # put
        initial_option_price = black_scholes_put(
            spot=initial_spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            dividend_yield=dividend_yield,
        )

    initial_delta = black_scholes_delta(
        spot=initial_spot,
        strike=strike,
        rate=rate,
        vol=vol,
        time_to_expiry=time_to_expiry,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )
    # Cast to float since we're using scalar inputs
    initial_delta_val: float = float(initial_delta) if not isinstance(initial_delta, float) else initial_delta

    # Initial hedge: buy/sell stock to offset option delta
    target_delta = -option_position * initial_delta_val  # Need to hedge option's delta
    shares_to_trade: float = float(target_delta - stock_position)

    if abs(shares_to_trade) > 1e-10:  # Avoid tiny trades
        exec_price, cost = calculate_total_execution_cost(
            trade_size=shares_to_trade,
            reference_price=initial_spot,
            cost_per_share=cost_per_share,
            bid_ask_spread_bps=bid_ask_spread_bps,
            slippage_bps=slippage_bps,
            impact_factor=impact_factor,
        )
        exec_price_val: float = float(exec_price)
        cost_val: float = float(cost)

        cash -= shares_to_trade * exec_price_val  # Buy = negative cash
        cash -= cost_val  # Pay transaction cost
        stock_position += shares_to_trade
        total_transaction_costs += cost_val
        total_slippage_costs += abs(shares_to_trade) * (exec_price_val - initial_spot)
        num_trades += 1

    initial_value = option_position * initial_option_price + stock_position * initial_spot + cash

    # Simulate over time
    for i in range(1, num_periods):
        current_spot = spot_prices[i]
        remaining_time = time_to_expiry - i * dt

        if remaining_time <= 0:
            break

        # Calculate current delta
        current_delta = black_scholes_delta(
            spot=current_spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=remaining_time,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )
        # Cast to float since we're using scalar inputs
        current_delta_val: float = float(current_delta) if not isinstance(current_delta, float) else current_delta

        # Calculate target hedge
        target_delta = -option_position * current_delta_val
        shares_to_trade = float(target_delta - stock_position)

        # Rebalance if needed (threshold to avoid tiny trades)
        if abs(shares_to_trade) > 1e-6:
            exec_price, cost = calculate_total_execution_cost(
                trade_size=shares_to_trade,
                reference_price=current_spot,
                cost_per_share=cost_per_share,
                bid_ask_spread_bps=bid_ask_spread_bps,
                slippage_bps=slippage_bps,
                impact_factor=impact_factor,
            )
            exec_price_val = float(exec_price)
            cost_val = float(cost)

            cash -= shares_to_trade * exec_price_val  # Buy = negative cash
            cash -= cost_val
            stock_position += shares_to_trade
            total_transaction_costs += cost_val
            total_slippage_costs += abs(shares_to_trade) * (exec_price_val - current_spot)
            num_trades += 1

    # Final settlement
    final_spot = spot_prices[-1]
    final_option_price = 0.0  # Option expired (or calculate final value)
    if remaining_time <= dt:  # Option expired
        if option_type == "call":
            final_option_price = max(final_spot - strike, 0.0)
        else:  # put
            final_option_price = max(strike - final_spot, 0.0)
    else:
        # Option still has value (shouldn't happen in typical backtest)
        if option_type == "call":
            final_option_price_raw = black_scholes_call(
                spot=final_spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=remaining_time,
                dividend_yield=dividend_yield,
            )
            final_option_price = float(final_option_price_raw) if not isinstance(final_option_price_raw, float) else final_option_price_raw
        else:  # put
            final_option_price_raw = black_scholes_put(
                spot=final_spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=remaining_time,
                dividend_yield=dividend_yield,
            )
            final_option_price = float(final_option_price_raw) if not isinstance(final_option_price_raw, float) else final_option_price_raw

    # Final portfolio value
    option_pnl = option_position * (final_option_price - initial_option_price)
    stock_pnl = stock_position * final_spot  # Stock position value
    final_value = option_pnl + stock_pnl + cash

    total_pnl = final_value - initial_value

    return BacktestResult(
        total_pnl=total_pnl,
        transaction_costs=total_transaction_costs,
        slippage_costs=total_slippage_costs,
        num_trades=num_trades,
        initial_value=initial_value,
        final_value=final_value,
        returns=0.0,  # Calculated in __post_init__
        strategy_type="delta_hedge",
    )
