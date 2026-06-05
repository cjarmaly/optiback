"""Delta-hedge backtesting strategy."""

from __future__ import annotations

import numpy as np

from optiback.backtest.costs import calculate_total_execution_cost
from optiback.backtest.engine import BacktestResult, rebalance_periods
from optiback.pricing import black_scholes_call, black_scholes_delta, black_scholes_put
from optiback.pricing.array import as_float


def _option_mark_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str,
    dividend_yield: float,
) -> float:
    if time_to_expiry <= 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)

    price_func = black_scholes_call if option_type == "call" else black_scholes_put
    return as_float(
        price_func(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            dividend_yield=dividend_yield,
        )
    )


def _portfolio_value(
    spot: float,
    option_position: float,
    option_price: float,
    stock_position: float,
    cash: float,
) -> float:
    return option_position * option_price + stock_position * spot + cash


def _trade_stock(
    shares: float,
    reference_price: float,
    stock_position: float,
    cash: float,
    *,
    cost_per_share: float,
    bid_ask_spread_bps: float,
    slippage_bps: float,
    impact_factor: float,
) -> tuple[float, float, float, float, int]:
    if abs(shares) <= 1e-6:
        return stock_position, cash, 0.0, 0.0, 0

    exec_price, cost = calculate_total_execution_cost(
        trade_size=shares,
        reference_price=reference_price,
        cost_per_share=cost_per_share,
        bid_ask_spread_bps=bid_ask_spread_bps,
        slippage_bps=slippage_bps,
        impact_factor=impact_factor,
    )
    cash -= shares * float(exec_price) + float(cost)
    stock_position += shares
    slippage = abs(shares) * (float(exec_price) - reference_price)
    return stock_position, cash, float(cost), slippage, 1


def backtest_delta_hedge(
    spot_prices: np.ndarray,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str = "call",
    option_position: float = -1.0,
    rebalance_frequency: str = "daily",
    dividend_yield: float = 0.0,
    cost_per_share: float = 0.01,
    bid_ask_spread_bps: float = 5.0,
    slippage_bps: float = 5.0,
    impact_factor: float = 0.1,
) -> BacktestResult:
    """Backtest a delta-hedged option position with discrete rebalancing and costs."""
    if len(spot_prices) < 2:
        raise ValueError("spot_prices must have at least 2 values")
    if np.isnan(spot_prices).any():
        raise ValueError("spot_prices must not contain NaN values")

    num_periods = len(spot_prices)
    dt = time_to_expiry / num_periods
    rebalance_every = rebalance_periods(rebalance_frequency)

    stock_position = 0.0
    cash = 0.0
    total_transaction_costs = 0.0
    total_slippage_costs = 0.0
    num_trades = 0
    equity_curve: list[float] = []
    cost_kwargs = {
        "cost_per_share": cost_per_share,
        "bid_ask_spread_bps": bid_ask_spread_bps,
        "slippage_bps": slippage_bps,
        "impact_factor": impact_factor,
    }

    initial_spot = float(spot_prices[0])
    initial_option_price = _option_mark_price(
        initial_spot, strike, rate, vol, time_to_expiry, option_type, dividend_yield
    )

    initial_delta = as_float(
        black_scholes_delta(
            spot=initial_spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )
    )
    stock_position, cash, cost, slippage, trades = _trade_stock(
        -option_position * initial_delta - stock_position,
        initial_spot,
        stock_position,
        cash,
        **cost_kwargs,
    )
    total_transaction_costs += cost
    total_slippage_costs += slippage
    num_trades += trades

    initial_value = _portfolio_value(
        initial_spot, option_position, initial_option_price, stock_position, cash
    )
    equity_curve.append(initial_value)
    remaining_time = time_to_expiry

    for i in range(1, num_periods):
        current_spot = float(spot_prices[i])
        remaining_time = time_to_expiry - i * dt
        if remaining_time <= 0:
            break

        if i % rebalance_every == 0:
            current_delta = as_float(
                black_scholes_delta(
                    spot=current_spot,
                    strike=strike,
                    rate=rate,
                    vol=vol,
                    time_to_expiry=remaining_time,
                    option_type=option_type,
                    dividend_yield=dividend_yield,
                )
            )
            stock_position, cash, cost, slippage, trades = _trade_stock(
                -option_position * current_delta - stock_position,
                current_spot,
                stock_position,
                cash,
                **cost_kwargs,
            )
            total_transaction_costs += cost
            total_slippage_costs += slippage
            num_trades += trades

        option_price = _option_mark_price(
            current_spot, strike, rate, vol, remaining_time, option_type, dividend_yield
        )
        equity_curve.append(
            _portfolio_value(current_spot, option_position, option_price, stock_position, cash)
        )

    final_spot = float(spot_prices[-1])
    expiry = remaining_time <= dt
    final_option_price = _option_mark_price(
        final_spot,
        strike,
        rate,
        vol,
        0.0 if expiry else remaining_time,
        option_type,
        dividend_yield,
    )

    final_value = (
        option_position * (final_option_price - initial_option_price)
        + stock_position * final_spot
        + cash
    )
    if equity_curve:
        equity_curve[-1] = final_value

    return BacktestResult(
        total_pnl=final_value - initial_value,
        transaction_costs=total_transaction_costs,
        slippage_costs=total_slippage_costs,
        num_trades=num_trades,
        initial_value=initial_value,
        final_value=final_value,
        returns=0.0,
        strategy_type="delta_hedge",
        equity_curve=np.array(equity_curve, dtype=np.float64),
    )
