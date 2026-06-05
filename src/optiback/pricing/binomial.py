"""Binomial Tree option pricing for American options (CRR)."""

from __future__ import annotations

import numpy as np

from optiback.pricing.array import map_over_spots


def _binomial_tree_scalar(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float,
    steps: int,
    *,
    is_call: bool,
) -> float:
    dt = time_to_expiry / steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    disc_factor = np.exp(-rate * dt)
    p = (np.exp((rate - dividend_yield) * dt) - d) / (u - d)

    stock_prices = np.array([spot * (u ** (steps - j)) * (d**j) for j in range(steps + 1)])
    if is_call:
        option_values = np.maximum(stock_prices - strike, 0.0)
    else:
        option_values = np.maximum(strike - stock_prices, 0.0)

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            stock_price = spot * (u ** (i - j)) * (d**j)
            continuation = disc_factor * (p * option_values[j] + (1 - p) * option_values[j + 1])
            exercise = max(stock_price - strike, 0.0) if is_call else max(strike - stock_price, 0.0)
            option_values[j] = max(continuation, exercise)

    return float(option_values[0])


def _price_binomial(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float,
    steps: int,
    *,
    is_call: bool,
) -> float | np.ndarray:
    if steps <= 0:
        raise ValueError(f"steps must be greater than 0, got {steps}")

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    if time_to_expiry <= 0:
        intrinsic = (
            np.maximum(spot_arr - strike_arr, 0.0)
            if is_call
            else np.maximum(strike_arr - spot_arr, 0.0)
        )
        return float(intrinsic.item()) if is_scalar else intrinsic

    if vol <= 0:
        if is_call:
            intrinsic = np.maximum(
                spot_arr * np.exp(-dividend_yield * time_to_expiry)
                - strike_arr * np.exp(-rate * time_to_expiry),
                0.0,
            )
        else:
            intrinsic = np.maximum(
                strike_arr * np.exp(-rate * time_to_expiry)
                - spot_arr * np.exp(-dividend_yield * time_to_expiry),
                0.0,
            )
        return float(intrinsic.item()) if is_scalar else intrinsic

    scalar_fn = lambda s, k: _binomial_tree_scalar(  # noqa: E731
        s, k, rate, vol, time_to_expiry, dividend_yield, steps, is_call=is_call
    )
    return map_over_spots(scalar_fn, spot, strike)


def binomial_tree_call(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    steps: int = 100,
) -> float | np.ndarray:
    """Price an American call option using the Cox-Ross-Rubinstein binomial tree."""
    return _price_binomial(
        spot, strike, rate, vol, time_to_expiry, dividend_yield, steps, is_call=True
    )


def binomial_tree_put(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    steps: int = 100,
) -> float | np.ndarray:
    """Price an American put option using the Cox-Ross-Rubinstein binomial tree."""
    return _price_binomial(
        spot, strike, rate, vol, time_to_expiry, dividend_yield, steps, is_call=False
    )
