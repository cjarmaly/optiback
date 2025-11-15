"""Binomial Tree option pricing model implementation for American options."""

from __future__ import annotations

import numpy as np


def binomial_tree_call(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    steps: int = 100,
) -> float | np.ndarray:
    """
    Calculate the price of an American call option using the Binomial Tree model (CRR).

    Uses the Cox-Ross-Rubinstein (CRR) model to build a binomial tree and prices
    the option by working backwards from expiration, applying early exercise at
    each node for American options.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)
        steps: Number of time steps in the binomial tree (default: 100)

    Returns:
        The theoretical price of the American call option

    Examples:
        >>> price = binomial_tree_call(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5, steps=100)
        >>> round(price, 4)  # Should be close to Black-Scholes European call for same params
        7.5168
    """
    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Validate steps
    if steps <= 0:
        raise ValueError(f"steps must be greater than 0, got {steps}")

    # Handle edge cases
    if time_to_expiry <= 0:
        result = np.maximum(spot_arr - strike_arr, 0.0)
        if is_scalar:
            return float(result.item())
        return result

    if vol <= 0:
        # Zero volatility: option value is intrinsic value discounted
        intrinsic = np.maximum(
            spot_arr * np.exp(-dividend_yield * time_to_expiry)
            - strike_arr * np.exp(-rate * time_to_expiry),
            0.0,
        )
        if is_scalar:
            return float(intrinsic.item())
        return intrinsic  # type: ignore[no-any-return]

    # Handle scalar vs array inputs
    if is_scalar:
        return _binomial_tree_call_scalar(
            float(spot_arr.item()),
            float(strike_arr.item()),
            rate,
            vol,
            time_to_expiry,
            dividend_yield,
            steps,
        )

    # Array inputs: iterate over each element
    result = np.zeros_like(spot_arr, dtype=np.float64)
    spot_flat = spot_arr.flatten()
    strike_is_scalar = strike_arr.ndim == 0
    strike_scalar_value = float(strike_arr.item()) if strike_is_scalar else None
    if not strike_is_scalar:
        strike_flat = strike_arr.flatten()
    result_flat = result.flatten()

    for i in range(len(spot_flat)):
        if strike_is_scalar:
            strike_val: float = strike_scalar_value  # type: ignore[assignment]
        else:
            strike_val = float(strike_flat[i])
        result_flat[i] = _binomial_tree_call_scalar(
            float(spot_flat[i]),
            strike_val,
            rate,
            vol,
            time_to_expiry,
            dividend_yield,
            steps,
        )

    # Copy back from flat view to result (flatten() creates a view, so modifications are preserved)
    result[:] = result_flat.reshape(spot_arr.shape)
    return result


def _binomial_tree_call_scalar(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float,
    steps: int,
) -> float:
    """
    Internal helper function to price a single American call option using binomial tree.

    Uses CRR model:
    - u = exp(vol * sqrt(dt))
    - d = 1/u
    - p = (exp((r-q)*dt) - d) / (u - d)
    """
    dt = time_to_expiry / steps
    u = np.exp(vol * np.sqrt(dt))  # Up factor
    d = 1.0 / u  # Down factor
    disc_factor = np.exp(-rate * dt)  # Discount factor per step
    growth_factor = np.exp((rate - dividend_yield) * dt)  # Growth factor with dividend

    # Risk-neutral probability
    p = (growth_factor - d) / (u - d)

    # Build stock price tree (forward)
    stock_prices = np.zeros(steps + 1)
    for j in range(steps + 1):
        stock_prices[j] = spot * (u ** (steps - j)) * (d**j)

    # Initialize option values at expiration
    option_values = np.maximum(stock_prices - strike, 0.0)

    # Work backwards through the tree, applying early exercise
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            # Stock price at this node
            stock_price = spot * (u ** (i - j)) * (d**j)

            # Option value from continuation (risk-neutral expectation)
            continuation_value = disc_factor * (
                p * option_values[j] + (1 - p) * option_values[j + 1]
            )

            # Exercise value (intrinsic value)
            exercise_value = max(stock_price - strike, 0.0)

            # American option: take maximum of exercise and continuation
            option_values[j] = max(continuation_value, exercise_value)

    return float(option_values[0])


def binomial_tree_put(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    steps: int = 100,
) -> float | np.ndarray:
    """
    Calculate the price of an American put option using the Binomial Tree model (CRR).

    Uses the Cox-Ross-Rubinstein (CRR) model to build a binomial tree and prices
    the option by working backwards from expiration, applying early exercise at
    each node for American options.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)
        steps: Number of time steps in the binomial tree (default: 100)

    Returns:
        The theoretical price of the American put option

    Examples:
        >>> price = binomial_tree_put(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5, steps=100)
        >>> round(price, 4)  # Should be close to Black-Scholes European put for same params
        6.5218
    """
    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Validate steps
    if steps <= 0:
        raise ValueError(f"steps must be greater than 0, got {steps}")

    # Handle edge cases
    if time_to_expiry <= 0:
        result = np.maximum(strike_arr - spot_arr, 0.0)
        if is_scalar:
            return float(result.item())
        return result

    if vol <= 0:
        # Zero volatility: option value is intrinsic value discounted
        intrinsic = np.maximum(
            strike_arr * np.exp(-rate * time_to_expiry)
            - spot_arr * np.exp(-dividend_yield * time_to_expiry),
            0.0,
        )
        if is_scalar:
            return float(intrinsic.item())
        return intrinsic  # type: ignore[no-any-return]

    # Handle scalar vs array inputs
    if is_scalar:
        return _binomial_tree_put_scalar(
            float(spot_arr.item()),
            float(strike_arr.item()),
            rate,
            vol,
            time_to_expiry,
            dividend_yield,
            steps,
        )

    # Array inputs: iterate over each element
    result = np.zeros_like(spot_arr, dtype=np.float64)
    spot_flat = spot_arr.flatten()
    strike_is_scalar = strike_arr.ndim == 0
    strike_scalar_value = float(strike_arr.item()) if strike_is_scalar else None
    if not strike_is_scalar:
        strike_flat = strike_arr.flatten()
    result_flat = result.flatten()

    for i in range(len(spot_flat)):
        if strike_is_scalar:
            strike_val: float = strike_scalar_value  # type: ignore[assignment]
        else:
            strike_val = float(strike_flat[i])
        result_flat[i] = _binomial_tree_put_scalar(
            float(spot_flat[i]),
            strike_val,
            rate,
            vol,
            time_to_expiry,
            dividend_yield,
            steps,
        )

    # Copy back from flat view to result (flatten() creates a view, so modifications are preserved)
    result[:] = result_flat.reshape(spot_arr.shape)
    return result


def _binomial_tree_put_scalar(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float,
    steps: int,
) -> float:
    """
    Internal helper function to price a single American put option using binomial tree.

    Uses CRR model:
    - u = exp(vol * sqrt(dt))
    - d = 1/u
    - p = (exp((r-q)*dt) - d) / (u - d)
    """
    dt = time_to_expiry / steps
    u = np.exp(vol * np.sqrt(dt))  # Up factor
    d = 1.0 / u  # Down factor
    disc_factor = np.exp(-rate * dt)  # Discount factor per step
    growth_factor = np.exp((rate - dividend_yield) * dt)  # Growth factor with dividend

    # Risk-neutral probability
    p = (growth_factor - d) / (u - d)

    # Build stock price tree (forward)
    stock_prices = np.zeros(steps + 1)
    for j in range(steps + 1):
        stock_prices[j] = spot * (u ** (steps - j)) * (d**j)

    # Initialize option values at expiration
    option_values = np.maximum(strike - stock_prices, 0.0)

    # Work backwards through the tree, applying early exercise
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            # Stock price at this node
            stock_price = spot * (u ** (i - j)) * (d**j)

            # Option value from continuation (risk-neutral expectation)
            continuation_value = disc_factor * (
                p * option_values[j] + (1 - p) * option_values[j + 1]
            )

            # Exercise value (intrinsic value)
            exercise_value = max(strike - stock_price, 0.0)

            # American option: take maximum of exercise and continuation
            option_values[j] = max(continuation_value, exercise_value)

    return float(option_values[0])
