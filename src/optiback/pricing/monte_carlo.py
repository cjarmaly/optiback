"""Monte Carlo option pricing model implementation for European options."""

from __future__ import annotations

import numpy as np


def monte_carlo_call(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    simulations: int = 100000,
    seed: int | None = None,
) -> float | np.ndarray:
    """
    Calculate the price of a European call option using Monte Carlo simulation.

    Uses geometric Brownian motion to simulate stock price paths and estimates
    the option price by averaging discounted payoffs. Implements antithetic
    variates for variance reduction.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)
        simulations: Number of Monte Carlo simulations (default: 100000)
        seed: Random seed for reproducibility (default: None)

    Returns:
        The estimated price of the European call option

    Examples:
        >>> price = monte_carlo_call(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5, simulations=100000)
        >>> round(price, 2)  # Should be close to Black-Scholes (around 7.52)
        7.52
    """
    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Validate simulations
    if simulations <= 0:
        raise ValueError(f"simulations must be greater than 0, got {simulations}")

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
        return _monte_carlo_call_scalar(
            float(spot_arr.item()),
            float(strike_arr.item()),
            rate,
            vol,
            time_to_expiry,
            dividend_yield,
            simulations,
            seed,
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
        result_flat[i] = _monte_carlo_call_scalar(
            float(spot_flat[i]),
            strike_val,
            rate,
            vol,
            time_to_expiry,
            dividend_yield,
            simulations,
            seed,
        )

    # Copy back from flat view to result
    result[:] = result_flat.reshape(spot_arr.shape)
    return result


def _monte_carlo_call_scalar(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float,
    simulations: int,
    seed: int | None,
) -> float:
    """
    Internal helper function to price a single European call option using Monte Carlo.

    Uses geometric Brownian motion:
    S_T = S_0 * exp((r - q - σ²/2)*T + σ*√T*Z)
    where Z ~ N(0,1)
    """
    # Set up random number generator
    rng = np.random.default_rng(seed)

    # Generate random normal variates
    # Use half simulations with antithetic variates (Z and -Z)
    n_pairs = simulations // 2
    z = rng.standard_normal(n_pairs)

    # Drift and diffusion terms
    drift = (rate - dividend_yield - 0.5 * vol**2) * time_to_expiry
    diffusion = vol * np.sqrt(time_to_expiry)

    # Simulate stock prices at expiry (with antithetic variates)
    # For Z: S_T = S_0 * exp(drift + diffusion * Z)
    # For -Z: S_T = S_0 * exp(drift + diffusion * (-Z))
    stock_price_positive = spot * np.exp(drift + diffusion * z)
    stock_price_negative = spot * np.exp(drift - diffusion * z)

    # Calculate payoffs
    payoff_positive = np.maximum(stock_price_positive - strike, 0.0)
    payoff_negative = np.maximum(stock_price_negative - strike, 0.0)

    # Average payoffs (antithetic variates)
    avg_payoff = (np.mean(payoff_positive) + np.mean(payoff_negative)) / 2.0

    # If simulations is odd, add one more simulation
    if simulations % 2 == 1:
        z_extra = rng.standard_normal(1)[0]
        stock_price_extra = spot * np.exp(drift + diffusion * z_extra)
        payoff_extra = max(stock_price_extra - strike, 0.0)
        # Weighted average: (n_pairs * avg_payoff + payoff_extra) / (n_pairs + 1)
        avg_payoff = (n_pairs * 2 * avg_payoff + payoff_extra) / (simulations)

    # Discount to present value
    discount_factor = np.exp(-rate * time_to_expiry)
    return float(discount_factor * avg_payoff)


def monte_carlo_put(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
    simulations: int = 100000,
    seed: int | None = None,
) -> float | np.ndarray:
    """
    Calculate the price of a European put option using Monte Carlo simulation.

    Uses geometric Brownian motion to simulate stock price paths and estimates
    the option price by averaging discounted payoffs. Implements antithetic
    variates for variance reduction.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)
        simulations: Number of Monte Carlo simulations (default: 100000)
        seed: Random seed for reproducibility (default: None)

    Returns:
        The estimated price of the European put option

    Examples:
        >>> price = monte_carlo_put(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5, simulations=100000)
        >>> round(price, 2)  # Should be close to Black-Scholes (around 6.52)
        6.52
    """
    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Validate simulations
    if simulations <= 0:
        raise ValueError(f"simulations must be greater than 0, got {simulations}")

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
        return _monte_carlo_put_scalar(
            float(spot_arr.item()),
            float(strike_arr.item()),
            rate,
            vol,
            time_to_expiry,
            dividend_yield,
            simulations,
            seed,
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
        result_flat[i] = _monte_carlo_put_scalar(
            float(spot_flat[i]),
            strike_val,
            rate,
            vol,
            time_to_expiry,
            dividend_yield,
            simulations,
            seed,
        )

    # Copy back from flat view to result
    result[:] = result_flat.reshape(spot_arr.shape)
    return result


def _monte_carlo_put_scalar(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float,
    simulations: int,
    seed: int | None,
) -> float:
    """
    Internal helper function to price a single European put option using Monte Carlo.

    Uses geometric Brownian motion:
    S_T = S_0 * exp((r - q - σ²/2)*T + σ*√T*Z)
    where Z ~ N(0,1)
    """
    # Set up random number generator
    rng = np.random.default_rng(seed)

    # Generate random normal variates
    # Use half simulations with antithetic variates (Z and -Z)
    n_pairs = simulations // 2
    z = rng.standard_normal(n_pairs)

    # Drift and diffusion terms
    drift = (rate - dividend_yield - 0.5 * vol**2) * time_to_expiry
    diffusion = vol * np.sqrt(time_to_expiry)

    # Simulate stock prices at expiry (with antithetic variates)
    stock_price_positive = spot * np.exp(drift + diffusion * z)
    stock_price_negative = spot * np.exp(drift - diffusion * z)

    # Calculate payoffs for put
    payoff_positive = np.maximum(strike - stock_price_positive, 0.0)
    payoff_negative = np.maximum(strike - stock_price_negative, 0.0)

    # Average payoffs (antithetic variates)
    avg_payoff = (np.mean(payoff_positive) + np.mean(payoff_negative)) / 2.0

    # If simulations is odd, add one more simulation
    if simulations % 2 == 1:
        z_extra = rng.standard_normal(1)[0]
        stock_price_extra = spot * np.exp(drift + diffusion * z_extra)
        payoff_extra = max(strike - stock_price_extra, 0.0)
        # Weighted average: (n_pairs * avg_payoff + payoff_extra) / (n_pairs + 1)
        avg_payoff = (n_pairs * 2 * avg_payoff + payoff_extra) / (simulations)

    # Discount to present value
    discount_factor = np.exp(-rate * time_to_expiry)
    return float(discount_factor * avg_payoff)

