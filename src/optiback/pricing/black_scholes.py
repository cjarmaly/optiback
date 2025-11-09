"""Black-Scholes option pricing model implementation."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_scholes_call(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate the price of a European call option using the Black-Scholes formula.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)

    Returns:
        The theoretical price of the call option

    Examples:
        >>> price = black_scholes_call(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5)
        >>> round(price, 4)
        6.8887
    """
    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Handle edge cases
    if time_to_expiry <= 0:
        result = np.maximum(spot_arr - strike_arr, 0.0)
        if is_scalar:
            return float(result.item())
        return result
    if vol <= 0:
        result = np.maximum(
            spot_arr * np.exp(-dividend_yield * time_to_expiry)
            - strike_arr * np.exp(-rate * time_to_expiry),
            0.0,
        )
        if is_scalar:
            return float(result.item())
        return result  # type: ignore[no-any-return]

    # Black-Scholes formula
    d1 = (
        np.log(spot_arr / strike_arr) + (rate - dividend_yield + 0.5 * vol**2) * time_to_expiry
    ) / (vol * np.sqrt(time_to_expiry))

    d2 = d1 - vol * np.sqrt(time_to_expiry)

    # Calculate call price
    call_price = spot_arr * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(
        d1
    ) - strike_arr * np.exp(-rate * time_to_expiry) * norm.cdf(d2)

    # Return scalar if inputs were scalar
    if is_scalar:
        return float(call_price.item())
    return call_price  # type: ignore[no-any-return]


def black_scholes_put(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate the price of a European put option using the Black-Scholes formula.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)

    Returns:
        The theoretical price of the put option

    Examples:
        >>> price = black_scholes_put(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5)
        >>> round(price, 4)
        5.8164
    """
    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    # Use put-call parity: Put = Call - Spot*e^(-qT) + Strike*e^(-rT)
    call_price = black_scholes_call(
        spot=spot,
        strike=strike,
        rate=rate,
        vol=vol,
        time_to_expiry=time_to_expiry,
        dividend_yield=dividend_yield,
    )

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Convert call_price to array for consistent computation
    call_price_arr = np.asarray(call_price, dtype=np.float64)

    put_price = (
        call_price_arr
        - spot_arr * np.exp(-dividend_yield * time_to_expiry)
        + strike_arr * np.exp(-rate * time_to_expiry)
    )

    # Return scalar if inputs were scalar
    if is_scalar:
        return float(put_price.item())
    return put_price  # type: ignore[no-any-return]
