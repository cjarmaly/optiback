"""Black-Scholes implied volatility calculation."""

from __future__ import annotations

import numpy as np

from optiback.pricing.black_scholes import black_scholes_call, black_scholes_put
from optiback.pricing.greeks import black_scholes_vega


def black_scholes_implied_volatility(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    time_to_expiry: float,
    market_price: float | np.ndarray,
    option_type: str,
    dividend_yield: float = 0.0,
    initial_guess: float = 0.2,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float | np.ndarray:
    """
    Calculate implied volatility using Newton-Raphson method.

    Finds the volatility that, when plugged into Black-Scholes formula,
    produces the given market price.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        time_to_expiry: Time to expiration in years
        market_price: Observed market price of the option
        option_type: Type of option, either 'call' or 'put'
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)
        initial_guess: Initial volatility guess for Newton-Raphson (default: 0.2 = 20%)
        max_iterations: Maximum number of iterations (default: 100)
        tolerance: Convergence tolerance (default: 1e-6)

    Returns:
        The implied volatility that produces the market price

    Raises:
        ValueError: If market price is below intrinsic value or no solution found
        RuntimeError: If Newton-Raphson fails to converge

    Examples:
        >>> iv = black_scholes_implied_volatility(
        ...     spot=100.0, strike=100.0, rate=0.02, time_to_expiry=0.5,
        ...     market_price=7.5168, option_type='call'
        ... )
        >>> round(iv, 4)
        0.25
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike) and np.isscalar(market_price)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)
    market_price_arr = np.asarray(market_price, dtype=np.float64)

    # Handle edge cases
    if time_to_expiry <= 0:
        # At expiry, implied vol is undefined (price equals intrinsic value)
        raise ValueError("Implied volatility is undefined at expiry (time_to_expiry must be > 0)")

    # Calculate intrinsic value to validate market price
    if option_type == "call":
        intrinsic = np.maximum(
            spot_arr * np.exp(-dividend_yield * time_to_expiry)
            - strike_arr * np.exp(-rate * time_to_expiry),
            0.0,
        )
    else:  # put
        intrinsic = np.maximum(
            strike_arr * np.exp(-rate * time_to_expiry)
            - spot_arr * np.exp(-dividend_yield * time_to_expiry),
            0.0,
        )

    # Check if market price is below intrinsic value
    if np.any(market_price_arr < intrinsic):
        if is_scalar:
            raise ValueError(
                f"Market price ({market_price}) must be >= intrinsic value ({intrinsic.item():.4f})"
            )
        raise ValueError("Market price must be >= intrinsic value for all options")

    # If market price equals intrinsic, implied vol is 0
    price_equals_intrinsic = np.abs(market_price_arr - intrinsic) < tolerance
    if np.all(price_equals_intrinsic):
        if is_scalar:
            return 0.0
        return np.zeros_like(spot_arr)

    # Initialize volatility guess
    vol = np.full_like(spot_arr, initial_guess, dtype=np.float64)

    # Newton-Raphson iteration
    # Handle scalar vs array cases
    if is_scalar:
        # Scalar case: simple iteration
        vol_val = float(vol.item())
        market_price_val = float(market_price_arr.item())

        for iteration in range(max_iterations):
            # Calculate theoretical price with current volatility guess
            if option_type == "call":
                theoretical_price_val = black_scholes_call(
                    spot=spot_arr.item(),
                    strike=strike_arr.item(),
                    rate=rate,
                    vol=vol_val,
                    time_to_expiry=time_to_expiry,
                    dividend_yield=dividend_yield,
                )
            else:  # put
                theoretical_price_val = black_scholes_put(
                    spot=spot_arr.item(),
                    strike=strike_arr.item(),
                    rate=rate,
                    vol=vol_val,
                    time_to_expiry=time_to_expiry,
                    dividend_yield=dividend_yield,
                )

            # Calculate error
            error_val = float(theoretical_price_val - market_price_val)

            # Check convergence
            tolerance_float = float(tolerance)
            if abs(error_val) < tolerance_float:
                return vol_val

            # Calculate Vega
            vega_val = black_scholes_vega(
                spot=spot_arr.item(),
                strike=strike_arr.item(),
                rate=rate,
                vol=vol_val,
                time_to_expiry=time_to_expiry,
                dividend_yield=dividend_yield,
            )

            # Convert vega from per 1% to per decimal
            # vega is per 1% = per 0.01, so vega * 100 = per 1.0
            vega_decimal = float(vega_val * 100.0)

            # Handle zero or very small vega
            if abs(vega_decimal) < 1e-10:
                vega_decimal = 1e-10

            # Newton-Raphson update
            vol_val = float(vol_val - error_val / vega_decimal)

            # Ensure volatility stays positive and within reasonable bounds
            vol_val = max(0.001, min(5.0, vol_val))

    else:
        # Array case: iterate for each element
        for iteration in range(max_iterations):
            # Calculate theoretical price with current volatility guess
            # For arrays, we need to compute each element separately since vol must be scalar
            theoretical_price = np.zeros_like(spot_arr)

            for i in range(len(spot_arr)):
                vol_i = float(vol[i])
                spot_i = float(spot_arr[i])
                strike_i = float(strike_arr[i] if strike_arr.ndim > 0 else strike_arr)

                if option_type == "call":
                    theoretical_price[i] = black_scholes_call(
                        spot=spot_i,
                        strike=strike_i,
                        rate=rate,
                        vol=vol_i,
                        time_to_expiry=time_to_expiry,
                        dividend_yield=dividend_yield,
                    )
                else:  # put
                    theoretical_price[i] = black_scholes_put(
                        spot=spot_i,
                        strike=strike_i,
                        rate=rate,
                        vol=vol_i,
                        time_to_expiry=time_to_expiry,
                        dividend_yield=dividend_yield,
                    )

            # Calculate error
            error = theoretical_price - market_price_arr

            # Check convergence
            max_error = np.max(np.abs(error))
            if max_error < float(tolerance):
                return vol

            # Calculate Vega for each element
            vega = np.zeros_like(spot_arr)

            for i in range(len(spot_arr)):
                vol_i = float(vol[i])
                spot_i = float(spot_arr[i])
                strike_i = float(strike_arr[i] if strike_arr.ndim > 0 else strike_arr)

                vega[i] = black_scholes_vega(
                    spot=spot_i,
                    strike=strike_i,
                    rate=rate,
                    vol=vol_i,
                    time_to_expiry=time_to_expiry,
                    dividend_yield=dividend_yield,
                )

            # Convert vega from per 1% to per decimal
            vega_decimal_arr = vega * 100.0

            # Handle zero or very small vega
            vega_safe = np.where(np.abs(vega_decimal_arr) < 1e-10, 1e-10, vega_decimal_arr)

            # Newton-Raphson update
            vol_update = error / vega_safe
            vol = vol - vol_update

            # Ensure volatility stays positive and within reasonable bounds
            vol = np.clip(vol, 0.001, 5.0)

            # For options at intrinsic, set vol to 0
            vol = np.where(price_equals_intrinsic, 0.0, vol)

        # Final check for array case
        theoretical_price = np.zeros_like(spot_arr)
        for i in range(len(spot_arr)):
            vol_i = float(vol[i])
            spot_i = float(spot_arr[i])
            strike_i = float(strike_arr[i] if strike_arr.ndim > 0 else strike_arr)

            if option_type == "call":
                theoretical_price[i] = black_scholes_call(
                    spot=spot_i,
                    strike=strike_i,
                    rate=rate,
                    vol=vol_i,
                    time_to_expiry=time_to_expiry,
                    dividend_yield=dividend_yield,
                )
            else:  # put
                theoretical_price[i] = black_scholes_put(
                    spot=spot_i,
                    strike=strike_i,
                    rate=rate,
                    vol=vol_i,
                    time_to_expiry=time_to_expiry,
                    dividend_yield=dividend_yield,
                )

        final_error = np.max(np.abs(theoretical_price - market_price_arr))
        if final_error < float(tolerance) * 100:
            return vol

        raise RuntimeError(
            f"Failed to converge after {max_iterations} iterations. "
            f"Maximum final error: {final_error:.6f}, tolerance: {tolerance}"
        )

    # If we reach here (scalar case), didn't converge
    raise RuntimeError(
        f"Failed to converge after {max_iterations} iterations. "
        f"Final error: {abs(error_val):.6f}, tolerance: {tolerance}"
    )
