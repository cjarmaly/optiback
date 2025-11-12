"""Black-Scholes Greeks implementation."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _compute_d1_d2(
    spot: np.ndarray,
    strike: np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute d1 and d2 for Black-Scholes Greeks.

    Internal helper function to compute d1 and d2 values used in Greeks calculations.
    """
    if time_to_expiry <= 0:
        # Return zeros to avoid division by zero
        zeros = np.zeros_like(spot)
        return zeros, zeros

    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * vol**2) * time_to_expiry) / (
        vol * np.sqrt(time_to_expiry)
    )

    d2 = d1 - vol * np.sqrt(time_to_expiry)

    return d1, d2


def black_scholes_delta(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate the Delta (Δ) of an option using the Black-Scholes model.

    Delta measures the rate of change of the option price with respect to changes
    in the underlying asset price.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        option_type: Type of option, either 'call' or 'put'
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)

    Returns:
        The Delta of the option. For calls: 0 to 1. For puts: -1 to 0.

    Examples:
        >>> delta = black_scholes_delta(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5, option_type='call')
        >>> round(delta, 4)
        0.5367
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Handle edge cases
    if time_to_expiry <= 0:
        # At expiry, delta is 1 for ITM calls, 0 for OTM calls
        # For puts: -1 for ITM, 0 for OTM
        if option_type == "call":
            result = np.where(spot_arr > strike_arr, 1.0, 0.0)
        else:  # put
            result = np.where(spot_arr < strike_arr, -1.0, 0.0)
        if is_scalar:
            return float(result.item())
        return result

    if vol <= 0:
        # With zero vol, delta depends on moneyness
        if option_type == "call":
            result = np.where(
                spot_arr * np.exp(-dividend_yield * time_to_expiry)
                > strike_arr * np.exp(-rate * time_to_expiry),
                1.0,
                0.0,
            )
        else:  # put
            result = np.where(
                spot_arr * np.exp(-dividend_yield * time_to_expiry)
                < strike_arr * np.exp(-rate * time_to_expiry),
                -1.0,
                0.0,
            )
        if is_scalar:
            return float(result.item())
        return result

    # Compute d1 and d2
    d1, _ = _compute_d1_d2(spot_arr, strike_arr, rate, vol, time_to_expiry, dividend_yield)

    # Delta formula
    # Call: Delta = e^(-qT) * N(d1)
    # Put: Delta = e^(-qT) * (N(d1) - 1) = e^(-qT) * N(d1) - e^(-qT)
    discount_factor = np.exp(-dividend_yield * time_to_expiry)
    n_d1 = norm.cdf(d1)

    if option_type == "call":
        delta = discount_factor * n_d1
    else:  # put
        delta = discount_factor * (n_d1 - 1.0)

    # Return scalar if inputs were scalar
    if is_scalar:
        return float(delta.item())
    return delta  # type: ignore[no-any-return]


def black_scholes_gamma(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate the Gamma (Γ) of an option using the Black-Scholes model.

    Gamma measures the rate of change of Delta with respect to changes in the
    underlying asset price. Gamma is the same for calls and puts.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)

    Returns:
        The Gamma of the option (always positive).

    Examples:
        >>> gamma = black_scholes_gamma(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5)
        >>> round(gamma, 4)
        0.0187
    """
    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Handle edge cases
    if time_to_expiry <= 0 or vol <= 0:
        # At expiry or with zero vol, gamma is 0 (delta is step function)
        result = np.zeros_like(spot_arr)
        if is_scalar:
            return float(result.item())
        return result

    # Compute d1
    d1, _ = _compute_d1_d2(spot_arr, strike_arr, rate, vol, time_to_expiry, dividend_yield)

    # Gamma formula: Gamma = e^(-qT) * N'(d1) / (S * σ * √T)
    # where N'(d1) is the PDF of standard normal distribution
    discount_factor = np.exp(-dividend_yield * time_to_expiry)
    n_prime_d1 = norm.pdf(d1)
    sqrt_t = np.sqrt(time_to_expiry)

    gamma = (discount_factor * n_prime_d1) / (spot_arr * vol * sqrt_t)

    # Return scalar if inputs were scalar
    if is_scalar:
        return float(gamma.item())
    return gamma  # type: ignore[no-any-return]


def black_scholes_vega(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate the Vega (ν) of an option using the Black-Scholes model.

    Vega measures the rate of change of the option price with respect to changes
    in volatility. Vega is the same for calls and puts.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)

    Returns:
        The Vega of the option per 1% change in volatility (divided by 100).

    Examples:
        >>> vega = black_scholes_vega(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5)
        >>> round(vega, 4)
        0.1872
    """
    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Handle edge cases
    if time_to_expiry <= 0:
        # At expiry, vega is 0 (no time value)
        result = np.zeros_like(spot_arr)
        if is_scalar:
            return float(result.item())
        return result

    if vol <= 0:
        # With zero vol, vega is 0
        result = np.zeros_like(spot_arr)
        if is_scalar:
            return float(result.item())
        return result

    # Compute d1
    d1, _ = _compute_d1_d2(spot_arr, strike_arr, rate, vol, time_to_expiry, dividend_yield)

    # Vega formula: Vega = S * e^(-qT) * N'(d1) * √T
    # Often reported per 1% change in vol, so divide by 100
    discount_factor = np.exp(-dividend_yield * time_to_expiry)
    n_prime_d1 = norm.pdf(d1)
    sqrt_t = np.sqrt(time_to_expiry)

    vega = spot_arr * discount_factor * n_prime_d1 * sqrt_t / 100.0

    # Return scalar if inputs were scalar
    if is_scalar:
        return float(vega.item())
    return vega  # type: ignore[no-any-return]


def black_scholes_theta(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate the Theta (Θ) of an option using the Black-Scholes model.

    Theta measures the rate of change of the option price with respect to time.
    Theta is typically negative (options lose value as time passes).

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        option_type: Type of option, either 'call' or 'put'
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)

    Returns:
        The Theta of the option per day (negative value, time decay).

    Examples:
        >>> theta = black_scholes_theta(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5, option_type='call')
        >>> round(theta, 4)
        -0.0145
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Handle edge cases
    if time_to_expiry <= 0:
        # At expiry, theta is 0 (no more time decay)
        result = np.zeros_like(spot_arr)
        if is_scalar:
            return float(result.item())
        return result

    if vol <= 0:
        # With zero vol, theta is 0 (no time value decay)
        result = np.zeros_like(spot_arr)
        if is_scalar:
            return float(result.item())
        return result

    # Compute d1 and d2
    d1, d2 = _compute_d1_d2(spot_arr, strike_arr, rate, vol, time_to_expiry, dividend_yield)

    # Theta formula components
    discount_factor_spot = np.exp(-dividend_yield * time_to_expiry)
    discount_factor_strike = np.exp(-rate * time_to_expiry)
    n_prime_d1 = norm.pdf(d1)
    sqrt_t = np.sqrt(time_to_expiry)

    # Common term: -S * e^(-qT) * N'(d1) * σ / (2√T)
    common_term = -spot_arr * discount_factor_spot * n_prime_d1 * vol / (2.0 * sqrt_t)

    # Divide by 365 to get per-day theta
    days_per_year = 365.0

    if option_type == "call":
        # Theta_call = common_term - r*K*e^(-rT)*N(d2) + q*S*e^(-qT)*N(d1)
        n_d2 = norm.cdf(d2)
        n_d1 = norm.cdf(d1)
        theta = (
            common_term
            - rate * strike_arr * discount_factor_strike * n_d2
            + dividend_yield * spot_arr * discount_factor_spot * n_d1
        ) / days_per_year
    else:  # put
        # Theta_put = common_term + r*K*e^(-rT)*N(-d2) - q*S*e^(-qT)*N(-d1)
        n_neg_d2 = norm.cdf(-d2)
        n_neg_d1 = norm.cdf(-d1)
        theta = (
            common_term
            + rate * strike_arr * discount_factor_strike * n_neg_d2
            - dividend_yield * spot_arr * discount_factor_spot * n_neg_d1
        ) / days_per_year

    # Return scalar if inputs were scalar
    if is_scalar:
        return float(theta.item())
    return theta  # type: ignore[no-any-return]


def black_scholes_rho(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate the Rho (ρ) of an option using the Black-Scholes model.

    Rho measures the rate of change of the option price with respect to changes in
    the risk-free interest rate.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        option_type: Type of option, either 'call' or 'put'
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)

    Returns:
        The Rho of the option per 1% change in interest rate (divided by 100).

    Examples:
        >>> rho = black_scholes_rho(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5, option_type='call')
        >>> round(rho, 4)
        0.0208
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Check if inputs are scalar before converting to arrays
    is_scalar = np.isscalar(spot) and np.isscalar(strike)

    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    # Handle edge cases
    if time_to_expiry <= 0:
        # At expiry, rho is 0 (no time value)
        result = np.zeros_like(spot_arr)
        if is_scalar:
            return float(result.item())
        return result

    if vol <= 0:
        # With zero vol, rho is 0
        result = np.zeros_like(spot_arr)
        if is_scalar:
            return float(result.item())
        return result

    # Compute d1 and d2
    d1, d2 = _compute_d1_d2(spot_arr, strike_arr, rate, vol, time_to_expiry, dividend_yield)

    # Rho formula
    discount_factor = np.exp(-rate * time_to_expiry)

    if option_type == "call":
        # Rho_call = K * T * e^(-rT) * N(d2) / 100
        n_d2 = norm.cdf(d2)
        rho = strike_arr * time_to_expiry * discount_factor * n_d2 / 100.0
    else:  # put
        # Rho_put = -K * T * e^(-rT) * N(-d2) / 100
        n_neg_d2 = norm.cdf(-d2)
        rho = -strike_arr * time_to_expiry * discount_factor * n_neg_d2 / 100.0

    # Return scalar if inputs were scalar
    if is_scalar:
        return float(rho.item())
    return rho  # type: ignore[no-any-return]


def black_scholes_greeks(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> dict[str, float | np.ndarray]:
    """
    Calculate all Greeks (Delta, Gamma, Vega, Theta, Rho) for an option.

    This is a convenience function that calculates all Greeks at once.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        rate: Risk-free interest rate (annualized, continuously compounded)
        vol: Volatility of the underlying asset (annualized)
        time_to_expiry: Time to expiration in years
        option_type: Type of option, either 'call' or 'put'
        dividend_yield: Dividend yield of the underlying asset (annualized, default: 0.0)

    Returns:
        Dictionary containing all Greeks: 'delta', 'gamma', 'vega', 'theta', 'rho'

    Examples:
        >>> greeks = black_scholes_greeks(spot=100.0, strike=100.0, rate=0.02, vol=0.25, time_to_expiry=0.5, option_type='call')
        >>> round(greeks['delta'], 4)
        0.5367
    """
    return {
        "delta": black_scholes_delta(
            spot, strike, rate, vol, time_to_expiry, option_type, dividend_yield
        ),
        "gamma": black_scholes_gamma(spot, strike, rate, vol, time_to_expiry, dividend_yield),
        "vega": black_scholes_vega(spot, strike, rate, vol, time_to_expiry, dividend_yield),
        "theta": black_scholes_theta(
            spot, strike, rate, vol, time_to_expiry, option_type, dividend_yield
        ),
        "rho": black_scholes_rho(
            spot, strike, rate, vol, time_to_expiry, option_type, dividend_yield
        ),
    }
