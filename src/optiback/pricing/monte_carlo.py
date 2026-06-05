"""Monte Carlo option pricing for European options."""

from __future__ import annotations

import numpy as np

from optiback.pricing.array import map_over_spots


def _monte_carlo_scalar(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float,
    simulations: int,
    seed: int | None,
    *,
    is_call: bool,
) -> float:
    rng = np.random.default_rng(seed)
    n_pairs = simulations // 2
    z = rng.standard_normal(n_pairs)

    drift = (rate - dividend_yield - 0.5 * vol**2) * time_to_expiry
    diffusion = vol * np.sqrt(time_to_expiry)
    stock_up = spot * np.exp(drift + diffusion * z)
    stock_down = spot * np.exp(drift - diffusion * z)

    if is_call:
        payoff_up = np.maximum(stock_up - strike, 0.0)
        payoff_down = np.maximum(stock_down - strike, 0.0)
    else:
        payoff_up = np.maximum(strike - stock_up, 0.0)
        payoff_down = np.maximum(strike - stock_down, 0.0)

    avg_payoff = (np.mean(payoff_up) + np.mean(payoff_down)) / 2.0
    if simulations % 2 == 1:
        z_extra = rng.standard_normal(1)[0]
        stock_extra = spot * np.exp(drift + diffusion * z_extra)
        payoff_extra = max(stock_extra - strike, 0.0) if is_call else max(strike - stock_extra, 0.0)
        avg_payoff = (n_pairs * 2 * avg_payoff + payoff_extra) / simulations

    return float(np.exp(-rate * time_to_expiry) * avg_payoff)


def _price_monte_carlo(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float,
    vol: float,
    time_to_expiry: float,
    dividend_yield: float,
    simulations: int,
    seed: int | None,
    *,
    is_call: bool,
) -> float | np.ndarray:
    if simulations <= 0:
        raise ValueError(f"simulations must be greater than 0, got {simulations}")

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

    scalar_fn = lambda s, k: _monte_carlo_scalar(  # noqa: E731
        s, k, rate, vol, time_to_expiry, dividend_yield, simulations, seed, is_call=is_call
    )
    return map_over_spots(scalar_fn, spot, strike)


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
    """Price a European call option using Monte Carlo simulation with antithetic variates."""
    return _price_monte_carlo(
        spot, strike, rate, vol, time_to_expiry, dividend_yield, simulations, seed, is_call=True
    )


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
    """Price a European put option using Monte Carlo simulation with antithetic variates."""
    return _price_monte_carlo(
        spot, strike, rate, vol, time_to_expiry, dividend_yield, simulations, seed, is_call=False
    )
