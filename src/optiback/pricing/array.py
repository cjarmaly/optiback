"""Shared helpers for scalar and array option pricing inputs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def as_float(value: float | np.ndarray) -> float:
    """Coerce a scalar pricing result to float."""
    return float(value) if not isinstance(value, float) else value


def map_over_spots(
    scalar_fn: Callable[[float, float], float],
    spot: float | np.ndarray,
    strike: float | np.ndarray,
) -> float | np.ndarray:
    """Apply a two-argument scalar pricer across spot/strike arrays."""
    is_scalar = np.isscalar(spot) and np.isscalar(strike)
    spot_arr = np.asarray(spot, dtype=np.float64)
    strike_arr = np.asarray(strike, dtype=np.float64)

    if is_scalar:
        return scalar_fn(float(spot_arr.item()), float(strike_arr.item()))

    spot_flat = spot_arr.ravel()
    strike_is_scalar = strike_arr.ndim == 0 or (strike_arr.size == 1 and np.isscalar(strike))
    if strike_is_scalar:
        strike_value = float(strike_arr.item())
        prices = (scalar_fn(float(s), strike_value) for s in spot_flat)
    else:
        strike_flat = strike_arr.ravel()
        prices = (
            scalar_fn(float(spot_flat[i]), float(strike_flat[i])) for i in range(spot_flat.size)
        )

    result = np.fromiter(prices, dtype=np.float64, count=spot_flat.size)
    return result.reshape(spot_arr.shape)
