"""Tests for Monte Carlo option pricing."""

import numpy as np
import pytest

from optiback.pricing.black_scholes import black_scholes_call, black_scholes_put
from optiback.pricing.monte_carlo import monte_carlo_call, monte_carlo_put


class TestMonteCarloCall:
    """Test cases for monte_carlo_call function."""

    def test_basic_call_price(self):
        """Test basic call option pricing."""
        price = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=100000,
            seed=42,
        )
        # Should be close to Black-Scholes for same parameters
        bs_price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )
        # Allow reasonable Monte Carlo error (within 2% or 0.15 absolute)
        error = abs(price - bs_price)
        assert error < max(0.15, bs_price * 0.02)
        assert price > 0
        assert isinstance(price, float)

    def test_call_at_the_money(self):
        """Test at-the-money call option."""
        price = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            simulations=100000,
            seed=42,
        )
        assert price > 0
        assert isinstance(price, float)

    def test_call_in_the_money(self):
        """Test in-the-money call option."""
        price = monte_carlo_call(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            simulations=100000,
            seed=42,
        )
        # ITM call should be worth at least intrinsic value
        assert price >= 10.0  # Intrinsic value = 110 - 100 = 10

    def test_call_out_of_the_money(self):
        """Test out-of-the-money call option."""
        price = monte_carlo_call(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            simulations=100000,
            seed=42,
        )
        # OTM call should have positive value due to time value
        assert price > 0
        assert price < 10.0  # Less than intrinsic value if ITM

    def test_call_zero_time_to_expiry(self):
        """Test call option with zero time to expiry (intrinsic value)."""
        price = monte_carlo_call(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
            simulations=100000,
            seed=42,
        )
        # At expiry, value equals intrinsic value
        assert price == 10.0

    def test_call_zero_volatility(self):
        """Test call option with zero volatility."""
        price = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.0,
            time_to_expiry=1.0,
            simulations=100000,
            seed=42,
        )
        # Zero vol should give discounted intrinsic value
        assert price >= 0.0

    def test_call_array_inputs(self):
        """Test call pricing with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        prices = monte_carlo_call(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            simulations=100000,
            seed=42,
        )

        # Should return array
        assert isinstance(prices, np.ndarray)
        assert len(prices) == 3
        # OTM < ATM < ITM
        assert prices[0] < prices[1] < prices[2]

    def test_call_scalar_strike_with_array_spot(self):
        """Test call pricing with scalar strike and array spot."""
        spots = np.array([90.0, 100.0, 110.0])
        strike = 100.0

        prices = monte_carlo_call(
            spot=spots,
            strike=strike,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            simulations=100000,
            seed=42,
        )

        assert isinstance(prices, np.ndarray)
        assert len(prices) == 3
        assert all(p > 0 for p in prices)

    def test_call_scalar_return_type(self):
        """Test that scalar inputs return scalar (float)."""
        price = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=100000,
            seed=42,
        )
        assert isinstance(price, float)
        assert not isinstance(price, np.ndarray)

    def test_call_invalid_simulations(self):
        """Test that invalid simulations parameter raises ValueError."""
        with pytest.raises(ValueError, match="simulations must be greater than 0"):
            monte_carlo_call(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                vol=0.25,
                time_to_expiry=0.5,
                simulations=0,
            )

        with pytest.raises(ValueError, match="simulations must be greater than 0"):
            monte_carlo_call(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                vol=0.25,
                time_to_expiry=0.5,
                simulations=-1,
            )

    def test_call_dividend_yield(self):
        """Test call pricing with dividend yield."""
        price_with_dividend = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            dividend_yield=0.02,
            simulations=100000,
            seed=42,
        )

        price_no_dividend = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            dividend_yield=0.0,
            simulations=100000,
            seed=42,
        )

        # With dividends, call price should be lower (ceteris paribus)
        assert price_with_dividend < price_no_dividend

    def test_call_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        price1 = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=100000,
            seed=42,
        )

        price2 = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=100000,
            seed=42,
        )

        # Same seed should produce identical results
        assert price1 == price2

    def test_call_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        price1 = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=100000,
            seed=42,
        )

        price2 = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=100000,
            seed=123,
        )

        # Different seeds should produce different results (very likely)
        # Note: There's a tiny chance they could be the same, but extremely unlikely
        assert price1 != price2

    def test_call_convergence_with_more_simulations(self):
        """Test that more simulations improve accuracy."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        vol = 0.20
        time_to_expiry = 1.0

        bs_price = black_scholes_call(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        # Test with fewer simulations
        price_10k = monte_carlo_call(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            simulations=10000,
            seed=42,
        )

        # Test with more simulations
        price_100k = monte_carlo_call(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            simulations=100000,
            seed=42,
        )

        error_10k = abs(price_10k - bs_price)
        error_100k = abs(price_100k - bs_price)

        # More simulations should generally be closer (or at least not worse)
        # Allow some tolerance since MC is stochastic
        assert error_100k <= error_10k * 1.5  # Allow 50% tolerance for stochastic nature

    def test_call_different_simulation_counts(self):
        """Test that different simulation counts produce reasonable results."""
        price_50k = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=50000,
            seed=42,
        )

        price_200k = monte_carlo_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=200000,
            seed=42,
        )

        # Both should be positive and reasonably close
        assert price_50k > 0
        assert price_200k > 0
        # Allow up to 5% difference due to Monte Carlo variance
        assert abs(price_50k - price_200k) / max(price_50k, price_200k) < 0.05


class TestMonteCarloPut:
    """Test cases for monte_carlo_put function."""

    def test_basic_put_price(self):
        """Test basic put option pricing."""
        price = monte_carlo_put(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=100000,
            seed=42,
        )
        # Should be close to Black-Scholes for same parameters
        bs_price = black_scholes_put(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )
        # Allow reasonable Monte Carlo error
        error = abs(price - bs_price)
        assert error < max(0.15, bs_price * 0.02)
        assert price > 0
        assert isinstance(price, float)

    def test_put_at_the_money(self):
        """Test at-the-money put option."""
        price = monte_carlo_put(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            simulations=100000,
            seed=42,
        )
        assert price > 0
        assert isinstance(price, float)

    def test_put_in_the_money(self):
        """Test in-the-money put option."""
        price = monte_carlo_put(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            simulations=100000,
            seed=42,
        )
        # ITM put should be worth at least intrinsic value
        assert price >= 10.0  # Intrinsic value = 100 - 90 = 10

    def test_put_array_inputs(self):
        """Test put pricing with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        prices = monte_carlo_put(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            simulations=100000,
            seed=42,
        )

        assert isinstance(prices, np.ndarray)
        assert len(prices) == 3
        # ITM > ATM > OTM for puts
        assert prices[0] > prices[1] > prices[2]

    def test_put_scalar_return_type(self):
        """Test that scalar inputs return scalar (float)."""
        price = monte_carlo_put(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=100000,
            seed=42,
        )
        assert isinstance(price, float)
        assert not isinstance(price, np.ndarray)

    def test_put_invalid_simulations(self):
        """Test that invalid simulations parameter raises ValueError."""
        with pytest.raises(ValueError, match="simulations must be greater than 0"):
            monte_carlo_put(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                vol=0.25,
                time_to_expiry=0.5,
                simulations=0,
            )

    def test_put_dividend_yield(self):
        """Test put pricing with dividend yield."""
        price_with_dividend = monte_carlo_put(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            dividend_yield=0.02,
            simulations=100000,
            seed=42,
        )

        price_no_dividend = monte_carlo_put(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            dividend_yield=0.0,
            simulations=100000,
            seed=42,
        )

        # With dividends, put price should be higher (ceteris paribus)
        assert price_with_dividend > price_no_dividend

    def test_put_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        price1 = monte_carlo_put(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=100000,
            seed=42,
        )

        price2 = monte_carlo_put(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            simulations=100000,
            seed=42,
        )

        # Same seed should produce identical results
        assert price1 == price2

    def test_put_convergence_to_black_scholes(self):
        """Test that Monte Carlo converges to Black-Scholes as simulations increase."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        vol = 0.20
        time_to_expiry = 1.0

        bs_price = black_scholes_put(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        price_100k = monte_carlo_put(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            simulations=100000,
            seed=42,
        )

        error = abs(price_100k - bs_price)
        # Should be within reasonable Monte Carlo error (2% or 0.15)
        assert error < max(0.15, bs_price * 0.02)


class TestMonteCarloAccuracy:
    """Test Monte Carlo accuracy against Black-Scholes."""

    def test_call_accuracy_against_black_scholes(self):
        """Test that Monte Carlo call prices are close to Black-Scholes."""
        # Use multiple test cases
        test_cases = [
            (100.0, 100.0, 0.05, 0.20, 1.0),  # ATM
            (110.0, 100.0, 0.05, 0.20, 1.0),  # ITM
            (90.0, 100.0, 0.05, 0.20, 1.0),   # OTM
        ]

        for spot, strike, rate, vol, time_to_expiry in test_cases:
            bs_price = black_scholes_call(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time_to_expiry,
            )

            mc_price = monte_carlo_call(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time_to_expiry,
                simulations=200000,  # Higher simulations for better accuracy
                seed=42,
            )

            error = abs(mc_price - bs_price)
            # Allow 2% error or 0.15 absolute, whichever is larger
            max_error = max(0.15, bs_price * 0.02)
            assert error < max_error, f"Error {error} too large for spot={spot}, strike={strike}"

    def test_put_accuracy_against_black_scholes(self):
        """Test that Monte Carlo put prices are close to Black-Scholes."""
        test_cases = [
            (100.0, 100.0, 0.05, 0.20, 1.0),  # ATM
            (90.0, 100.0, 0.05, 0.20, 1.0),   # ITM
            (110.0, 100.0, 0.05, 0.20, 1.0),  # OTM
        ]

        for spot, strike, rate, vol, time_to_expiry in test_cases:
            bs_price = black_scholes_put(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time_to_expiry,
            )

            mc_price = monte_carlo_put(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time_to_expiry,
                simulations=200000,
                seed=42,
            )

            error = abs(mc_price - bs_price)
            max_error = max(0.15, bs_price * 0.02)
            assert error < max_error, f"Error {error} too large for spot={spot}, strike={strike}"

