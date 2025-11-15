"""Tests for Binomial Tree option pricing."""

import numpy as np
import pytest

from optiback.pricing.binomial import binomial_tree_call, binomial_tree_put
from optiback.pricing.black_scholes import black_scholes_call, black_scholes_put


class TestBinomialTreeCall:
    """Test cases for binomial_tree_call function."""

    def test_basic_call_price(self):
        """Test basic call option pricing."""
        price = binomial_tree_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            steps=100,
        )
        # Should be close to Black-Scholes for same parameters
        bs_price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )
        # Allow small difference due to discretization
        assert abs(price - bs_price) < 0.1
        assert price > 0
        assert isinstance(price, float)

    def test_call_at_the_money(self):
        """Test at-the-money call option."""
        price = binomial_tree_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            steps=100,
        )
        assert price > 0
        assert isinstance(price, float)

    def test_call_in_the_money(self):
        """Test in-the-money call option."""
        price = binomial_tree_call(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            steps=100,
        )
        # ITM call should be worth at least intrinsic value
        assert price >= 10.0  # Intrinsic value = 110 - 100 = 10

    def test_call_out_of_the_money(self):
        """Test out-of-the-money call option."""
        price = binomial_tree_call(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            steps=100,
        )
        # OTM call should have positive value due to time value
        assert price > 0
        assert price < 10.0  # Less than intrinsic value if ITM

    def test_call_zero_time_to_expiry(self):
        """Test call option with zero time to expiry (intrinsic value)."""
        price = binomial_tree_call(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
            steps=100,
        )
        # At expiry, value equals intrinsic value
        assert price == 10.0

    def test_call_zero_time_otm(self):
        """Test OTM call with zero time to expiry."""
        price = binomial_tree_call(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
            steps=100,
        )
        # OTM option at expiry is worthless
        assert price == 0.0

    def test_call_zero_volatility(self):
        """Test call option with zero volatility."""
        price = binomial_tree_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.0,
            time_to_expiry=1.0,
            steps=100,
        )
        # Zero vol should give discounted intrinsic value
        assert price >= 0.0

    def test_call_array_inputs(self):
        """Test call pricing with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        prices = binomial_tree_call(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            steps=100,
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

        prices = binomial_tree_call(
            spot=spots,
            strike=strike,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            steps=100,
        )

        assert isinstance(prices, np.ndarray)
        assert len(prices) == 3
        assert all(p > 0 for p in prices)

    def test_call_scalar_return_type(self):
        """Test that scalar inputs return scalar (float)."""
        price = binomial_tree_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            steps=100,
        )
        assert isinstance(price, float)
        assert not isinstance(price, np.ndarray)

    def test_call_invalid_steps(self):
        """Test that invalid steps parameter raises ValueError."""
        with pytest.raises(ValueError, match="steps must be greater than 0"):
            binomial_tree_call(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                vol=0.25,
                time_to_expiry=0.5,
                steps=0,
            )

        with pytest.raises(ValueError, match="steps must be greater than 0"):
            binomial_tree_call(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                vol=0.25,
                time_to_expiry=0.5,
                steps=-1,
            )

    def test_call_dividend_yield(self):
        """Test call pricing with dividend yield."""
        price_with_dividend = binomial_tree_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            dividend_yield=0.02,
            steps=100,
        )

        price_no_dividend = binomial_tree_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            dividend_yield=0.0,
            steps=100,
        )

        # With dividends, call price should be lower (ceteris paribus)
        assert price_with_dividend < price_no_dividend

    def test_call_convergence_to_black_scholes(self):
        """Test that binomial tree converges to Black-Scholes as steps increase."""
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

        # Test with increasing steps
        prices_100 = binomial_tree_call(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            steps=100,
        )

        prices_500 = binomial_tree_call(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            steps=500,
        )

        # More steps should be closer to Black-Scholes
        error_100 = abs(prices_100 - bs_price)
        error_500 = abs(prices_500 - bs_price)

        # Higher steps should have lower error (or at least similar)
        assert error_500 <= error_100 * 1.1  # Allow small tolerance

    def test_call_different_steps(self):
        """Test that different step counts produce reasonable results."""
        price_50 = binomial_tree_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            steps=50,
        )

        price_200 = binomial_tree_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            steps=200,
        )

        # Both should be positive and reasonably close
        assert price_50 > 0
        assert price_200 > 0
        # Allow up to 5% difference due to discretization
        assert abs(price_50 - price_200) / max(price_50, price_200) < 0.05


class TestBinomialTreePut:
    """Test cases for binomial_tree_put function."""

    def test_basic_put_price(self):
        """Test basic put option pricing."""
        price = binomial_tree_put(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            steps=100,
        )
        # Should be close to Black-Scholes for same parameters
        bs_price = black_scholes_put(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )
        # Allow small difference due to discretization
        assert abs(price - bs_price) < 0.1
        assert price > 0
        assert isinstance(price, float)

    def test_put_at_the_money(self):
        """Test at-the-money put option."""
        price = binomial_tree_put(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            steps=100,
        )
        assert price > 0
        assert isinstance(price, float)

    def test_put_in_the_money(self):
        """Test in-the-money put option."""
        price = binomial_tree_put(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            steps=100,
        )
        # ITM put should be worth at least intrinsic value
        assert price >= 10.0  # Intrinsic value = 100 - 90 = 10

    def test_put_out_of_the_money(self):
        """Test out-of-the-money put option."""
        price = binomial_tree_put(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            steps=100,
        )
        # OTM put should have positive value due to time value
        assert price > 0

    def test_put_zero_time_to_expiry(self):
        """Test put option with zero time to expiry."""
        price = binomial_tree_put(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
            steps=100,
        )
        # At expiry, value equals intrinsic value
        assert price == 10.0

    def test_put_array_inputs(self):
        """Test put pricing with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        prices = binomial_tree_put(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            steps=100,
        )

        assert isinstance(prices, np.ndarray)
        assert len(prices) == 3
        # ITM > ATM > OTM for puts
        assert prices[0] > prices[1] > prices[2]

    def test_put_scalar_return_type(self):
        """Test that scalar inputs return scalar (float)."""
        price = binomial_tree_put(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            steps=100,
        )
        assert isinstance(price, float)
        assert not isinstance(price, np.ndarray)

    def test_put_invalid_steps(self):
        """Test that invalid steps parameter raises ValueError."""
        with pytest.raises(ValueError, match="steps must be greater than 0"):
            binomial_tree_put(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                vol=0.25,
                time_to_expiry=0.5,
                steps=0,
            )

    def test_put_dividend_yield(self):
        """Test put pricing with dividend yield."""
        price_with_dividend = binomial_tree_put(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            dividend_yield=0.02,
            steps=100,
        )

        price_no_dividend = binomial_tree_put(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            dividend_yield=0.0,
            steps=100,
        )

        # With dividends, put price should be higher (ceteris paribus)
        assert price_with_dividend > price_no_dividend

    def test_put_convergence_to_black_scholes(self):
        """Test that binomial tree converges to Black-Scholes as steps increase."""
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

        prices_100 = binomial_tree_put(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            steps=100,
        )

        prices_500 = binomial_tree_put(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            steps=500,
        )

        error_100 = abs(prices_100 - bs_price)
        error_500 = abs(prices_500 - bs_price)

        # Higher steps should have lower error (or at least similar)
        assert error_500 <= error_100 * 1.1  # Allow small tolerance


class TestBinomialTreeAmericanFeatures:
    """Test American option specific features."""

    def test_american_call_early_exercise(self):
        """Test that American call without dividends should not be exercised early."""
        # For non-dividend paying stocks, American call = European call
        # So binomial tree should match Black-Scholes closely
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

        binomial_price = binomial_tree_call(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            dividend_yield=0.0,
            steps=200,
        )

        # Should be very close (within discretization error)
        assert abs(binomial_price - bs_price) < 0.15

    def test_american_put_may_exercise_early(self):
        """Test that American put may be worth more than European put."""
        # American puts can be worth more due to early exercise possibility
        spot = 90.0  # Deep ITM
        strike = 100.0
        rate = 0.05
        vol = 0.10  # Low volatility
        time_to_expiry = 1.0

        european_price = black_scholes_put(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        american_price = binomial_tree_put(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            steps=200,
        )

        # American put should be >= European put
        assert american_price >= european_price - 0.1  # Allow small discretization error
