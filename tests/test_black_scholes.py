"""Tests for Black-Scholes option pricing."""

import numpy as np

from optiback.pricing.black_scholes import black_scholes_call, black_scholes_put


class TestBlackScholesCall:
    """Test cases for black_scholes_call function."""

    def test_basic_call_price(self):
        """Test basic call option pricing with known values."""
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )
        # Verified value from actual calculation
        assert round(price, 4) == 7.5168

    def test_call_at_the_money(self):
        """Test at-the-money call option."""
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )
        # At-the-money call should have positive time value
        assert price > 0
        assert isinstance(price, float)

    def test_call_in_the_money(self):
        """Test in-the-money call option."""
        price = black_scholes_call(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )
        # ITM call should be worth at least intrinsic value
        assert price >= 10.0  # Intrinsic value = 110 - 100 = 10

    def test_call_out_of_the_money(self):
        """Test out-of-the-money call option."""
        price = black_scholes_call(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )
        # OTM call should have positive value due to time value
        assert price > 0
        assert price < 10.0  # Less than intrinsic value if ITM

    def test_call_zero_time_to_expiry(self):
        """Test call option with zero time to expiry (intrinsic value)."""
        price = black_scholes_call(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
        )
        # At expiry, value equals intrinsic value
        assert price == 10.0

    def test_call_zero_time_otm(self):
        """Test OTM call with zero time to expiry."""
        price = black_scholes_call(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
        )
        # OTM option at expiry is worthless
        assert price == 0.0

    def test_call_zero_volatility(self):
        """Test call option with zero volatility."""
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.0,
            time_to_expiry=1.0,
        )
        # Zero vol should give discounted intrinsic value
        # With spot=strike and r>0, should be slightly positive due to discounting
        assert price >= 0.0

    def test_call_array_inputs(self):
        """Test call pricing with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        prices = black_scholes_call(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )

        # Should return array
        assert isinstance(prices, np.ndarray)
        assert len(prices) == 3
        # OTM < ATM < ITM
        assert prices[0] < prices[1] < prices[2]

    def test_call_scalar_return_type(self):
        """Test that scalar inputs return scalar (float)."""
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )
        assert isinstance(price, float)
        assert not isinstance(price, np.ndarray)


class TestBlackScholesPut:
    """Test cases for black_scholes_put function."""

    def test_basic_put_price(self):
        """Test basic put option pricing."""
        price = black_scholes_put(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )
        # Verified value from actual calculation
        assert round(price, 4) == 6.5218

    def test_put_at_the_money(self):
        """Test at-the-money put option."""
        price = black_scholes_put(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )
        assert price > 0
        assert isinstance(price, float)

    def test_put_in_the_money(self):
        """Test in-the-money put option."""
        price = black_scholes_put(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )
        # ITM put should be worth at least intrinsic value
        assert price >= 10.0  # Intrinsic value = 100 - 90 = 10

    def test_put_out_of_the_money(self):
        """Test out-of-the-money put option."""
        price = black_scholes_put(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )
        # OTM put should have positive value due to time value
        assert price > 0

    def test_put_zero_time_to_expiry(self):
        """Test put option with zero time to expiry."""
        price = black_scholes_put(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
        )
        # At expiry, value equals intrinsic value
        assert price == 10.0

    def test_put_array_inputs(self):
        """Test put pricing with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        prices = black_scholes_put(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )

        assert isinstance(prices, np.ndarray)
        assert len(prices) == 3
        # ITM > ATM > OTM for puts
        assert prices[0] > prices[1] > prices[2]


class TestPutCallParity:
    """Test put-call parity relationship."""

    def test_put_call_parity(self):
        """Test that put-call parity holds: C - P = S - K*e^(-rT)."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        vol = 0.20
        time_to_expiry = 1.0

        call_price = black_scholes_call(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        put_price = black_scholes_put(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        # Put-call parity: C - P = S - K*e^(-rT)
        left_side = call_price - put_price
        right_side = spot - strike * np.exp(-rate * time_to_expiry)

        # Should be approximately equal (within floating point precision)
        assert abs(left_side - right_side) < 1e-10

    def test_put_call_parity_with_dividends(self):
        """Test put-call parity with dividend yield."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        vol = 0.20
        time_to_expiry = 1.0
        dividend_yield = 0.02

        call_price = black_scholes_call(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            dividend_yield=dividend_yield,
        )

        put_price = black_scholes_put(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            dividend_yield=dividend_yield,
        )

        # Put-call parity with dividends: C - P = S*e^(-qT) - K*e^(-rT)
        left_side = call_price - put_price
        right_side = spot * np.exp(-dividend_yield * time_to_expiry) - strike * np.exp(
            -rate * time_to_expiry
        )

        assert abs(left_side - right_side) < 1e-10

    def test_put_call_parity_array(self):
        """Test put-call parity with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])
        rate = 0.05
        vol = 0.20
        time_to_expiry = 1.0

        call_prices = black_scholes_call(
            spot=spots,
            strike=strikes,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        put_prices = black_scholes_put(
            spot=spots,
            strike=strikes,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        # Check put-call parity for each element
        left_side = call_prices - put_prices
        right_side = spots - strikes * np.exp(-rate * time_to_expiry)

        np.testing.assert_allclose(left_side, right_side, atol=1e-10)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_time_to_expiry(self):
        """Test with very small time to expiry."""
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1e-6,
        )
        # Should be close to intrinsic value (0 for ATM)
        assert price >= 0.0
        assert price < 1.0  # Very small time value

    def test_very_high_volatility(self):
        """Test with very high volatility."""
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=2.0,  # 200% volatility
            time_to_expiry=1.0,
        )
        # High vol should increase option value
        assert price > 0.0

    def test_different_spot_strike_ratios(self):
        """Test with different spot/strike ratios."""
        strike = 100.0
        rates = [0.02, 0.05, 0.10]

        for rate in rates:
            # Deep ITM
            price_itm = black_scholes_call(
                spot=150.0,
                strike=strike,
                rate=rate,
                vol=0.20,
                time_to_expiry=1.0,
            )
            assert price_itm >= 50.0  # At least intrinsic value

            # Deep OTM
            price_otm = black_scholes_call(
                spot=50.0,
                strike=strike,
                rate=rate,
                vol=0.20,
                time_to_expiry=1.0,
            )
            assert price_otm >= 0.0
            assert price_otm < 10.0  # Much less than intrinsic if ITM
