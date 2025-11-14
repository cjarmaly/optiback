"""Tests for Black-Scholes implied volatility calculation."""

import numpy as np
import pytest

from optiback.pricing.black_scholes import black_scholes_call, black_scholes_put
from optiback.pricing.implied_vol import black_scholes_implied_volatility


class TestBlackScholesImpliedVolatility:
    """Test cases for black_scholes_implied_volatility function."""

    def test_basic_call_implied_volatility(self):
        """Test basic implied volatility calculation for call option."""
        # Calculate price with known volatility
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )

        # Calculate implied volatility from that price
        implied_vol = black_scholes_implied_volatility(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="call",
        )

        # Should recover the original volatility
        assert abs(implied_vol - 0.25) < 1e-4
        assert isinstance(implied_vol, float)

    def test_basic_put_implied_volatility(self):
        """Test basic implied volatility calculation for put option."""
        # Calculate price with known volatility
        price = black_scholes_put(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )

        # Calculate implied volatility from that price
        implied_vol = black_scholes_implied_volatility(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="put",
        )

        # Should recover the original volatility
        assert abs(implied_vol - 0.25) < 1e-4
        assert isinstance(implied_vol, float)

    def test_implied_volatility_different_levels(self):
        """Test implied volatility at different volatility levels."""
        volatilities = [0.10, 0.20, 0.30, 0.40, 0.50]

        for vol in volatilities:
            price = black_scholes_call(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                vol=vol,
                time_to_expiry=0.5,
            )

            implied_vol = black_scholes_implied_volatility(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                time_to_expiry=0.5,
                market_price=price,
                option_type="call",
            )

            # Should recover the original volatility
            assert abs(implied_vol - vol) < 1e-3

    def test_implied_volatility_with_dividends(self):
        """Test implied volatility calculation with dividend yield."""
        dividend_yield = 0.01

        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            dividend_yield=dividend_yield,
        )

        implied_vol = black_scholes_implied_volatility(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="call",
            dividend_yield=dividend_yield,
        )

        # Should recover the original volatility
        assert abs(implied_vol - 0.25) < 1e-4

    def test_implied_volatility_at_intrinsic_value(self):
        """Test that implied vol is 0 when price equals intrinsic value."""
        # ITM call - intrinsic value with discounting
        spot = 110.0
        strike = 100.0
        rate = 0.02
        time_to_expiry = 0.5
        # Intrinsic value with discounting: max(S*e^(-qT) - K*e^(-rT), 0)
        intrinsic_value = spot * np.exp(0) - strike * np.exp(-rate * time_to_expiry)

        implied_vol = black_scholes_implied_volatility(
            spot=spot,
            strike=strike,
            rate=rate,
            time_to_expiry=time_to_expiry,
            market_price=intrinsic_value,
            option_type="call",
        )

        # Implied vol should be 0 (or very close to 0)
        assert abs(implied_vol) < 1e-6

    def test_implied_volatility_invalid_price_below_intrinsic(self):
        """Test that market price below intrinsic value raises ValueError."""
        spot = 110.0
        strike = 100.0
        intrinsic_value = spot - strike  # 10.0

        # Market price below intrinsic should raise error
        with pytest.raises(ValueError, match="Market price.*must be >= intrinsic value"):
            black_scholes_implied_volatility(
                spot=spot,
                strike=strike,
                rate=0.02,
                time_to_expiry=0.5,
                market_price=intrinsic_value - 1.0,  # Below intrinsic
                option_type="call",
            )

    def test_implied_volatility_zero_time_to_expiry(self):
        """Test that zero time to expiry raises ValueError."""
        with pytest.raises(ValueError, match="Implied volatility is undefined at expiry"):
            black_scholes_implied_volatility(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                time_to_expiry=0.0,
                market_price=7.0,
                option_type="call",
            )

    def test_implied_volatility_invalid_option_type(self):
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            black_scholes_implied_volatility(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                time_to_expiry=0.5,
                market_price=7.0,
                option_type="invalid",
            )

    def test_implied_volatility_case_insensitive_option_type(self):
        """Test that option type is case-insensitive."""
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )

        iv1 = black_scholes_implied_volatility(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="CALL",
        )

        iv2 = black_scholes_implied_volatility(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="call",
        )

        assert abs(iv1 - iv2) < 1e-10

    def test_implied_volatility_array_inputs(self):
        """Test implied volatility calculation with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])
        vol = 0.30

        # Calculate prices with known volatility
        prices = black_scholes_call(
            spot=spots,
            strike=strikes,
            rate=0.02,
            vol=vol,
            time_to_expiry=0.5,
        )

        # Calculate implied volatilities
        implied_vols = black_scholes_implied_volatility(
            spot=spots,
            strike=strikes,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=prices,
            option_type="call",
        )

        # Should return array
        assert isinstance(implied_vols, np.ndarray)
        assert len(implied_vols) == 3

        # Should recover original volatility for all
        assert np.allclose(implied_vols, vol, atol=1e-3)

    def test_implied_volatility_round_trip(self):
        """Test round-trip: price -> implied vol -> price."""
        original_vol = 0.25

        # Step 1: Calculate price with known vol
        original_price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=original_vol,
            time_to_expiry=0.5,
        )

        # Step 2: Calculate implied vol from price
        implied_vol = black_scholes_implied_volatility(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=original_price,
            option_type="call",
        )

        # Step 3: Calculate price from implied vol
        recovered_price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=implied_vol,
            time_to_expiry=0.5,
        )

        # Recovered price should match original
        assert abs(recovered_price - original_price) < 1e-6

    def test_implied_volatility_itm_call(self):
        """Test implied volatility for in-the-money call option."""
        price = black_scholes_call(
            spot=110.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )

        implied_vol = black_scholes_implied_volatility(
            spot=110.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="call",
        )

        assert abs(implied_vol - 0.25) < 1e-4

    def test_implied_volatility_otm_call(self):
        """Test implied volatility for out-of-the-money call option."""
        price = black_scholes_call(
            spot=90.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )

        implied_vol = black_scholes_implied_volatility(
            spot=90.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="call",
        )

        assert abs(implied_vol - 0.25) < 1e-4

    def test_implied_volatility_itm_put(self):
        """Test implied volatility for in-the-money put option."""
        price = black_scholes_put(
            spot=90.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )

        implied_vol = black_scholes_implied_volatility(
            spot=90.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="put",
        )

        assert abs(implied_vol - 0.25) < 1e-4

    def test_implied_volatility_otm_put(self):
        """Test implied volatility for out-of-the-money put option."""
        price = black_scholes_put(
            spot=110.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )

        implied_vol = black_scholes_implied_volatility(
            spot=110.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="put",
        )

        assert abs(implied_vol - 0.25) < 1e-4

    def test_implied_volatility_different_times_to_expiry(self):
        """Test implied volatility with different times to expiry."""
        times = [0.25, 0.5, 1.0, 2.0]

        for time in times:
            price = black_scholes_call(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                vol=0.25,
                time_to_expiry=time,
            )

            implied_vol = black_scholes_implied_volatility(
                spot=100.0,
                strike=100.0,
                rate=0.02,
                time_to_expiry=time,
                market_price=price,
                option_type="call",
            )

            assert abs(implied_vol - 0.25) < 1e-4

    def test_implied_volatility_different_interest_rates(self):
        """Test implied volatility with different interest rates."""
        rates = [0.0, 0.02, 0.05, 0.10]

        for rate in rates:
            price = black_scholes_call(
                spot=100.0,
                strike=100.0,
                rate=rate,
                vol=0.25,
                time_to_expiry=0.5,
            )

            implied_vol = black_scholes_implied_volatility(
                spot=100.0,
                strike=100.0,
                rate=rate,
                time_to_expiry=0.5,
                market_price=price,
                option_type="call",
            )

            assert abs(implied_vol - 0.25) < 1e-4

    def test_implied_volatility_scalar_return_type(self):
        """Test that scalar inputs return scalar (float)."""
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )

        implied_vol = black_scholes_implied_volatility(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="call",
        )

        assert isinstance(implied_vol, float)
        assert not isinstance(implied_vol, np.ndarray)

    def test_implied_volatility_custom_tolerance(self):
        """Test that custom tolerance parameter works."""
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )

        # Use stricter tolerance
        implied_vol = black_scholes_implied_volatility(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="call",
            tolerance=1e-8,  # Stricter than default
        )

        assert abs(implied_vol - 0.25) < 1e-6

    def test_implied_volatility_custom_initial_guess(self):
        """Test that custom initial guess parameter works."""
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.30,  # Different vol
            time_to_expiry=0.5,
        )

        # Use custom initial guess
        implied_vol = black_scholes_implied_volatility(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=price,
            option_type="call",
            initial_guess=0.15,  # Different from default 0.2
        )

        assert abs(implied_vol - 0.30) < 1e-4

    def test_implied_volatility_put_intrinsic_value(self):
        """Test implied vol for put option at intrinsic value."""
        spot = 90.0
        strike = 100.0
        rate = 0.02
        time_to_expiry = 0.5
        # Intrinsic value with discounting: max(K*e^(-rT) - S*e^(-qT), 0)
        intrinsic_value = strike * np.exp(-rate * time_to_expiry) - spot * np.exp(0)

        implied_vol = black_scholes_implied_volatility(
            spot=spot,
            strike=strike,
            rate=rate,
            time_to_expiry=time_to_expiry,
            market_price=intrinsic_value,
            option_type="put",
        )

        # Implied vol should be 0 (or very close to 0)
        assert abs(implied_vol) < 1e-6

    def test_implied_volatility_very_high_market_price(self):
        """Test implied volatility with very high market price."""
        # Use a very high market price
        high_price = 50.0

        implied_vol = black_scholes_implied_volatility(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=high_price,
            option_type="call",
        )

        # Should calculate a very high implied vol, but within bounds
        assert implied_vol > 0.0
        assert implied_vol <= 5.0  # Within max bound

    def test_implied_volatility_array_scalar_mixed(self):
        """Test implied volatility with mixed array and scalar inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strike = 100.0  # Scalar
        vol = 0.30

        prices = black_scholes_call(
            spot=spots,
            strike=strike,
            rate=0.02,
            vol=vol,
            time_to_expiry=0.5,
        )

        implied_vols = black_scholes_implied_volatility(
            spot=spots,
            strike=strike,
            rate=0.02,
            time_to_expiry=0.5,
            market_price=prices,
            option_type="call",
        )

        assert isinstance(implied_vols, np.ndarray)
        assert np.allclose(implied_vols, vol, atol=1e-3)
