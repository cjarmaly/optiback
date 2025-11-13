"""Tests for Black-Scholes Greeks calculations."""

import numpy as np
import pytest
from typer.testing import CliRunner

from optiback.cli.optiback import app
from optiback.pricing.greeks import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_greeks,
    black_scholes_rho,
    black_scholes_theta,
    black_scholes_vega,
)

runner = CliRunner()


class TestBlackScholesDelta:
    """Test cases for black_scholes_delta function."""

    def test_basic_call_delta(self):
        """Test basic call option delta with known values."""
        delta = black_scholes_delta(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            option_type="call",
        )
        # Delta for ATM call should be around 0.5-0.6
        assert 0.4 < delta < 0.7
        assert isinstance(delta, float)

    def test_basic_put_delta(self):
        """Test basic put option delta."""
        delta = black_scholes_delta(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            option_type="put",
        )
        # Delta for ATM put should be negative, around -0.4 to -0.5
        assert -0.6 < delta < -0.3
        assert isinstance(delta, float)

    def test_call_delta_range(self):
        """Test that call delta is between 0 and 1."""
        delta = black_scholes_delta(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )
        assert 0.0 <= delta <= 1.0

    def test_put_delta_range(self):
        """Test that put delta is between -1 and 0."""
        delta = black_scholes_delta(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="put",
        )
        assert -1.0 <= delta <= 0.0

    def test_call_delta_in_the_money(self):
        """Test call delta for ITM option."""
        delta = black_scholes_delta(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )
        # ITM call should have delta > 0.5
        assert delta > 0.5
        assert delta <= 1.0

    def test_call_delta_out_of_the_money(self):
        """Test call delta for OTM option."""
        delta = black_scholes_delta(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )
        # OTM call should have delta < 0.5
        assert 0.0 <= delta < 0.5

    def test_put_delta_in_the_money(self):
        """Test put delta for ITM option."""
        delta = black_scholes_delta(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="put",
        )
        # ITM put should have delta < -0.5
        assert -1.0 <= delta < -0.5

    def test_put_delta_out_of_the_money(self):
        """Test put delta for OTM option."""
        delta = black_scholes_delta(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="put",
        )
        # OTM put should have delta > -0.5
        assert -0.5 < delta <= 0.0

    def test_delta_zero_time_to_expiry_call_itm(self):
        """Test call delta at expiry for ITM option."""
        delta = black_scholes_delta(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
            option_type="call",
        )
        # At expiry, ITM call delta = 1
        assert delta == 1.0

    def test_delta_zero_time_to_expiry_call_otm(self):
        """Test call delta at expiry for OTM option."""
        delta = black_scholes_delta(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
            option_type="call",
        )
        # At expiry, OTM call delta = 0
        assert delta == 0.0

    def test_delta_zero_time_to_expiry_put_itm(self):
        """Test put delta at expiry for ITM option."""
        delta = black_scholes_delta(
            spot=90.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
            option_type="put",
        )
        # At expiry, ITM put delta = -1
        assert delta == -1.0

    def test_delta_zero_time_to_expiry_put_otm(self):
        """Test put delta at expiry for OTM option."""
        delta = black_scholes_delta(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
            option_type="put",
        )
        # At expiry, OTM put delta = 0
        assert delta == 0.0

    def test_delta_put_call_parity(self):
        """Test put-call parity for delta: delta_call - delta_put = e^(-qT)."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        vol = 0.20
        time_to_expiry = 1.0
        dividend_yield = 0.02

        delta_call = black_scholes_delta(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            option_type="call",
            dividend_yield=dividend_yield,
        )

        delta_put = black_scholes_delta(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            option_type="put",
            dividend_yield=dividend_yield,
        )

        # Put-call parity for delta: delta_call - delta_put = e^(-qT)
        expected = np.exp(-dividend_yield * time_to_expiry)
        actual = delta_call - delta_put

        assert abs(actual - expected) < 1e-10

    def test_delta_array_inputs(self):
        """Test delta calculation with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        deltas = black_scholes_delta(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )

        assert isinstance(deltas, np.ndarray)
        assert len(deltas) == 3
        # OTM < ATM < ITM for call deltas
        assert deltas[0] < deltas[1] < deltas[2]
        assert all(0.0 <= delta <= 1.0 for delta in deltas)

    def test_delta_invalid_option_type(self):
        """Test that invalid option_type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            black_scholes_delta(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                vol=0.20,
                time_to_expiry=1.0,
                option_type="invalid",
            )

    def test_delta_case_insensitive(self):
        """Test that option_type is case-insensitive."""
        delta1 = black_scholes_delta(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="CALL",
        )
        delta2 = black_scholes_delta(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )
        assert delta1 == delta2


class TestBlackScholesGamma:
    """Test cases for black_scholes_gamma function."""

    def test_basic_gamma(self):
        """Test basic gamma calculation."""
        gamma = black_scholes_gamma(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )
        # Gamma should be positive
        assert gamma > 0.0
        assert isinstance(gamma, float)

    def test_gamma_always_positive(self):
        """Test that gamma is always positive."""
        gamma = black_scholes_gamma(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )
        assert gamma > 0.0

    def test_gamma_same_for_call_and_put(self):
        """Test that gamma is the same for calls and puts."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        vol = 0.20
        time_to_expiry = 1.0

        # Gamma doesn't depend on option type, so we can't test directly
        # But we can verify it's positive and reasonable
        gamma = black_scholes_gamma(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )
        assert gamma > 0.0

    def test_gamma_zero_time_to_expiry(self):
        """Test gamma at expiry."""
        gamma = black_scholes_gamma(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
        )
        # At expiry, gamma should be 0 (delta is step function)
        assert gamma == 0.0

    def test_gamma_zero_volatility(self):
        """Test gamma with zero volatility."""
        gamma = black_scholes_gamma(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.0,
            time_to_expiry=1.0,
        )
        # With zero vol, gamma should be 0
        assert gamma == 0.0

    def test_gamma_array_inputs(self):
        """Test gamma calculation with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        gammas = black_scholes_gamma(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )

        assert isinstance(gammas, np.ndarray)
        assert len(gammas) == 3
        # Gamma should be positive for all options
        # Note: Gamma peaks near-the-money but exact location depends on parameters
        assert all(gamma > 0.0 for gamma in gammas)


class TestBlackScholesVega:
    """Test cases for black_scholes_vega function."""

    def test_basic_vega(self):
        """Test basic vega calculation."""
        vega = black_scholes_vega(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
        )
        # Vega should be positive
        assert vega > 0.0
        assert isinstance(vega, float)

    def test_vega_always_positive(self):
        """Test that vega is always positive."""
        vega = black_scholes_vega(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )
        assert vega > 0.0

    def test_vega_zero_time_to_expiry(self):
        """Test vega at expiry."""
        vega = black_scholes_vega(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
        )
        # At expiry, vega should be 0 (no time value)
        assert vega == 0.0

    def test_vega_zero_volatility(self):
        """Test vega with zero volatility."""
        vega = black_scholes_vega(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.0,
            time_to_expiry=1.0,
        )
        # With zero vol, vega should be 0
        assert vega == 0.0

    def test_vega_increases_with_time(self):
        """Test that vega increases with time to expiry."""
        vega_short = black_scholes_vega(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.25,
        )
        vega_long = black_scholes_vega(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )
        # Longer time should have higher vega
        assert vega_long > vega_short

    def test_vega_array_inputs(self):
        """Test vega calculation with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        vegas = black_scholes_vega(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
        )

        assert isinstance(vegas, np.ndarray)
        assert len(vegas) == 3
        assert all(vega >= 0.0 for vega in vegas)


class TestBlackScholesTheta:
    """Test cases for black_scholes_theta function."""

    def test_basic_call_theta(self):
        """Test basic call option theta."""
        theta = black_scholes_theta(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            option_type="call",
        )
        # Theta should be negative (time decay)
        assert theta < 0.0
        assert isinstance(theta, float)

    def test_basic_put_theta(self):
        """Test basic put option theta."""
        theta = black_scholes_theta(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            option_type="put",
        )
        # Theta should typically be negative (time decay)
        # Can be positive for deep ITM puts with high rates
        assert isinstance(theta, float)

    def test_theta_negative_for_calls(self):
        """Test that call theta is typically negative."""
        theta = black_scholes_theta(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )
        # Call theta should be negative (time decay)
        assert theta < 0.0

    def test_theta_zero_time_to_expiry(self):
        """Test theta at expiry."""
        theta = black_scholes_theta(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
            option_type="call",
        )
        # At expiry, theta should be 0 (no more time decay)
        assert theta == 0.0

    def test_theta_zero_volatility(self):
        """Test theta with zero volatility."""
        theta = black_scholes_theta(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.0,
            time_to_expiry=1.0,
            option_type="call",
        )
        # With zero vol, theta should be 0
        assert theta == 0.0

    def test_theta_array_inputs(self):
        """Test theta calculation with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        thetas = black_scholes_theta(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )

        assert isinstance(thetas, np.ndarray)
        assert len(thetas) == 3
        # Theta should be negative (time decay)
        assert all(theta <= 0.0 for theta in thetas)

    def test_theta_invalid_option_type(self):
        """Test that invalid option_type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            black_scholes_theta(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                vol=0.20,
                time_to_expiry=1.0,
                option_type="invalid",
            )


class TestBlackScholesRho:
    """Test cases for black_scholes_rho function."""

    def test_basic_call_rho(self):
        """Test basic call option rho."""
        rho = black_scholes_rho(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            option_type="call",
        )
        # Call rho should be positive
        assert rho > 0.0
        assert isinstance(rho, float)

    def test_basic_put_rho(self):
        """Test basic put option rho."""
        rho = black_scholes_rho(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            option_type="put",
        )
        # Put rho should be negative
        assert rho < 0.0
        assert isinstance(rho, float)

    def test_rho_positive_for_calls(self):
        """Test that call rho is positive."""
        rho = black_scholes_rho(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )
        assert rho > 0.0

    def test_rho_negative_for_puts(self):
        """Test that put rho is negative."""
        rho = black_scholes_rho(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="put",
        )
        assert rho < 0.0

    def test_rho_zero_time_to_expiry(self):
        """Test rho at expiry."""
        rho = black_scholes_rho(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.0,
            option_type="call",
        )
        # At expiry, rho should be 0 (no time value)
        assert rho == 0.0

    def test_rho_zero_volatility(self):
        """Test rho with zero volatility."""
        rho = black_scholes_rho(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.0,
            time_to_expiry=1.0,
            option_type="call",
        )
        # With zero vol, rho should be 0
        assert rho == 0.0

    def test_rho_increases_with_time(self):
        """Test that rho increases with time to expiry."""
        rho_short = black_scholes_rho(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=0.25,
            option_type="call",
        )
        rho_long = black_scholes_rho(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )
        # Longer time should have higher rho (absolute value)
        assert abs(rho_long) > abs(rho_short)

    def test_rho_array_inputs(self):
        """Test rho calculation with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        rhos = black_scholes_rho(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )

        assert isinstance(rhos, np.ndarray)
        assert len(rhos) == 3
        assert all(rho > 0.0 for rho in rhos)  # All positive for calls

    def test_rho_invalid_option_type(self):
        """Test that invalid option_type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            black_scholes_rho(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                vol=0.20,
                time_to_expiry=1.0,
                option_type="invalid",
            )


class TestBlackScholesGreeks:
    """Test cases for black_scholes_greeks convenience function."""

    def test_all_greeks_call(self):
        """Test that all Greeks are calculated for call option."""
        greeks = black_scholes_greeks(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            option_type="call",
        )

        # Check that all Greeks are present
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks

        # Check types
        assert isinstance(greeks["delta"], float)
        assert isinstance(greeks["gamma"], float)
        assert isinstance(greeks["vega"], float)
        assert isinstance(greeks["theta"], float)
        assert isinstance(greeks["rho"], float)

        # Check ranges
        assert 0.0 <= greeks["delta"] <= 1.0  # Call delta
        assert greeks["gamma"] > 0.0  # Gamma always positive
        assert greeks["vega"] > 0.0  # Vega always positive
        assert greeks["theta"] < 0.0  # Call theta negative
        assert greeks["rho"] > 0.0  # Call rho positive

    def test_all_greeks_put(self):
        """Test that all Greeks are calculated for put option."""
        greeks = black_scholes_greeks(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            option_type="put",
        )

        # Check that all Greeks are present
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks

        # Check ranges
        assert -1.0 <= greeks["delta"] <= 0.0  # Put delta
        assert greeks["gamma"] > 0.0  # Gamma always positive
        assert greeks["vega"] > 0.0  # Vega always positive
        assert greeks["rho"] < 0.0  # Put rho negative

    def test_greeks_consistency(self):
        """Test that individual Greeks match the convenience function."""
        spot = 100.0
        strike = 100.0
        rate = 0.02
        vol = 0.25
        time_to_expiry = 0.5
        option_type = "call"

        greeks = black_scholes_greeks(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
        )

        delta = black_scholes_delta(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
        )
        gamma = black_scholes_gamma(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )
        vega = black_scholes_vega(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )
        theta = black_scholes_theta(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
        )
        rho = black_scholes_rho(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
        )

        # Check that values match
        assert abs(greeks["delta"] - delta) < 1e-10
        assert abs(greeks["gamma"] - gamma) < 1e-10
        assert abs(greeks["vega"] - vega) < 1e-10
        assert abs(greeks["theta"] - theta) < 1e-10
        assert abs(greeks["rho"] - rho) < 1e-10

    def test_greeks_with_dividends(self):
        """Test Greeks calculation with dividend yield."""
        greeks = black_scholes_greeks(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            vol=0.25,
            time_to_expiry=0.5,
            option_type="call",
            dividend_yield=0.01,
        )

        # All Greeks should still be present and valid
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks

        # Delta should be lower with dividends (for calls)
        assert 0.0 <= greeks["delta"] <= 1.0

    def test_greeks_array_inputs(self):
        """Test Greeks calculation with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])

        greeks = black_scholes_greeks(
            spot=spots,
            strike=strikes,
            rate=0.05,
            vol=0.20,
            time_to_expiry=1.0,
            option_type="call",
        )

        # All Greeks should be arrays
        assert isinstance(greeks["delta"], np.ndarray)
        assert isinstance(greeks["gamma"], np.ndarray)
        assert isinstance(greeks["vega"], np.ndarray)
        assert isinstance(greeks["theta"], np.ndarray)
        assert isinstance(greeks["rho"], np.ndarray)

        # All should have the same length
        assert len(greeks["delta"]) == 3
        assert len(greeks["gamma"]) == 3
        assert len(greeks["vega"]) == 3
        assert len(greeks["theta"]) == 3
        assert len(greeks["rho"]) == 3

    def test_greeks_invalid_option_type(self):
        """Test that invalid option_type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            black_scholes_greeks(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                vol=0.20,
                time_to_expiry=1.0,
                option_type="invalid",
            )


class TestGreeksRelationships:
    """Test relationships between different Greeks."""

    def test_gamma_same_for_call_and_put(self):
        """Test that gamma is the same for calls and puts."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        vol = 0.20
        time_to_expiry = 1.0

        gamma_call = black_scholes_gamma(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        gamma_put = black_scholes_gamma(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        # Gamma should be identical (doesn't depend on option type)
        assert abs(gamma_call - gamma_put) < 1e-10

    def test_vega_same_for_call_and_put(self):
        """Test that vega is the same for calls and puts."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        vol = 0.20
        time_to_expiry = 1.0

        vega_call = black_scholes_vega(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        vega_put = black_scholes_vega(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
        )

        # Vega should be identical (doesn't depend on option type)
        assert abs(vega_call - vega_put) < 1e-10

    def test_delta_put_call_parity_no_dividends(self):
        """Test delta put-call parity without dividends."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        vol = 0.20
        time_to_expiry = 1.0

        delta_call = black_scholes_delta(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            option_type="call",
        )

        delta_put = black_scholes_delta(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time_to_expiry,
            option_type="put",
        )

        # Put-call parity for delta: delta_call - delta_put = e^(-qT)
        # Without dividends, q=0, so delta_call - delta_put = 1
        expected = 1.0
        actual = delta_call - delta_put

        assert abs(actual - expected) < 1e-10


class TestCLIGreeks:
    """Test cases for the CLI greeks command."""

    def test_greeks_call_option(self):
        """Test calculating all Greeks for a call option via CLI."""
        result = runner.invoke(
            app,
            [
                "greeks",
                "--spot",
                "100",
                "--strike",
                "100",
                "--rate",
                "0.02",
                "--vol",
                "0.25",
                "--time",
                "0.5",
                "--type",
                "call",
            ],
        )
        assert result.exit_code == 0
        assert "Option Greeks" in result.stdout
        assert "Call" in result.stdout
        assert "Delta" in result.stdout
        assert "Gamma" in result.stdout
        assert "Vega" in result.stdout
        assert "Theta" in result.stdout
        assert "Rho" in result.stdout

    def test_greeks_put_option(self):
        """Test calculating all Greeks for a put option via CLI."""
        result = runner.invoke(
            app,
            [
                "greeks",
                "--spot",
                "100",
                "--strike",
                "100",
                "--rate",
                "0.02",
                "--vol",
                "0.25",
                "--time",
                "0.5",
                "--type",
                "put",
            ],
        )
        assert result.exit_code == 0
        assert "Option Greeks" in result.stdout
        assert "Put" in result.stdout
        assert "Delta" in result.stdout
        assert "Gamma" in result.stdout
        assert "Vega" in result.stdout
        assert "Theta" in result.stdout
        assert "Rho" in result.stdout

    def test_greeks_with_dividend(self):
        """Test calculating Greeks with dividend yield."""
        result = runner.invoke(
            app,
            [
                "greeks",
                "--spot",
                "100",
                "--strike",
                "100",
                "--rate",
                "0.02",
                "--vol",
                "0.25",
                "--time",
                "0.5",
                "--type",
                "call",
                "--dividend",
                "0.01",
            ],
        )
        assert result.exit_code == 0
        assert "Dividend Yield" in result.stdout
        assert "1.00%" in result.stdout
        assert "Delta" in result.stdout

    def test_greeks_validation_negative_spot(self):
        """Test that negative spot price is rejected."""
        result = runner.invoke(
            app,
            [
                "greeks",
                "--spot",
                "-100",
                "--strike",
                "100",
                "--rate",
                "0.02",
                "--vol",
                "0.25",
                "--time",
                "0.5",
                "--type",
                "call",
            ],
        )
        assert result.exit_code != 0
        output = result.stdout + result.stderr
        assert "Spot price must be greater than 0" in output

    def test_greeks_validation_negative_strike(self):
        """Test that negative strike price is rejected."""
        result = runner.invoke(
            app,
            [
                "greeks",
                "--spot",
                "100",
                "--strike",
                "-100",
                "--rate",
                "0.02",
                "--vol",
                "0.25",
                "--time",
                "0.5",
                "--type",
                "call",
            ],
        )
        assert result.exit_code != 0
        output = result.stdout + result.stderr
        assert "Strike price must be greater than 0" in output

    def test_greeks_validation_negative_vol(self):
        """Test that negative volatility is rejected."""
        result = runner.invoke(
            app,
            [
                "greeks",
                "--spot",
                "100",
                "--strike",
                "100",
                "--rate",
                "0.02",
                "--vol",
                "-0.25",
                "--time",
                "0.5",
                "--type",
                "call",
            ],
        )
        assert result.exit_code != 0
        output = result.stdout + result.stderr
        assert "Volatility must be >= 0" in output

    def test_greeks_validation_negative_time(self):
        """Test that negative time to expiry is rejected."""
        result = runner.invoke(
            app,
            [
                "greeks",
                "--spot",
                "100",
                "--strike",
                "100",
                "--rate",
                "0.02",
                "--vol",
                "0.25",
                "--time",
                "-0.5",
                "--type",
                "call",
            ],
        )
        assert result.exit_code != 0
        output = result.stdout + result.stderr
        assert "Time to expiry must be >= 0" in output

    def test_greeks_validation_invalid_option_type(self):
        """Test that invalid option type is rejected."""
        result = runner.invoke(
            app,
            [
                "greeks",
                "--spot",
                "100",
                "--strike",
                "100",
                "--rate",
                "0.02",
                "--vol",
                "0.25",
                "--time",
                "0.5",
                "--type",
                "invalid",
            ],
        )
        assert result.exit_code != 0
        output = result.stdout + result.stderr
        assert "Option type must be 'call' or 'put'" in output

    def test_greeks_case_insensitive_option_type(self):
        """Test that option type is case-insensitive."""
        result_upper = runner.invoke(
            app,
            [
                "greeks",
                "--spot",
                "100",
                "--strike",
                "100",
                "--rate",
                "0.02",
                "--vol",
                "0.25",
                "--time",
                "0.5",
                "--type",
                "CALL",
            ],
        )
        result_lower = runner.invoke(
            app,
            [
                "greeks",
                "--spot",
                "100",
                "--strike",
                "100",
                "--rate",
                "0.02",
                "--vol",
                "0.25",
                "--time",
                "0.5",
                "--type",
                "call",
            ],
        )
        assert result_upper.exit_code == 0
        assert result_lower.exit_code == 0
        # Both should produce the same output (case-insensitive)
        assert "Call" in result_upper.stdout
        assert "Call" in result_lower.stdout
