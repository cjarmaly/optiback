"""Tests for the CLI implied-vol command."""

from typer.testing import CliRunner

from optiback.cli.optiback import app

runner = CliRunner()


def test_implied_vol_call_option():
    """Test calculating implied volatility for a call option via CLI."""
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--time",
            "0.5",
            "--price",
            "7.5168",
            "--type",
            "call",
        ],
    )
    assert result.exit_code == 0
    assert "Implied Volatility" in result.stdout
    assert "Call" in result.stdout
    assert "0.2500" in result.stdout or "25.00%" in result.stdout
    assert "7.5168" in result.stdout


def test_implied_vol_put_option():
    """Test calculating implied volatility for a put option via CLI."""
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--time",
            "0.5",
            "--price",
            "6.5218",
            "--type",
            "put",
        ],
    )
    assert result.exit_code == 0
    assert "Implied Volatility" in result.stdout
    assert "Put" in result.stdout
    assert "0.2500" in result.stdout or "25.00%" in result.stdout


def test_implied_vol_with_dividend():
    """Test implied volatility calculation with dividend yield."""
    # Calculate price with dividend first
    from optiback.pricing import black_scholes_call

    price_with_dividend = black_scholes_call(
        spot=100.0,
        strike=100.0,
        rate=0.02,
        vol=0.25,
        time_to_expiry=0.5,
        dividend_yield=0.01,
    )

    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--time",
            "0.5",
            "--price",
            str(price_with_dividend),
            "--type",
            "call",
            "--dividend",
            "0.01",
        ],
    )
    assert result.exit_code == 0
    assert "Dividend Yield" in result.stdout
    assert "1.00%" in result.stdout
    assert "0.2500" in result.stdout or "25.00%" in result.stdout


def test_implied_vol_validation_negative_spot():
    """Test that negative spot price is rejected."""
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "-100",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--time",
            "0.5",
            "--price",
            "7.0",
            "--type",
            "call",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "Spot price must be greater than 0" in output


def test_implied_vol_validation_negative_strike():
    """Test that negative strike price is rejected."""
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "-100",
            "--rate",
            "0.02",
            "--time",
            "0.5",
            "--price",
            "7.0",
            "--type",
            "call",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "Strike price must be greater than 0" in output


def test_implied_vol_validation_negative_price():
    """Test that negative market price is rejected."""
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--time",
            "0.5",
            "--price",
            "-7.0",
            "--type",
            "call",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "Market price must be >= 0" in output


def test_implied_vol_validation_negative_time():
    """Test that negative time to expiry is rejected."""
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--time",
            "-0.5",
            "--price",
            "7.0",
            "--type",
            "call",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "Time to expiry must be >= 0" in output


def test_implied_vol_validation_zero_time():
    """Test that zero time to expiry raises error."""
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--time",
            "0.0",
            "--price",
            "7.0",
            "--type",
            "call",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "undefined at expiry" in output or "time_to_expiry must be > 0" in output


def test_implied_vol_validation_invalid_option_type():
    """Test that invalid option type is rejected."""
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--time",
            "0.5",
            "--price",
            "7.0",
            "--type",
            "invalid",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "Option type must be 'call' or 'put'" in output


def test_implied_vol_validation_case_insensitive_option_type():
    """Test that option type is case-insensitive."""
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--time",
            "0.5",
            "--price",
            "7.5168",
            "--type",
            "CALL",
        ],
    )
    assert result.exit_code == 0
    assert "Call" in result.stdout


def test_implied_vol_validation_price_below_intrinsic():
    """Test that market price below intrinsic value raises error."""
    # For spot=110, strike=100, intrinsic value is > 10
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "110",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--time",
            "0.5",
            "--price",
            "5.0",  # Below intrinsic
            "--type",
            "call",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "Market price" in output and "intrinsic value" in output


def test_implied_vol_missing_required_parameter():
    """Test that missing required parameters cause an error."""
    result = runner.invoke(
        app,
        [
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "100",
            # Missing other required parameters
        ],
    )
    assert result.exit_code != 0
