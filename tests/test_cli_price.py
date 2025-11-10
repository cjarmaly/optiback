"""Tests for the CLI price command."""

from typer.testing import CliRunner

from optiback.cli.optiback import app

runner = CliRunner()


def test_price_call_option():
    """Test pricing a call option via CLI."""
    result = runner.invoke(
        app,
        [
            "price",
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
    assert "Option Pricing Results" in result.stdout
    assert "Call" in result.stdout
    assert "7.5168" in result.stdout


def test_price_put_option():
    """Test pricing a put option via CLI."""
    result = runner.invoke(
        app,
        [
            "price",
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
    assert "Option Pricing Results" in result.stdout
    assert "Put" in result.stdout
    assert "6.5218" in result.stdout


def test_price_with_dividend():
    """Test pricing with dividend yield."""
    result = runner.invoke(
        app,
        [
            "price",
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


def test_price_validation_negative_spot():
    """Test that negative spot price is rejected."""
    result = runner.invoke(
        app,
        [
            "price",
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


def test_price_validation_negative_strike():
    """Test that negative strike price is rejected."""
    result = runner.invoke(
        app,
        [
            "price",
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


def test_price_validation_negative_volatility():
    """Test that negative volatility is rejected."""
    result = runner.invoke(
        app,
        [
            "price",
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


def test_price_validation_negative_time():
    """Test that negative time to expiry is rejected."""
    result = runner.invoke(
        app,
        [
            "price",
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


def test_price_validation_invalid_option_type():
    """Test that invalid option type is rejected."""
    result = runner.invoke(
        app,
        [
            "price",
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


def test_price_validation_case_insensitive_option_type():
    """Test that option type is case-insensitive."""
    result = runner.invoke(
        app,
        [
            "price",
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
    assert result.exit_code == 0
    assert "Call" in result.stdout


def test_price_zero_volatility():
    """Test that zero volatility is allowed."""
    result = runner.invoke(
        app,
        [
            "price",
            "--spot",
            "100",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--vol",
            "0.0",
            "--time",
            "0.5",
            "--type",
            "call",
        ],
    )
    assert result.exit_code == 0


def test_price_zero_time():
    """Test that zero time to expiry is allowed."""
    result = runner.invoke(
        app,
        [
            "price",
            "--spot",
            "110",
            "--strike",
            "100",
            "--rate",
            "0.02",
            "--vol",
            "0.25",
            "--time",
            "0.0",
            "--type",
            "call",
        ],
    )
    assert result.exit_code == 0
    # At expiry, ITM call should equal intrinsic value
    assert "10.00" in result.stdout


def test_price_missing_required_parameter():
    """Test that missing required parameters cause an error."""
    result = runner.invoke(
        app,
        [
            "price",
            "--spot",
            "100",
            "--strike",
            "100",
            # Missing other required parameters
        ],
    )
    assert result.exit_code != 0
