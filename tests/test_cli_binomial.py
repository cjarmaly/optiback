"""Tests for the CLI price-binomial command."""

from typer.testing import CliRunner

from optiback.cli.optiback import app

runner = CliRunner()


def test_price_binomial_call_option():
    """Test pricing a call option via binomial tree CLI."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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
    assert "Binomial Tree Option Pricing" in result.stdout
    assert "Results" in result.stdout
    assert "Call" in result.stdout
    assert "Steps" in result.stdout
    assert "100" in result.stdout  # Default steps
    # Should have a reasonable price
    assert any(char.isdigit() for char in result.stdout.split("Option Price")[-1])


def test_price_binomial_put_option():
    """Test pricing a put option via binomial tree CLI."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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
    assert "Binomial Tree Option Pricing" in result.stdout
    assert "Results" in result.stdout
    assert "Put" in result.stdout
    assert "Steps" in result.stdout


def test_price_binomial_with_custom_steps():
    """Test pricing with custom steps parameter."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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
            "--steps",
            "200",
        ],
    )
    assert result.exit_code == 0
    assert "Steps" in result.stdout
    assert "200" in result.stdout


def test_price_binomial_with_dividend():
    """Test pricing with dividend yield."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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


def test_price_binomial_validation_negative_spot():
    """Test that negative spot price is rejected."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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


def test_price_binomial_validation_negative_strike():
    """Test that negative strike price is rejected."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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


def test_price_binomial_validation_negative_volatility():
    """Test that negative volatility is rejected."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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


def test_price_binomial_validation_negative_time():
    """Test that negative time to expiry is rejected."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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


def test_price_binomial_validation_invalid_steps():
    """Test that invalid steps parameter is rejected."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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
            "--steps",
            "0",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "steps must be greater than 0" in output


def test_price_binomial_validation_negative_steps():
    """Test that negative steps parameter is rejected."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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
            "--steps",
            "-10",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "steps must be greater than 0" in output


def test_price_binomial_validation_invalid_option_type():
    """Test that invalid option type is rejected."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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


def test_price_binomial_validation_case_insensitive_option_type():
    """Test that option type is case-insensitive."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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


def test_price_binomial_zero_volatility():
    """Test that zero volatility is allowed."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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


def test_price_binomial_zero_time():
    """Test that zero time to expiry is allowed."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
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


def test_price_binomial_missing_required_parameter():
    """Test that missing required parameters cause an error."""
    result = runner.invoke(
        app,
        [
            "price-binomial",
            "--spot",
            "100",
            "--strike",
            "100",
            # Missing other required parameters
        ],
    )
    assert result.exit_code != 0


def test_price_binomial_different_step_counts():
    """Test that different step counts produce reasonable results."""
    result_50 = runner.invoke(
        app,
        [
            "price-binomial",
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
            "--steps",
            "50",
        ],
    )

    result_200 = runner.invoke(
        app,
        [
            "price-binomial",
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
            "--steps",
            "200",
        ],
    )

    assert result_50.exit_code == 0
    assert result_200.exit_code == 0
    # Both should show steps in output
    assert "50" in result_50.stdout
    assert "200" in result_200.stdout
