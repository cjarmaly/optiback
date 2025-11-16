"""Tests for the CLI price-montecarlo command."""

from typer.testing import CliRunner

from optiback.cli.optiback import app

runner = CliRunner()


def test_price_montecarlo_call_option():
    """Test pricing a call option via Monte Carlo CLI."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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
    assert "Monte Carlo Option Pricing" in result.stdout
    assert "Results" in result.stdout
    assert "Call" in result.stdout
    assert "Simulations" in result.stdout
    assert "100,000" in result.stdout or "100000" in result.stdout  # Default simulations
    # Should have a reasonable price
    assert any(char.isdigit() for char in result.stdout.split("Option Price")[-1])


def test_price_montecarlo_put_option():
    """Test pricing a put option via Monte Carlo CLI."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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
    assert "Monte Carlo Option Pricing" in result.stdout
    assert "Results" in result.stdout
    assert "Put" in result.stdout
    assert "Simulations" in result.stdout


def test_price_montecarlo_with_custom_simulations():
    """Test pricing with custom simulations parameter."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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
            "--simulations",
            "200000",
        ],
    )
    assert result.exit_code == 0
    assert "Simulations" in result.stdout
    assert "200,000" in result.stdout or "200000" in result.stdout


def test_price_montecarlo_with_seed():
    """Test pricing with seed parameter for reproducibility."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0
    assert "Seed" in result.stdout
    assert "42" in result.stdout


def test_price_montecarlo_with_dividend():
    """Test pricing with dividend yield."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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


def test_price_montecarlo_validation_negative_spot():
    """Test that negative spot price is rejected."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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


def test_price_montecarlo_validation_negative_strike():
    """Test that negative strike price is rejected."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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


def test_price_montecarlo_validation_negative_volatility():
    """Test that negative volatility is rejected."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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


def test_price_montecarlo_validation_negative_time():
    """Test that negative time to expiry is rejected."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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


def test_price_montecarlo_validation_invalid_simulations():
    """Test that invalid simulations parameter is rejected."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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
            "--simulations",
            "0",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "simulations must be greater than 0" in output


def test_price_montecarlo_validation_negative_simulations():
    """Test that negative simulations parameter is rejected."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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
            "--simulations",
            "-10",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "simulations must be greater than 0" in output


def test_price_montecarlo_validation_invalid_option_type():
    """Test that invalid option type is rejected."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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


def test_price_montecarlo_validation_case_insensitive_option_type():
    """Test that option type is case-insensitive."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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


def test_price_montecarlo_zero_volatility():
    """Test that zero volatility is allowed."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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


def test_price_montecarlo_zero_time():
    """Test that zero time to expiry is allowed."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
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


def test_price_montecarlo_missing_required_parameter():
    """Test that missing required parameters cause an error."""
    result = runner.invoke(
        app,
        [
            "price-montecarlo",
            "--spot",
            "100",
            "--strike",
            "100",
            # Missing other required parameters
        ],
    )
    assert result.exit_code != 0


def test_price_montecarlo_reproducibility_with_seed():
    """Test that same seed produces same results."""
    result1 = runner.invoke(
        app,
        [
            "price-montecarlo",
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
            "--simulations",
            "50000",
            "--seed",
            "42",
        ],
    )

    result2 = runner.invoke(
        app,
        [
            "price-montecarlo",
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
            "--simulations",
            "50000",
            "--seed",
            "42",
        ],
    )

    assert result1.exit_code == 0
    assert result2.exit_code == 0
    # Extract prices from output (they should be identical with same seed)
    # Both should show the same price
    price1 = [line for line in result1.stdout.split("\n") if "Option Price" in line][0]
    price2 = [line for line in result2.stdout.split("\n") if "Option Price" in line][0]
    assert price1 == price2
