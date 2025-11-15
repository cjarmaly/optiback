"""Tests for the CLI greeks command."""

from typer.testing import CliRunner

from optiback.cli.optiback import app

runner = CliRunner()


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
