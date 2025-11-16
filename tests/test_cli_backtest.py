"""Tests for backtest CLI commands."""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from optiback.cli.optiback import app

runner = CliRunner()


class TestDeltaHedgeCLI:
    """Test delta-hedge CLI command."""

    def test_backtest_delta_hedge_success(self):
        """Test successful delta-hedge backtest."""
        # Create temporary file with spot prices
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n102.0\n100.0\n")
            spot_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-delta-hedge",
                    "--spot-file",
                    spot_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                ],
            )

            assert result.exit_code == 0
            assert "Delta-Hedge Backtest Results" in result.stdout
            assert "Strategy" in result.stdout
            assert "Total P&L" in result.stdout

        finally:
            Path(spot_file).unlink()

    def test_backtest_delta_hedge_put(self):
        """Test delta-hedge backtest for put option."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n102.0\n100.0\n")
            spot_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-delta-hedge",
                    "--spot-file",
                    spot_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "put",
                ],
            )

            assert result.exit_code == 0
            assert "Delta-Hedge Backtest Results" in result.stdout

        finally:
            Path(spot_file).unlink()

    def test_backtest_delta_hedge_with_dividend(self):
        """Test delta-hedge backtest with dividend yield."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n102.0\n100.0\n")
            spot_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-delta-hedge",
                    "--spot-file",
                    spot_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                    "--dividend",
                    "0.03",
                ],
            )

            assert result.exit_code == 0

        finally:
            Path(spot_file).unlink()

    def test_backtest_delta_hedge_invalid_file(self):
        """Test delta-hedge backtest with non-existent file."""
        result = runner.invoke(
            app,
            [
                "backtest-delta-hedge",
                "--spot-file",
                "nonexistent.csv",
                "--strike",
                "100.0",
                "--rate",
                "0.02",
                "--vol",
                "0.25",
                "--time",
                "0.25",
                "--type",
                "call",
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.stderr.lower() or "Error" in result.stderr

    def test_backtest_delta_hedge_invalid_strike(self):
        """Test delta-hedge backtest with invalid strike."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n")
            spot_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-delta-hedge",
                    "--spot-file",
                    spot_file,
                    "--strike",
                    "-100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                ],
            )

            assert result.exit_code != 0
            assert "strike" in result.stderr.lower()

        finally:
            Path(spot_file).unlink()

    def test_backtest_delta_hedge_invalid_type(self):
        """Test delta-hedge backtest with invalid option type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n")
            spot_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-delta-hedge",
                    "--spot-file",
                    spot_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "invalid",
                ],
            )

            assert result.exit_code != 0
            assert "option type" in result.stderr.lower() or "call" in result.stderr.lower()

        finally:
            Path(spot_file).unlink()

    def test_backtest_delta_hedge_too_few_prices(self):
        """Test delta-hedge backtest with only one price."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n")
            spot_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-delta-hedge",
                    "--spot-file",
                    spot_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                ],
            )

            assert result.exit_code != 0
            assert "at least 2" in result.stderr.lower() or "Error" in result.stderr

        finally:
            Path(spot_file).unlink()

    def test_backtest_delta_hedge_case_insensitive_type(self):
        """Test delta-hedge backtest with case-insensitive option type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n102.0\n100.0\n")
            spot_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-delta-hedge",
                    "--spot-file",
                    spot_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "CALL",  # Uppercase
                ],
            )

            assert result.exit_code == 0

        finally:
            Path(spot_file).unlink()


class TestMispricingCLI:
    """Test mispricing CLI command."""

    def test_backtest_mispricing_success_black_scholes(self):
        """Test successful mispricing backtest with Black-Scholes."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n102.0\n100.0\n")
            spot_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("7.0\n7.5\n6.5\n8.0\n7.0\n")
            market_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-mispricing",
                    "--spot-file",
                    spot_file,
                    "--market-price-file",
                    market_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                    "--model",
                    "black_scholes",
                ],
            )

            assert result.exit_code == 0
            assert "Mispricing Backtest Results" in result.stdout
            assert "Strategy" in result.stdout
            assert "Model" in result.stdout

        finally:
            Path(spot_file).unlink()
            Path(market_file).unlink()

    def test_backtest_mispricing_binomial_model(self):
        """Test mispricing backtest with binomial model."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n102.0\n100.0\n")
            spot_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("7.0\n7.5\n6.5\n8.0\n7.0\n")
            market_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-mispricing",
                    "--spot-file",
                    spot_file,
                    "--market-price-file",
                    market_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                    "--model",
                    "binomial",
                ],
            )

            assert result.exit_code == 0
            assert "Mispricing Backtest Results" in result.stdout

        finally:
            Path(spot_file).unlink()
            Path(market_file).unlink()

    def test_backtest_mispricing_monte_carlo_model(self):
        """Test mispricing backtest with Monte Carlo model."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n102.0\n100.0\n")
            spot_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("7.0\n7.5\n6.5\n8.0\n7.0\n")
            market_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-mispricing",
                    "--spot-file",
                    spot_file,
                    "--market-price-file",
                    market_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                    "--model",
                    "monte_carlo",
                    "--simulations",
                    "10000",
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0
            assert "Mispricing Backtest Results" in result.stdout

        finally:
            Path(spot_file).unlink()
            Path(market_file).unlink()

    def test_backtest_mispricing_invalid_model(self):
        """Test mispricing backtest with invalid model."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n")
            spot_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("7.0\n7.5\n")
            market_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-mispricing",
                    "--spot-file",
                    spot_file,
                    "--market-price-file",
                    market_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                    "--model",
                    "invalid_model",
                ],
            )

            assert result.exit_code != 0
            assert "model" in result.stderr.lower()

        finally:
            Path(spot_file).unlink()
            Path(market_file).unlink()

    def test_backtest_mispricing_mismatched_lengths(self):
        """Test mispricing backtest with mismatched file lengths."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n")
            spot_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("7.0\n7.5\n")
            market_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-mispricing",
                    "--spot-file",
                    spot_file,
                    "--market-price-file",
                    market_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                ],
            )

            assert result.exit_code != 0
            assert "same length" in result.stderr.lower() or "Error" in result.stderr

        finally:
            Path(spot_file).unlink()
            Path(market_file).unlink()

    def test_backtest_mispricing_put(self):
        """Test mispricing backtest for put option."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n102.0\n100.0\n")
            spot_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("6.0\n5.5\n6.5\n5.0\n6.0\n")
            market_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-mispricing",
                    "--spot-file",
                    spot_file,
                    "--market-price-file",
                    market_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "put",
                ],
            )

            assert result.exit_code == 0

        finally:
            Path(spot_file).unlink()
            Path(market_file).unlink()

    def test_backtest_mispricing_with_threshold(self):
        """Test mispricing backtest with custom threshold."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n102.0\n100.0\n")
            spot_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("7.0\n7.5\n6.5\n8.0\n7.0\n")
            market_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-mispricing",
                    "--spot-file",
                    spot_file,
                    "--market-price-file",
                    market_file,
                    "--strike",
                    "100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                    "--threshold",
                    "0.10",  # 10% threshold
                ],
            )

            assert result.exit_code == 0

        finally:
            Path(spot_file).unlink()
            Path(market_file).unlink()

    def test_backtest_mispricing_invalid_strike(self):
        """Test mispricing backtest with invalid strike."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n")
            spot_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("7.0\n7.5\n")
            market_file = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "backtest-mispricing",
                    "--spot-file",
                    spot_file,
                    "--market-price-file",
                    market_file,
                    "--strike",
                    "-100.0",
                    "--rate",
                    "0.02",
                    "--vol",
                    "0.25",
                    "--time",
                    "0.25",
                    "--type",
                    "call",
                ],
            )

            assert result.exit_code != 0
            assert "strike" in result.stderr.lower()

        finally:
            Path(spot_file).unlink()
            Path(market_file).unlink()
