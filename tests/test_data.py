"""Tests for data loading and fetching."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from optiback.data import fetch_spot_history, load_prices, save_prices


class TestLoadPrices:
    def test_load_single_column_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n101.0\n99.0\n")
            path = f.name

        try:
            series = load_prices(path)
            assert len(series) == 3
            np.testing.assert_allclose(series.values, [100.0, 101.0, 99.0])
        finally:
            Path(path).unlink()

    def test_load_prices_drops_nan(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("100.0\n\n102.0\n")
            path = f.name

        try:
            series = load_prices(path)
            np.testing.assert_allclose(series.values, [100.0, 102.0])
        finally:
            Path(path).unlink()

    def test_load_csv_with_close_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date,Close\n2024-01-01,100.0\n2024-01-02,101.0\n")
            path = f.name

        try:
            series = load_prices(path)
            np.testing.assert_allclose(series.values, [100.0, 101.0])
        finally:
            Path(path).unlink()

    def test_load_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prices.parquet"
            pd.Series([100.0, 101.0, 102.0], name="price").to_frame().to_parquet(path)
            series = load_prices(path)
            np.testing.assert_allclose(series.values, [100.0, 101.0, 102.0])

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_prices("/nonexistent/prices.csv")

    def test_load_unsupported_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_prices(path)
        finally:
            Path(path).unlink()


class TestSavePrices:
    def test_save_and_load_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            save_prices(np.array([1.0, 2.0, 3.0]), path)
            loaded = load_prices(path)
            np.testing.assert_allclose(loaded.values, [1.0, 2.0, 3.0])

    def test_save_and_load_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.parquet"
            save_prices(pd.Series([4.0, 5.0]), path)
            loaded = load_prices(path)
            np.testing.assert_allclose(loaded.values, [4.0, 5.0])


class TestFetchSpotHistory:
    @patch("yfinance.Ticker")
    def test_fetch_spot_history_with_period(self, mock_ticker_cls):
        mock_history = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        )
        mock_ticker_cls.return_value.history.return_value = mock_history

        series = fetch_spot_history("SPY", period="1mo")
        mock_ticker_cls.assert_called_once_with("SPY")
        mock_ticker_cls.return_value.history.assert_called_once_with(period="1mo")
        np.testing.assert_allclose(series.values, [100.0, 101.0, 102.0])

    @patch("yfinance.Ticker")
    def test_fetch_spot_history_with_dates(self, mock_ticker_cls):
        mock_history = pd.DataFrame(
            {"Close": [200.0]},
            index=pd.to_datetime(["2024-06-01"]),
        )
        mock_ticker_cls.return_value.history.return_value = mock_history

        series = fetch_spot_history("AAPL", start="2024-01-01", end="2024-06-01")
        mock_ticker_cls.return_value.history.assert_called_once_with(
            start="2024-01-01", end="2024-06-01"
        )
        assert series.iloc[0] == 200.0

    @patch("yfinance.Ticker")
    def test_fetch_spot_history_empty(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No price data"):
            fetch_spot_history("INVALID")

    @patch("yfinance.Ticker")
    def test_fetch_spot_history_drops_nan(self, mock_ticker_cls):
        mock_history = pd.DataFrame(
            {"Close": [100.0, np.nan, 102.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        )
        mock_ticker_cls.return_value.history.return_value = mock_history

        series = fetch_spot_history("SPY", period="1mo")
        np.testing.assert_allclose(series.values, [100.0, 102.0])

    @patch("yfinance.Ticker")
    def test_fetch_spot_history_all_nan_raises(self, mock_ticker_cls):
        mock_history = pd.DataFrame(
            {"Close": [np.nan, np.nan]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        mock_ticker_cls.return_value.history.return_value = mock_history

        with pytest.raises(ValueError, match="No valid close prices"):
            fetch_spot_history("SPY", period="1mo")
