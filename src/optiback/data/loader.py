"""Load and save price series from files or market data providers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_PRICE_COLUMNS = ("Close", "close", "price", "Price", "adj close", "Adj Close")


def load_prices(path: str | Path, column: str | None = None) -> pd.Series:
    """Load a price series from CSV, text, or Parquet."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return _extract_price_series(pd.read_parquet(file_path), column).dropna()

    if suffix in {".csv", ".txt"}:
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] == 1:
                first_col = str(df.columns[0])
                try:
                    float(first_col)
                    df = pd.read_csv(file_path, header=None)
                except ValueError:
                    pass
                if df.shape[1] == 1:
                    series = df.iloc[:, 0].astype(float).dropna()
                    series.name = column or "price"
                    return series
            return _extract_price_series(df, column).dropna()
        except (pd.errors.ParserError, ValueError):
            values = np.loadtxt(file_path, delimiter=",", dtype=float).ravel()
            return pd.Series(values, name=column or "price").dropna()

    raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .txt, or .parquet")


def _extract_price_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is not None:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
        return df[column].astype(float)

    for candidate in _PRICE_COLUMNS:
        if candidate in df.columns:
            return df[candidate].astype(float)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric price column found in file")
    return df[numeric_cols[0]].astype(float)


def save_prices(
    prices: pd.Series | np.ndarray,
    path: str | Path,
    *,
    column: str = "price",
) -> None:
    """Save a price series to CSV or Parquet."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    series = pd.Series(prices, name=column) if isinstance(prices, np.ndarray) else prices
    if series.name is None:
        series = series.rename(column)

    if suffix == ".parquet":
        series.to_frame().to_parquet(file_path)
    elif suffix == ".csv":
        series.to_csv(file_path)
    else:
        raise ValueError(f"Unsupported output format: {suffix}. Use .csv or .parquet")


def fetch_spot_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    period: str | None = None,
) -> pd.Series:
    """Fetch historical close prices for a ticker via yfinance."""
    import yfinance as yf

    ticker_obj = yf.Ticker(ticker)
    if start is not None and end is not None:
        history = ticker_obj.history(start=start, end=end)
    else:
        history = ticker_obj.history(period=period or "3mo")

    if history.empty or "Close" not in history.columns:
        raise ValueError(f"No price data returned for ticker '{ticker}'")

    closes = history["Close"].astype(float).dropna()
    if closes.empty:
        raise ValueError(f"No valid close prices returned for ticker '{ticker}'")
    return closes
