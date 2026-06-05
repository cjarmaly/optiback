#!/usr/bin/env python3
"""
End-to-end delta-hedge walkthrough.

Fetches spot history (or uses synthetic prices), runs a delta-hedge backtest,
and plots the equity curve.

Usage:
    python examples/delta_hedge_walkthrough.py
    python examples/delta_hedge_walkthrough.py --synthetic
    python examples/delta_hedge_walkthrough.py --ticker SPY --output equity.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optiback.backtest import backtest_delta_hedge
from optiback.data import fetch_spot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delta-hedge backtest walkthrough")
    parser.add_argument("--ticker", default="SPY", help="Ticker to fetch (default: SPY)")
    parser.add_argument("--period", default="3mo", help="yfinance period (default: 3mo)")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic spot prices instead of fetching market data",
    )
    parser.add_argument(
        "--strike", type=float, default=None, help="Option strike (default: last spot)"
    )
    parser.add_argument("--rate", type=float, default=0.04, help="Risk-free rate")
    parser.add_argument("--vol", type=float, default=0.20, help="Implied volatility")
    parser.add_argument("--time", type=float, default=0.25, help="Time to expiry in years")
    parser.add_argument("--type", choices=["call", "put"], default="call", help="Option type")
    parser.add_argument(
        "--output",
        default="equity_curve.png",
        help="Path to save equity curve plot (default: equity_curve.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.synthetic:
        rng = np.random.default_rng(42)
        spot_prices = 100.0 + np.cumsum(rng.normal(0, 0.5, size=60))
        print("Using synthetic spot prices (60 periods)")
    else:
        print(f"Fetching {args.ticker} history (period={args.period})...")
        series = fetch_spot_history(args.ticker, period=args.period)
        spot_prices = series.to_numpy(dtype=float)
        print(f"Loaded {len(spot_prices)} prices")

    strike = args.strike if args.strike is not None else float(spot_prices[-1])
    print(f"Running delta-hedge backtest: strike={strike:.2f}, vol={args.vol:.0%}")

    result = backtest_delta_hedge(
        spot_prices=spot_prices,
        strike=strike,
        rate=args.rate,
        vol=args.vol,
        time_to_expiry=args.time,
        option_type=args.type,
        rebalance_frequency="daily",
    )

    summary = result.summary()
    print("\n--- Backtest summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if result.equity_curve is None:
        raise RuntimeError("Expected equity curve in backtest result")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.equity_curve, linewidth=1.5)
    ax.set_title("Delta-Hedge Equity Curve")
    ax.set_xlabel("Period")
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(True, alpha=0.3)
    if result.sharpe_ratio is not None:
        ax.text(
            0.02,
            0.98,
            f"Sharpe: {result.sharpe_ratio:.2f}",
            transform=ax.transAxes,
            va="top",
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nEquity curve saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
