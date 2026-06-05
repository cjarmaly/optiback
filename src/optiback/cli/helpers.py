"""Shared CLI validation, formatting, and error handling."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from optiback.backtest.engine import BacktestResult
from optiback.data import fetch_spot_history, load_prices, save_prices

T = TypeVar("T")

console = Console()

PRICING_MODELS = frozenset({"black_scholes", "binomial", "monte_carlo"})


def validate_positive(value: float, name: str) -> float:
    if value <= 0:
        raise typer.BadParameter(f"{name} must be greater than 0, got {value}")
    return value


def validate_non_negative(value: float, name: str) -> float:
    if value < 0:
        raise typer.BadParameter(f"{name} must be >= 0, got {value}")
    return value


def validate_option_type(option_type: str) -> str:
    normalized = option_type.lower()
    if normalized not in ("call", "put"):
        raise typer.BadParameter(f"Option type must be 'call' or 'put', got '{option_type}'")
    return normalized


def validate_steps(steps: int) -> int:
    if steps <= 0:
        raise typer.BadParameter(f"steps must be greater than 0, got {steps}")
    return steps


def validate_simulations(simulations: int) -> int:
    if simulations <= 0:
        raise typer.BadParameter(f"simulations must be greater than 0, got {simulations}")
    return simulations


def validate_pricing_inputs(
    spot: float,
    strike: float,
    vol: float,
    time: float,
    option_type: str,
    dividend: float = 0.0,
) -> tuple[float, float, float, float, str, float]:
    """Validate the common pricing command inputs."""
    return (
        validate_positive(spot, "Spot price"),
        validate_positive(strike, "Strike price"),
        validate_non_negative(vol, "Volatility"),
        validate_non_negative(time, "Time to expiry"),
        validate_option_type(option_type),
        validate_non_negative(dividend, "Dividend yield"),
    )


def run_command(action: Callable[[], T]) -> T:
    """Run a CLI command and surface user-facing errors through Typer."""
    try:
        return action()
    except typer.Exit:
        raise
    except typer.BadParameter:
        raise
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc


def make_table(title: str) -> Table:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green", justify="right")
    return table


def add_common_option_rows(
    table: Table,
    *,
    option_type: str,
    spot: float,
    strike: float,
    rate: float,
    vol: float | None,
    time: float,
    dividend: float,
) -> None:
    table.add_row("Option Type", option_type.capitalize())
    table.add_row("Spot Price", f"{spot:.2f}")
    table.add_row("Strike Price", f"{strike:.2f}")
    table.add_row("Risk-Free Rate", f"{rate:.2%}")
    if vol is not None:
        table.add_row("Volatility", f"{vol:.2%}")
    table.add_row("Time to Expiry", f"{time:.4f} years")
    if dividend > 0:
        table.add_row("Dividend Yield", f"{dividend:.2%}")


def resolve_spot_prices(
    spot_file: str | None,
    ticker: str | None,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    *,
    min_prices: int = 1,
) -> np.ndarray:
    if spot_file and ticker:
        raise typer.BadParameter("Provide either --spot-file or --ticker, not both")
    if not spot_file and not ticker:
        raise typer.BadParameter("Provide either --spot-file or --ticker")

    try:
        if spot_file:
            series = load_prices(spot_file)
        else:
            assert ticker is not None
            series = fetch_spot_history(ticker, start=start, end=end, period=period)
        prices = np.asarray(series.to_numpy(dtype=float))
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if len(prices) < min_prices:
        raise typer.BadParameter(f"Spot prices must contain at least {min_prices} values")
    return prices


def load_prices_array(file_path: str) -> np.ndarray:
    try:
        return np.asarray(load_prices(file_path).to_numpy(dtype=float))
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def display_backtest_result(
    result: BacktestResult,
    title: str,
    extra_rows: list[tuple[str, str]] | None = None,
    output_csv: str | None = None,
) -> None:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Strategy", result.strategy_type.replace("_", "-").title())
    for label, value in extra_rows or []:
        table.add_row(label, value)
    table.add_row("Initial Value", f"${result.initial_value:.2f}")
    table.add_row("Final Value", f"${result.final_value:.2f}")
    table.add_row("Total P&L", f"${result.total_pnl:.2f}")
    table.add_row("Transaction Costs", f"${result.transaction_costs:.2f}")
    table.add_row("Slippage Costs", f"${result.slippage_costs:.2f}")
    table.add_row(
        "Net P&L",
        f"${result.total_pnl - result.transaction_costs - result.slippage_costs:.2f}",
    )
    table.add_row("Number of Trades", f"{result.num_trades}")
    table.add_row("Return", f"{result.returns:.2%}")
    if result.sharpe_ratio is not None:
        table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.4f}")

    console.print(table)

    if output_csv and result.equity_curve is not None:
        save_prices(result.equity_curve, output_csv, column="equity")
        console.print(f"[dim]Equity curve saved to {output_csv}[/]")
