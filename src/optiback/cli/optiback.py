from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from optiback.pricing import black_scholes_call, black_scholes_put

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.callback()
def cli_callback():
    """
    OptiBack: Options pricing & backtesting toolkit.
    """


@app.command("version")
def version():
    console.print("[bold green]OptiBack[/] 0.1.0")


def validate_positive(value: float, param_name: str) -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise typer.BadParameter(f"{param_name} must be greater than 0, got {value}")
    return value


def validate_non_negative(value: float, param_name: str) -> float:
    """Validate that a value is non-negative."""
    if value < 0:
        raise typer.BadParameter(f"{param_name} must be >= 0, got {value}")
    return value


def validate_option_type(option_type: str) -> str:
    """Validate that option type is 'call' or 'put'."""
    option_type_lower = option_type.lower()
    if option_type_lower not in ("call", "put"):
        raise typer.BadParameter(f"Option type must be 'call' or 'put', got '{option_type}'")
    return option_type_lower


@app.command("price")
def price(
    spot: float = typer.Option(..., help="Current spot price of the underlying asset"),
    strike: float = typer.Option(..., help="Strike price of the option"),
    rate: float = typer.Option(..., help="Risk-free interest rate (annualized)"),
    vol: float = typer.Option(..., help="Volatility of the underlying asset (annualized)"),
    time: float = typer.Option(..., help="Time to expiration in years"),
    type: str = typer.Option(..., help="Option type: 'call' or 'put'"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized, default: 0.0)"),
) -> None:
    """
    Price an option using the Black-Scholes model.

    Examples:
        optiback price --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call
        optiback price --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type put --dividend 0.01
    """
    # Validate inputs
    spot = validate_positive(spot, "Spot price")
    strike = validate_positive(strike, "Strike price")
    vol = validate_non_negative(vol, "Volatility")
    time = validate_non_negative(time, "Time to expiry")
    type = validate_option_type(type)
    dividend = validate_non_negative(dividend, "Dividend yield")

    # Calculate option price
    try:
        if type == "call":
            price_value = black_scholes_call(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time,
                dividend_yield=dividend,
            )
        else:  # put
            price_value = black_scholes_put(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time,
                dividend_yield=dividend,
            )

        # Display results in a formatted table
        table = Table(title="Option Pricing Results", show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Option Type", type.capitalize())
        table.add_row("Spot Price", f"{spot:.2f}")
        table.add_row("Strike Price", f"{strike:.2f}")
        table.add_row("Risk-Free Rate", f"{rate:.2%}")
        table.add_row("Volatility", f"{vol:.2%}")
        table.add_row("Time to Expiry", f"{time:.4f} years")
        if dividend > 0:
            table.add_row("Dividend Yield", f"{dividend:.2%}")
        table.add_row("", "")  # Empty row for spacing
        table.add_row("[bold]Option Price[/]", f"[bold green]{price_value:.4f}[/]")

        console.print(table)

    except Exception as e:
        error_console = Console(file=sys.stderr)
        error_console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1) from e


def main():
    app()


if __name__ == "__main__":
    main()
