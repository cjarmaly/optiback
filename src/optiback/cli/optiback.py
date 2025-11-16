from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from optiback.pricing import (
    binomial_tree_call,
    binomial_tree_put,
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_implied_volatility,
    black_scholes_put,
    monte_carlo_call,
    monte_carlo_put,
)

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


def validate_steps(steps: int) -> int:
    """Validate that steps is a positive integer."""
    if steps <= 0:
        raise typer.BadParameter(f"steps must be greater than 0, got {steps}")
    return steps


def validate_simulations(simulations: int) -> int:
    """Validate that simulations is a positive integer."""
    if simulations <= 0:
        raise typer.BadParameter(f"simulations must be greater than 0, got {simulations}")
    return simulations


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


@app.command("greeks")
def greeks(
    spot: float = typer.Option(..., help="Current spot price of the underlying asset"),
    strike: float = typer.Option(..., help="Strike price of the option"),
    rate: float = typer.Option(..., help="Risk-free interest rate (annualized)"),
    vol: float = typer.Option(..., help="Volatility of the underlying asset (annualized)"),
    time: float = typer.Option(..., help="Time to expiration in years"),
    type: str = typer.Option(..., help="Option type: 'call' or 'put'"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized, default: 0.0)"),
) -> None:
    """
    Calculate all Greeks (Delta, Gamma, Vega, Theta, Rho) for an option.

    Examples:
        optiback greeks --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call
        optiback greeks --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type put --dividend 0.01
    """
    # Validate inputs
    spot = validate_positive(spot, "Spot price")
    strike = validate_positive(strike, "Strike price")
    vol = validate_non_negative(vol, "Volatility")
    time = validate_non_negative(time, "Time to expiry")
    type = validate_option_type(type)
    dividend = validate_non_negative(dividend, "Dividend yield")

    # Calculate all Greeks
    try:
        greeks_dict = black_scholes_greeks(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            time_to_expiry=time,
            option_type=type,
            dividend_yield=dividend,
        )

        # Display results in a formatted table
        table = Table(title="Option Greeks", show_header=True, header_style="bold magenta")
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
        table.add_row("[bold]Delta (Δ)[/]", f"[bold green]{greeks_dict['delta']:.4f}[/]")
        table.add_row("[bold]Gamma (Γ)[/]", f"[bold green]{greeks_dict['gamma']:.4f}[/]")
        table.add_row("[bold]Vega (ν)[/]", f"[bold green]{greeks_dict['vega']:.4f}[/]")
        table.add_row("[bold]Theta (Θ)[/]", f"[bold green]{greeks_dict['theta']:.4f}[/]")
        table.add_row("[bold]Rho (ρ)[/]", f"[bold green]{greeks_dict['rho']:.4f}[/]")

        console.print(table)

    except Exception as e:
        error_console = Console(file=sys.stderr)
        error_console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1) from e


@app.command("implied-vol")
def implied_vol(
    spot: float = typer.Option(..., help="Current spot price of the underlying asset"),
    strike: float = typer.Option(..., help="Strike price of the option"),
    rate: float = typer.Option(..., help="Risk-free interest rate (annualized)"),
    time: float = typer.Option(..., help="Time to expiration in years"),
    price: float = typer.Option(..., help="Observed market price of the option"),
    type: str = typer.Option(..., help="Option type: 'call' or 'put'"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized, default: 0.0)"),
) -> None:
    """
    Calculate implied volatility from market price using Black-Scholes model.

    Examples:
        optiback implied-vol --spot 100 --strike 100 --rate 0.02 --time 0.5 --price 7.5168 --type call
        optiback implied-vol --spot 100 --strike 100 --rate 0.02 --time 0.5 --price 6.5218 --type put --dividend 0.01
    """
    # Validate inputs
    spot = validate_positive(spot, "Spot price")
    strike = validate_positive(strike, "Strike price")
    time = validate_non_negative(time, "Time to expiry")
    price = validate_non_negative(price, "Market price")
    type = validate_option_type(type)
    dividend = validate_non_negative(dividend, "Dividend yield")

    # Calculate implied volatility
    try:
        implied_vol_value = black_scholes_implied_volatility(
            spot=spot,
            strike=strike,
            rate=rate,
            time_to_expiry=time,
            market_price=price,
            option_type=type,
            dividend_yield=dividend,
        )

        # Display results in a formatted table
        table = Table(title="Implied Volatility", show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Option Type", type.capitalize())
        table.add_row("Spot Price", f"{spot:.2f}")
        table.add_row("Strike Price", f"{strike:.2f}")
        table.add_row("Risk-Free Rate", f"{rate:.2%}")
        table.add_row("Time to Expiry", f"{time:.4f} years")
        if dividend > 0:
            table.add_row("Dividend Yield", f"{dividend:.2%}")
        table.add_row("Market Price", f"{price:.4f}")
        table.add_row("", "")  # Empty row for spacing
        table.add_row(
            "[bold]Implied Volatility[/]",
            f"[bold green]{implied_vol_value:.4f}[/] ({implied_vol_value:.2%})",
        )

        console.print(table)

    except Exception as e:
        error_console = Console(file=sys.stderr)
        error_console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1) from e


@app.command("price-binomial")
def price_binomial(
    spot: float = typer.Option(..., help="Current spot price of the underlying asset"),
    strike: float = typer.Option(..., help="Strike price of the option"),
    rate: float = typer.Option(..., help="Risk-free interest rate (annualized)"),
    vol: float = typer.Option(..., help="Volatility of the underlying asset (annualized)"),
    time: float = typer.Option(..., help="Time to expiration in years"),
    type: str = typer.Option(..., help="Option type: 'call' or 'put'"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized, default: 0.0)"),
    steps: int = typer.Option(100, help="Number of time steps in the binomial tree (default: 100)"),
) -> None:
    """
    Price an American option using the Binomial Tree model (CRR).

    Examples:
        optiback price-binomial --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call
        optiback price-binomial --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type put --dividend 0.01 --steps 200
    """
    # Validate inputs
    spot = validate_positive(spot, "Spot price")
    strike = validate_positive(strike, "Strike price")
    vol = validate_non_negative(vol, "Volatility")
    time = validate_non_negative(time, "Time to expiry")
    type = validate_option_type(type)
    dividend = validate_non_negative(dividend, "Dividend yield")
    steps = validate_steps(steps)

    # Calculate option price
    try:
        if type == "call":
            price_value = binomial_tree_call(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time,
                dividend_yield=dividend,
                steps=steps,
            )
        else:  # put
            price_value = binomial_tree_put(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time,
                dividend_yield=dividend,
                steps=steps,
            )

        # Display results in a formatted table
        table = Table(
            title="Binomial Tree Option Pricing Results",
            show_header=True,
            header_style="bold magenta",
        )
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
        table.add_row("Steps", f"{steps}")
        table.add_row("", "")  # Empty row for spacing
        table.add_row("[bold]Option Price[/]", f"[bold green]{price_value:.4f}[/]")

        console.print(table)

    except Exception as e:
        error_console = Console(file=sys.stderr)
        error_console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1) from e


@app.command("price-montecarlo")
def price_montecarlo(
    spot: float = typer.Option(..., help="Current spot price of the underlying asset"),
    strike: float = typer.Option(..., help="Strike price of the option"),
    rate: float = typer.Option(..., help="Risk-free interest rate (annualized)"),
    vol: float = typer.Option(..., help="Volatility of the underlying asset (annualized)"),
    time: float = typer.Option(..., help="Time to expiration in years"),
    type: str = typer.Option(..., help="Option type: 'call' or 'put'"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized, default: 0.0)"),
    simulations: int = typer.Option(
        100000, help="Number of Monte Carlo simulations (default: 100000)"
    ),
    seed: int | None = typer.Option(None, help="Random seed for reproducibility (optional)"),
) -> None:
    """
    Price a European option using Monte Carlo simulation.

    Examples:
        optiback price-montecarlo --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type call
        optiback price-montecarlo --spot 100 --strike 100 --rate 0.02 --vol 0.25 --time 0.5 --type put --simulations 200000 --seed 42
    """
    # Validate inputs
    spot = validate_positive(spot, "Spot price")
    strike = validate_positive(strike, "Strike price")
    vol = validate_non_negative(vol, "Volatility")
    time = validate_non_negative(time, "Time to expiry")
    type = validate_option_type(type)
    dividend = validate_non_negative(dividend, "Dividend yield")
    simulations = validate_simulations(simulations)

    # Calculate option price
    try:
        if type == "call":
            price_value = monte_carlo_call(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time,
                dividend_yield=dividend,
                simulations=simulations,
                seed=seed,
            )
        else:  # put
            price_value = monte_carlo_put(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time,
                dividend_yield=dividend,
                simulations=simulations,
                seed=seed,
            )

        # Display results in a formatted table
        table = Table(
            title="Monte Carlo Option Pricing Results",
            show_header=True,
            header_style="bold magenta",
        )
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
        table.add_row("Simulations", f"{simulations:,}")
        if seed is not None:
            table.add_row("Seed", f"{seed}")
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
