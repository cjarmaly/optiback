from __future__ import annotations

import typer

from optiback.backtest import backtest_delta_hedge, backtest_mispricing
from optiback.cli.helpers import (
    PRICING_MODELS,
    add_common_option_rows,
    console,
    display_backtest_result,
    load_prices_array,
    make_table,
    resolve_spot_prices,
    run_command,
    validate_non_negative,
    validate_option_type,
    validate_positive,
    validate_pricing_inputs,
    validate_simulations,
    validate_steps,
)
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


@app.callback()
def cli_callback() -> None:
    """OptiBack: Options pricing and backtesting toolkit."""


@app.command("version")
def version() -> None:
    console.print("[bold green]OptiBack[/] 0.1.0")


def _price_black_scholes(option_type: str, **kwargs: float) -> float:
    func = black_scholes_call if option_type == "call" else black_scholes_put
    return float(func(**kwargs))


def _price_binomial(option_type: str, steps: int, **kwargs: float) -> float:
    func = binomial_tree_call if option_type == "call" else binomial_tree_put
    return float(func(steps=steps, **kwargs))


def _price_monte_carlo(
    option_type: str,
    simulations: int,
    seed: int | None,
    **kwargs: float,
) -> float:
    func = monte_carlo_call if option_type == "call" else monte_carlo_put
    return float(func(simulations=simulations, seed=seed, **kwargs))


@app.command("price")
def price(
    spot: float = typer.Option(..., help="Current spot price"),
    strike: float = typer.Option(..., help="Strike price"),
    rate: float = typer.Option(..., help="Risk-free rate (annualized)"),
    vol: float = typer.Option(..., help="Volatility (annualized)"),
    time: float = typer.Option(..., help="Time to expiry in years"),
    type: str = typer.Option(..., help="Option type: call or put"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized)"),
) -> None:
    """Price a European option with Black-Scholes."""

    def action() -> None:
        spot_v, strike_v, vol_v, time_v, option_type, dividend_v = validate_pricing_inputs(
            spot, strike, vol, time, type, dividend
        )
        price_value = _price_black_scholes(
            option_type,
            spot=spot_v,
            strike=strike_v,
            rate=rate,
            vol=vol_v,
            time_to_expiry=time_v,
            dividend_yield=dividend_v,
        )
        table = make_table("Option Pricing Results")
        add_common_option_rows(
            table,
            option_type=option_type,
            spot=spot_v,
            strike=strike_v,
            rate=rate,
            vol=vol_v,
            time=time_v,
            dividend=dividend_v,
        )
        table.add_row("", "")
        table.add_row("[bold]Option Price[/]", f"[bold green]{price_value:.4f}[/]")
        console.print(table)

    run_command(action)


@app.command("greeks")
def greeks(
    spot: float = typer.Option(..., help="Current spot price"),
    strike: float = typer.Option(..., help="Strike price"),
    rate: float = typer.Option(..., help="Risk-free rate (annualized)"),
    vol: float = typer.Option(..., help="Volatility (annualized)"),
    time: float = typer.Option(..., help="Time to expiry in years"),
    type: str = typer.Option(..., help="Option type: call or put"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized)"),
) -> None:
    """Calculate Black-Scholes Greeks."""

    def action() -> None:
        spot_v, strike_v, vol_v, time_v, option_type, dividend_v = validate_pricing_inputs(
            spot, strike, vol, time, type, dividend
        )
        values = black_scholes_greeks(
            spot=spot_v,
            strike=strike_v,
            rate=rate,
            vol=vol_v,
            time_to_expiry=time_v,
            option_type=option_type,
            dividend_yield=dividend_v,
        )
        table = make_table("Option Greeks")
        add_common_option_rows(
            table,
            option_type=option_type,
            spot=spot_v,
            strike=strike_v,
            rate=rate,
            vol=vol_v,
            time=time_v,
            dividend=dividend_v,
        )
        table.add_row("", "")
        labels = {
            "delta": "Delta (Δ)",
            "gamma": "Gamma (Γ)",
            "vega": "Vega (ν)",
            "theta": "Theta (Θ)",
            "rho": "Rho (ρ)",
        }
        for key, label in labels.items():
            table.add_row(f"[bold]{label}[/]", f"[bold green]{values[key]:.4f}[/]")
        console.print(table)

    run_command(action)


@app.command("implied-vol")
def implied_vol(
    spot: float = typer.Option(..., help="Current spot price"),
    strike: float = typer.Option(..., help="Strike price"),
    rate: float = typer.Option(..., help="Risk-free rate (annualized)"),
    time: float = typer.Option(..., help="Time to expiry in years"),
    price: float = typer.Option(..., help="Observed market price"),
    type: str = typer.Option(..., help="Option type: call or put"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized)"),
) -> None:
    """Infer implied volatility from a market price."""

    def action() -> None:
        spot_v = validate_positive(spot, "Spot price")
        strike_v = validate_positive(strike, "Strike price")
        time_v = validate_non_negative(time, "Time to expiry")
        market_price = validate_non_negative(price, "Market price")
        option_type = validate_option_type(type)
        dividend_v = validate_non_negative(dividend, "Dividend yield")

        implied = black_scholes_implied_volatility(
            spot=spot_v,
            strike=strike_v,
            rate=rate,
            time_to_expiry=time_v,
            market_price=market_price,
            option_type=option_type,
            dividend_yield=dividend_v,
        )
        table = make_table("Implied Volatility")
        add_common_option_rows(
            table,
            option_type=option_type,
            spot=spot_v,
            strike=strike_v,
            rate=rate,
            vol=None,
            time=time_v,
            dividend=dividend_v,
        )
        table.add_row("Market Price", f"{market_price:.4f}")
        table.add_row("", "")
        table.add_row(
            "[bold]Implied Volatility[/]",
            f"[bold green]{implied:.4f}[/] ({implied:.2%})",
        )
        console.print(table)

    run_command(action)


@app.command("price-binomial")
def price_binomial(
    spot: float = typer.Option(..., help="Current spot price"),
    strike: float = typer.Option(..., help="Strike price"),
    rate: float = typer.Option(..., help="Risk-free rate (annualized)"),
    vol: float = typer.Option(..., help="Volatility (annualized)"),
    time: float = typer.Option(..., help="Time to expiry in years"),
    type: str = typer.Option(..., help="Option type: call or put"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized)"),
    steps: int = typer.Option(100, help="Binomial tree steps"),
) -> None:
    """Price an American option with a binomial tree."""

    def action() -> None:
        spot_v, strike_v, vol_v, time_v, option_type, dividend_v = validate_pricing_inputs(
            spot, strike, vol, time, type, dividend
        )
        steps_v = validate_steps(steps)
        price_value = _price_binomial(
            option_type,
            steps_v,
            spot=spot_v,
            strike=strike_v,
            rate=rate,
            vol=vol_v,
            time_to_expiry=time_v,
            dividend_yield=dividend_v,
        )
        table = make_table("Binomial Tree Option Pricing Results")
        add_common_option_rows(
            table,
            option_type=option_type,
            spot=spot_v,
            strike=strike_v,
            rate=rate,
            vol=vol_v,
            time=time_v,
            dividend=dividend_v,
        )
        table.add_row("Steps", str(steps_v))
        table.add_row("", "")
        table.add_row("[bold]Option Price[/]", f"[bold green]{price_value:.4f}[/]")
        console.print(table)

    run_command(action)


@app.command("price-montecarlo")
def price_montecarlo(
    spot: float = typer.Option(..., help="Current spot price"),
    strike: float = typer.Option(..., help="Strike price"),
    rate: float = typer.Option(..., help="Risk-free rate (annualized)"),
    vol: float = typer.Option(..., help="Volatility (annualized)"),
    time: float = typer.Option(..., help="Time to expiry in years"),
    type: str = typer.Option(..., help="Option type: call or put"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized)"),
    simulations: int = typer.Option(100_000, help="Number of simulations"),
    seed: int | None = typer.Option(None, help="Random seed"),
) -> None:
    """Price a European option with Monte Carlo simulation."""

    def action() -> None:
        spot_v, strike_v, vol_v, time_v, option_type, dividend_v = validate_pricing_inputs(
            spot, strike, vol, time, type, dividend
        )
        simulations_v = validate_simulations(simulations)
        price_value = _price_monte_carlo(
            option_type,
            simulations_v,
            seed,
            spot=spot_v,
            strike=strike_v,
            rate=rate,
            vol=vol_v,
            time_to_expiry=time_v,
            dividend_yield=dividend_v,
        )
        table = make_table("Monte Carlo Option Pricing Results")
        add_common_option_rows(
            table,
            option_type=option_type,
            spot=spot_v,
            strike=strike_v,
            rate=rate,
            vol=vol_v,
            time=time_v,
            dividend=dividend_v,
        )
        table.add_row("Simulations", f"{simulations_v:,}")
        if seed is not None:
            table.add_row("Seed", str(seed))
        table.add_row("", "")
        table.add_row("[bold]Option Price[/]", f"[bold green]{price_value:.4f}[/]")
        console.print(table)

    run_command(action)


@app.command("backtest-delta-hedge")
def backtest_delta_hedge_cli(
    spot_file: str | None = typer.Option(None, help="Spot price file (CSV, Parquet, or text)"),
    ticker: str | None = typer.Option(None, help="Ticker symbol (alternative to --spot-file)"),
    start: str | None = typer.Option(None, help="Start date for --ticker (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, help="End date for --ticker (YYYY-MM-DD)"),
    period: str | None = typer.Option("3mo", help="yfinance period for --ticker"),
    strike: float = typer.Option(..., help="Strike price"),
    rate: float = typer.Option(..., help="Risk-free rate (annualized)"),
    vol: float = typer.Option(..., help="Volatility (annualized)"),
    time: float = typer.Option(..., help="Time to expiry in years"),
    type: str = typer.Option(..., help="Option type: call or put"),
    option_position: float = typer.Option(-1.0, help="Option position (negative = short)"),
    rebalance_frequency: str = typer.Option("daily", help="daily, weekly, or monthly"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized)"),
    cost_per_share: float = typer.Option(0.01, help="Transaction cost per share"),
    spread_bps: float = typer.Option(5.0, help="Bid-ask spread in basis points"),
    slippage_bps: float = typer.Option(5.0, help="Slippage in basis points"),
    impact_factor: float = typer.Option(0.1, help="Market impact factor"),
    output_csv: str | None = typer.Option(None, help="Save equity curve to CSV"),
) -> None:
    """Backtest a delta-hedged option position."""

    def action() -> None:
        option_type = validate_option_type(type)
        fetch_period = period if start is None or end is None else None
        spot_prices = resolve_spot_prices(spot_file, ticker, start, end, fetch_period, min_prices=2)
        result = backtest_delta_hedge(
            spot_prices=spot_prices,
            strike=validate_positive(strike, "Strike price"),
            rate=rate,
            vol=validate_non_negative(vol, "Volatility"),
            time_to_expiry=validate_non_negative(time, "Time to expiry"),
            option_type=option_type,
            option_position=option_position,
            rebalance_frequency=rebalance_frequency,
            dividend_yield=validate_non_negative(dividend, "Dividend yield"),
            cost_per_share=validate_non_negative(cost_per_share, "Cost per share"),
            bid_ask_spread_bps=validate_non_negative(spread_bps, "Spread"),
            slippage_bps=validate_non_negative(slippage_bps, "Slippage"),
            impact_factor=validate_non_negative(impact_factor, "Impact factor"),
        )
        display_backtest_result(
            result,
            title="Delta-Hedge Backtest Results",
            extra_rows=[("Rebalance Freq", rebalance_frequency.title())],
            output_csv=output_csv,
        )

    run_command(action)


@app.command("backtest-mispricing")
def backtest_mispricing_cli(
    spot_file: str | None = typer.Option(None, help="Spot price file (CSV, Parquet, or text)"),
    ticker: str | None = typer.Option(None, help="Ticker symbol (alternative to --spot-file)"),
    start: str | None = typer.Option(None, help="Start date for --ticker (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, help="End date for --ticker (YYYY-MM-DD)"),
    period: str | None = typer.Option("3mo", help="yfinance period for --ticker"),
    market_price_file: str = typer.Option(..., help="Market option price file"),
    strike: float = typer.Option(..., help="Strike price"),
    rate: float = typer.Option(..., help="Risk-free rate (annualized)"),
    vol: float = typer.Option(..., help="Volatility (annualized)"),
    time: float = typer.Option(..., help="Time to expiry in years"),
    type: str = typer.Option(..., help="Option type: call or put"),
    model: str = typer.Option("black_scholes", help="black_scholes, binomial, or monte_carlo"),
    threshold: float = typer.Option(0.05, help="Mispricing threshold (decimal)"),
    dividend: float = typer.Option(0.0, help="Dividend yield (annualized)"),
    cost_per_share: float = typer.Option(0.01, help="Transaction cost per share"),
    spread_bps: float = typer.Option(5.0, help="Bid-ask spread in basis points"),
    slippage_bps: float = typer.Option(5.0, help="Slippage in basis points"),
    impact_factor: float = typer.Option(0.1, help="Market impact factor"),
    steps: int = typer.Option(100, help="Binomial steps (binomial model only)"),
    simulations: int = typer.Option(100_000, help="Monte Carlo paths (monte_carlo model only)"),
    seed: int | None = typer.Option(None, help="Monte Carlo seed"),
    output_csv: str | None = typer.Option(None, help="Save equity curve to CSV"),
) -> None:
    """Backtest trading on theoretical vs market mispricing."""

    def action() -> None:
        if model not in PRICING_MODELS:
            raise typer.BadParameter(
                f"Model must be one of {sorted(PRICING_MODELS)}, got '{model}'"
            )

        option_type = validate_option_type(type)
        fetch_period = period if start is None or end is None else None
        spot_prices = resolve_spot_prices(spot_file, ticker, start, end, fetch_period, min_prices=1)
        market_prices = load_prices_array(market_price_file)
        if len(spot_prices) != len(market_prices):
            raise typer.BadParameter("Spot prices and market prices must have the same length")

        result = backtest_mispricing(
            spot_prices=spot_prices,
            market_option_prices=market_prices,
            strike=validate_positive(strike, "Strike price"),
            rate=rate,
            vol=validate_non_negative(vol, "Volatility"),
            time_to_expiry=validate_non_negative(time, "Time to expiry"),
            option_type=option_type,
            theoretical_model=model,
            dividend_yield=validate_non_negative(dividend, "Dividend yield"),
            mispricing_threshold=validate_non_negative(threshold, "Threshold"),
            cost_per_share=validate_non_negative(cost_per_share, "Cost per share"),
            bid_ask_spread_bps=validate_non_negative(spread_bps, "Spread"),
            slippage_bps=validate_non_negative(slippage_bps, "Slippage"),
            impact_factor=validate_non_negative(impact_factor, "Impact factor"),
            simulations=validate_simulations(simulations),
            seed=seed,
            steps=validate_steps(steps),
        )
        display_backtest_result(
            result,
            title="Mispricing Backtest Results",
            extra_rows=[("Model", model.replace("_", "-").title())],
            output_csv=output_csv,
        )

    run_command(action)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
