from __future__ import annotations

import typer
from rich.console import Console

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


def main():
    app()


if __name__ == "__main__":
    main()
