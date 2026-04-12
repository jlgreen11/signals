"""CLI commands for the automation layer (signals auto ...)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from signals.automation.cash_overlay import CashOverlay
from signals.automation.insights_engine import InsightsEngine
from signals.automation.paper_runner import PaperTradeRunner
from signals.automation.signal_store import SignalStore
from signals.config import SETTINGS
from signals.data.storage import DataStore

auto_app = typer.Typer(
    no_args_is_help=True,
    help="Automation: daily signals, paper trading, and portfolio management",
)
console = Console()


def _make_engine(
    capital: float = 100_000.0,
    momentum_weight: float = 0.50,
    tsmom_weight: float = 0.30,
    pead_weight: float = 0.20,
) -> tuple[InsightsEngine, SignalStore, CashOverlay]:
    """Build the full automation stack."""
    data_store = DataStore(SETTINGS.data.dir)
    signal_store = SignalStore(db_path=str(data_store.db_path))
    overlay = CashOverlay(
        total_capital=capital,
        model_weights={
            "momentum": momentum_weight,
            "tsmom": tsmom_weight,
            "pead": pead_weight,
        },
    )
    engine = InsightsEngine(
        signal_store=signal_store,
        cash_overlay=overlay,
        data_store=data_store,
    )
    return engine, signal_store, overlay


@auto_app.command("daily")
def auto_daily(
    capital: float = typer.Option(100_000.0, "--capital"),
    momentum_weight: float = typer.Option(0.50, "--momentum-weight"),
    tsmom_weight: float = typer.Option(0.30, "--tsmom-weight"),
    pead_weight: float = typer.Option(0.20, "--pead-weight"),
) -> None:
    """Run daily signal generation and print the insights report (no trading)."""
    engine, _store, _overlay = _make_engine(
        capital, momentum_weight, tsmom_weight, pead_weight
    )
    report = engine.run_daily()
    console.print(report["report_text"])
    console.print(
        f"\n[dim]Signals stored: momentum={report['n_momentum_signals']}, "
        f"tsmom={report['n_tsmom_signals']}, pead={report['n_pead_signals']}[/dim]"
    )


@auto_app.command("trade")
def auto_trade(
    capital: float = typer.Option(100_000.0, "--capital"),
    momentum_weight: float = typer.Option(0.50, "--momentum-weight"),
    tsmom_weight: float = typer.Option(0.30, "--tsmom-weight"),
    pead_weight: float = typer.Option(0.20, "--pead-weight"),
    broker: str = typer.Option("paper", "--broker", help="paper or alpaca"),
) -> None:
    """Run daily signals and execute trades.

    --broker paper   (default) local PaperBroker with SQLite persistence
    --broker alpaca  Alpaca Trading API paper account (requires
                     ALPACA_API_KEY + ALPACA_SECRET_KEY env vars)
    """
    engine, _store, _overlay = _make_engine(
        capital, momentum_weight, tsmom_weight, pead_weight
    )
    runner = PaperTradeRunner(engine=engine, initial_capital=capital, broker=broker)
    result = runner.execute_daily()

    console.print(result["report"]["report_text"])
    console.print()

    if result["executed_orders"]:
        table = Table(title="Executed Paper Trades")
        table.add_column("Ticker")
        table.add_column("Action")
        table.add_column("Shares")
        table.add_column("Price")
        table.add_column("Notional")
        for order in result["executed_orders"]:
            table.add_row(
                order["ticker"],
                order["action"],
                f"{order['shares']:.4f}",
                f"${order['price']:,.2f}",
                f"${order['notional']:,.2f}",
            )
        console.print(table)
    else:
        console.print("[yellow]No trades executed (no price data available)[/yellow]")

    console.print(f"\n[bold]Equity:[/bold] ${result['equity']:,.2f}")
    console.print(f"[bold]Cash:[/bold] ${result['cash']:,.2f}")


@auto_app.command("positions")
def auto_positions(
    capital: float = typer.Option(100_000.0, "--capital"),
    broker: str = typer.Option("paper", "--broker", help="paper or alpaca"),
) -> None:
    """View current positions."""
    engine, _store, _overlay = _make_engine(capital)
    runner = PaperTradeRunner(engine=engine, initial_capital=capital, broker=broker)

    positions = runner.get_positions()
    if positions.empty:
        console.print("[yellow]No open positions[/yellow]")
        return

    table = Table(title="Paper Portfolio Positions")
    for col in positions.columns:
        table.add_column(col)
    for _, row in positions.iterrows():
        table.add_row(*[f"{v:,.4f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)


@auto_app.command("performance")
def auto_performance(
    capital: float = typer.Option(100_000.0, "--capital"),
    broker: str = typer.Option("paper", "--broker", help="paper or alpaca"),
) -> None:
    """View trading performance metrics."""
    engine, _store, _overlay = _make_engine(capital)
    runner = PaperTradeRunner(engine=engine, initial_capital=capital, broker=broker)

    perf = runner.get_performance()
    table = Table(title="Paper Trading Performance")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total Return", f"{perf['total_return'] * 100:.2f}%")
    table.add_row("Sharpe Ratio", f"{perf['sharpe']:.2f}")
    table.add_row("Max Drawdown", f"{perf['max_drawdown'] * 100:.2f}%")
    table.add_row("Trading Days", str(perf["n_days"]))
    table.add_row("Current Equity", f"${perf['current_equity']:,.2f}")
    console.print(table)


@auto_app.command("history")
def auto_history(
    ticker: str,
    days: int = typer.Option(30, "--days"),
    model: str | None = typer.Option(None, "--model"),
) -> None:
    """View signal history for a ticker."""
    data_store = DataStore(SETTINGS.data.dir)
    store = SignalStore(db_path=str(data_store.db_path))

    df = store.get_signal_history(ticker, model=model, days=days)
    if df.empty:
        console.print(f"[yellow]No signals found for {ticker}[/yellow]")
        return

    table = Table(title=f"Signal History -- {ticker} (last {days} days)")
    for col in df.columns:
        table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


@auto_app.command("config")
def auto_config(
    capital: float = typer.Option(100_000.0, "--capital"),
    momentum_weight: float = typer.Option(0.50, "--momentum-weight"),
    tsmom_weight: float = typer.Option(0.30, "--tsmom-weight"),
    pead_weight: float = typer.Option(0.20, "--pead-weight"),
) -> None:
    """Display the cash overlay configuration."""
    overlay = CashOverlay(
        total_capital=capital,
        model_weights={
            "momentum": momentum_weight,
            "tsmom": tsmom_weight,
            "pead": pead_weight,
        },
    )
    console.print(overlay.summary())


@auto_app.command("weekly")
def auto_weekly(
    capital: float = typer.Option(100_000.0, "--capital"),
    momentum_weight: float = typer.Option(0.50, "--momentum-weight"),
    tsmom_weight: float = typer.Option(0.30, "--tsmom-weight"),
    pead_weight: float = typer.Option(0.20, "--pead-weight"),
) -> None:
    """Run weekly deep analysis."""
    engine, _store, _overlay = _make_engine(
        capital, momentum_weight, tsmom_weight, pead_weight
    )
    report = engine.run_weekly()
    console.print(report["report_text"])
    console.print(
        f"\n[bold]Weekly summary:[/bold] "
        f"{report.get('weekly_signal_count', 0)} signals from "
        f"{report.get('weekly_models_active', 0)} models in last 7 days"
    )
