"""CLI commands for the automation layer (signals auto ...)."""

from __future__ import annotations

# Load .env before anything else touches os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os

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

# Account key mapping — each account reads its own env vars
ACCOUNT_KEYS = {
    "momentum": ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
    "multifactor": ("ALPACA_MULTIFACTOR_KEY", "ALPACA_MULTIFACTOR_SECRET"),
    "baseline": ("ALPACA_BASELINE_KEY", "ALPACA_BASELINE_SECRET"),
}


def _set_alpaca_keys(account: str) -> None:
    """Point the standard ALPACA_API_KEY/SECRET_KEY env vars at the
    right account's credentials. The AlpacaBroker reads the standard
    names, so we swap them before it initialises."""
    key_var, secret_var = ACCOUNT_KEYS.get(account, ACCOUNT_KEYS["momentum"])
    key = os.environ.get(key_var, "")
    secret = os.environ.get(secret_var, "")
    if not key or not secret:
        console.print(
            f"[red]Missing env vars {key_var} / {secret_var} for account '{account}'[/red]"
        )
        raise typer.Exit(1)
    os.environ["ALPACA_API_KEY"] = key
    os.environ["ALPACA_SECRET_KEY"] = secret


def _load_sp500_sectors() -> dict[str, str]:
    """Load GICS sector mapping for SP500 tickers.

    Checks three sources in order:
    1. Project-local cache at data/sp500_sectors.csv
    2. Legacy /tmp path
    3. GitHub fallback (cached to project-local on success)
    """
    from pathlib import Path

    import pandas as pd

    project_cache = Path("data/sp500_sectors.csv")
    legacy_path = Path("/tmp/sp500_with_sectors.csv")

    # 1. Project-local cache (survives /tmp cleanup)
    if project_cache.exists():
        df = pd.read_csv(project_cache)
        col = "GICS Sector" if "GICS Sector" in df.columns else "Sector"
        sym = "Symbol" if "Symbol" in df.columns else df.columns[0]
        return dict(zip(df[sym], df[col], strict=False))

    # 2. Legacy /tmp path
    if legacy_path.exists():
        df = pd.read_csv(legacy_path)
        col = "GICS Sector" if "GICS Sector" in df.columns else "Sector"
        sym = "Symbol" if "Symbol" in df.columns else df.columns[0]
        # Cache locally so it survives /tmp cleanup
        df.to_csv(project_cache, index=False)
        return dict(zip(df[sym], df[col], strict=False))

    # 3. GitHub fallback
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df = pd.read_csv(url)
        col = "GICS Sector" if "GICS Sector" in df.columns else "Sector"
        # Cache locally
        project_cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(project_cache, index=False)
        return dict(zip(df["Symbol"], df[col], strict=False))
    except Exception as e:
        console.print(f"[yellow]Warning: could not load S&P 500 sectors: {e}[/yellow]")
        return {}

    return {}


def _make_engine(
    capital: float = 100_000.0,
    account: str = "momentum",
) -> tuple[InsightsEngine, SignalStore, CashOverlay]:
    """Build the full automation stack.

    The account type determines the strategy mix:
      momentum    → 100% early-breakout momentum on SP500 (sector-diversified)
      multifactor → 100% multi-factor (momentum+value+quality+news filter)
      baseline    → not used (SPY B&H, no signals needed)
    """
    from signals.model.momentum import CrossSectionalMomentum

    data_store = DataStore(SETTINGS.data.dir)
    signal_store = SignalStore(db_path=str(data_store.db_path))
    sectors = _load_sp500_sectors()

    if account == "multifactor":
        overlay = CashOverlay(
            total_capital=capital,
            model_weights={"momentum": 1.0, "tsmom": 0.0, "pead": 0.0},
        )
        engine = InsightsEngine(
            signal_store=signal_store,
            cash_overlay=overlay,
            data_store=data_store,
            use_multifactor=True,
            sectors=sectors,
        )
    else:
        # Early breakout momentum — sector-diversified, acceleration-ranked
        overlay = CashOverlay(
            total_capital=capital,
            model_weights={"momentum": 1.0, "tsmom": 0.0, "pead": 0.0},
        )
        mom = CrossSectionalMomentum(
            mode="early_breakout",
            lookback_days=252,
            short_lookback=63,
            n_long=15,
            max_per_sector=2,
            max_12m_return=1.5,
            min_short_return=0.10,
            rebalance_freq=21,
        )
        # Use full universe (all tickers with data) — the 2026-04-21
        # evaluation showed full-universe lifts Sharpe from 0.659 to 1.075.
        # InsightsEngine._load_full_universe_tickers() discovers all
        # available tickers from data/raw/.
        engine = InsightsEngine(
            signal_store=signal_store,
            cash_overlay=overlay,
            data_store=data_store,
            tickers=None,  # triggers full-universe auto-discovery
            use_multifactor=False,
            momentum_model=mom,
            sectors=sectors,
        )
    return engine, signal_store, overlay


@auto_app.command("daily")
def auto_daily(
    capital: float = typer.Option(100_000.0, "--capital"),
    account: str = typer.Option("momentum", "--account",
                                help="momentum, multifactor, or baseline"),
) -> None:
    """Run daily signal generation and print the insights report (no trading)."""
    engine, _store, _overlay = _make_engine(capital, account)
    report = engine.run_daily()
    console.print(report["report_text"])


@auto_app.command("trade")
def auto_trade(
    capital: float = typer.Option(100_000.0, "--capital"),
    account: str = typer.Option("momentum", "--account",
                                help="momentum, multifactor, or baseline"),
) -> None:
    """Run daily signals and execute trades on Alpaca.

    --account momentum     Pure cross-sectional momentum, top-10 of SP500
    --account multifactor  Multi-factor (momentum+value+quality) + news filter
    --account baseline     Not applicable (SPY B&H, just sits)
    """
    if account == "baseline":
        console.print("[yellow]Baseline account is SPY B&H — no daily trades needed.[/yellow]")
        return

    _set_alpaca_keys(account)
    engine, _store, _overlay = _make_engine(capital, account)
    runner = PaperTradeRunner(engine=engine, initial_capital=capital, broker="alpaca")
    result = runner.execute_daily()

    console.print(result["report"]["report_text"])
    console.print()

    if result.get("rebalance"):
        console.print(
            f"[bold green]REBALANCE DAY[/bold green] — "
            f"{result['days_since_rebalance']} trading days since last rebalance"
        )
        if result["executed_orders"]:
            table = Table(title=f"Executed Trades — {account} account")
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
            console.print("[yellow]No trades needed (same top-10 as before)[/yellow]")
    else:
        next_in = result.get("next_rebalance_in", "?")
        console.print(
            f"[dim]No rebalance due — {result.get('days_since_rebalance', '?')}/21 "
            f"trading days since last. Next rebalance in ~{next_in} days.[/dim]"
        )

    console.print(f"\n[bold]Equity:[/bold] ${result['equity']:,.2f}")
    console.print(f"[bold]Cash:[/bold] ${result['cash']:,.2f}")


@auto_app.command("positions")
def auto_positions(
    account: str = typer.Option("momentum", "--account",
                                help="momentum, multifactor, or baseline"),
) -> None:
    """View current positions on an Alpaca account."""
    _set_alpaca_keys(account)

    from alpaca.trading.client import TradingClient
    client = TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )
    acct = client.get_account()
    positions = client.get_all_positions()

    table = Table(title=f"Positions — {account} account")
    table.add_column("Ticker")
    table.add_column("Shares", justify="right")
    table.add_column("Avg Price", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Market Value", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")

    for p in positions:
        pnl = float(p.unrealized_pl)
        pnl_pct = float(p.unrealized_plpc) * 100
        color = "green" if pnl >= 0 else "red"
        table.add_row(
            p.symbol,
            str(p.qty),
            f"${float(p.avg_entry_price):,.2f}",
            f"${float(p.current_price):,.2f}",
            f"${float(p.market_value):,.2f}",
            f"[{color}]${pnl:+,.2f}[/{color}]",
            f"[{color}]{pnl_pct:+.2f}%[/{color}]",
        )

    console.print(table)
    console.print(f"\n[bold]Cash:[/bold] ${float(acct.cash):,.2f}")
    console.print(f"[bold]Portfolio Value:[/bold] ${float(acct.portfolio_value):,.2f}")


@auto_app.command("performance")
def auto_performance(
    account: str = typer.Option("all", "--account",
                                help="momentum, multifactor, baseline, or all"),
) -> None:
    """View trading performance. Use --account all for side-by-side comparison."""
    accounts_to_show = (
        ["momentum", "multifactor", "baseline"] if account == "all"
        else [account]
    )

    from alpaca.trading.client import TradingClient

    table = Table(title="Performance Comparison")
    table.add_column("Account")
    table.add_column("Portfolio Value", justify="right")
    table.add_column("Cash", justify="right")
    table.add_column("Positions", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Return", justify="right")

    for acct_name in accounts_to_show:
        try:
            key_var, secret_var = ACCOUNT_KEYS[acct_name]
            key = os.environ.get(key_var, "")
            secret = os.environ.get(secret_var, "")
            if not key or not secret:
                table.add_row(acct_name, "—", "—", "—", "—", "[red]no keys[/red]")
                continue

            client = TradingClient(key, secret, paper=True)
            a = client.get_account()
            positions = client.get_all_positions()
            pv = float(a.portfolio_value)
            cash = float(a.cash)
            pnl = pv - 100_000
            ret = pnl / 100_000 * 100
            color = "green" if pnl >= 0 else "red"

            table.add_row(
                acct_name,
                f"${pv:,.2f}",
                f"${cash:,.2f}",
                str(len(positions)),
                f"[{color}]${pnl:+,.2f}[/{color}]",
                f"[{color}]{ret:+.2f}%[/{color}]",
            )
        except Exception as e:
            table.add_row(acct_name, "—", "—", "—", "—", f"[red]{e}[/red]")

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
def auto_config() -> None:
    """Display account configuration and connection status."""
    from alpaca.trading.client import TradingClient

    table = Table(title="Account Configuration")
    table.add_column("Account")
    table.add_column("Strategy")
    table.add_column("Key Env Var")
    table.add_column("Status")
    table.add_column("Cash")

    strategies = {
        "momentum": "Cross-sectional momentum top-10 (SP500)",
        "multifactor": "Multi-factor (mom+val+qual) + news filter",
        "baseline": "SPY buy-and-hold",
    }

    for acct_name, (key_var, secret_var) in ACCOUNT_KEYS.items():
        key = os.environ.get(key_var, "")
        secret = os.environ.get(secret_var, "")
        strategy = strategies.get(acct_name, "?")

        if not key or not secret:
            table.add_row(acct_name, strategy, key_var, "[red]missing keys[/red]", "—")
            continue

        try:
            client = TradingClient(key, secret, paper=True)
            a = client.get_account()
            table.add_row(
                acct_name, strategy, key_var,
                "[green]connected[/green]",
                f"${float(a.cash):,.2f}",
            )
        except Exception as e:
            table.add_row(acct_name, strategy, key_var, f"[red]{e}[/red]", "—")

    console.print(table)
