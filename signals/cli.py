"""Typer CLI entry point for the Signals package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from signals.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from signals.backtest.metrics import deflated_sharpe_ratio
from signals.config import SETTINGS
from signals.data.pipeline import DataPipeline
from signals.data.storage import DataStore
from signals.data.yahoo import YahooFinanceSource
from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.composite import CompositeMarkovChain
from signals.model.hmm import HiddenMarkovModel
from signals.model.homc import HigherOrderMarkovChain
from signals.model.signals import SignalGenerator
from signals.utils.logging import get_logger

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Signals CLI")
data_app = typer.Typer(no_args_is_help=True, help="Data ingestion commands")
model_app = typer.Typer(no_args_is_help=True, help="Markov model commands")
signal_app = typer.Typer(no_args_is_help=True, help="Signal generation commands")
backtest_app = typer.Typer(no_args_is_help=True, help="Backtesting commands")

app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(signal_app, name="signal")
app.add_typer(backtest_app, name="backtest")

console = Console()

VALID_MODELS = {"composite", "hmm", "homc", "hybrid", "trend", "golden_cross"}


def _store() -> DataStore:
    return DataStore(SETTINGS.data.dir)


def _pipeline() -> DataPipeline:
    return DataPipeline(source=YahooFinanceSource(), store=_store())


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "").replace("/", "_")


def _model_path(symbol: str, interval: str, model_type: str) -> Path:
    base = Path(SETTINGS.data.dir) / "models"
    if model_type == "hmm":
        return base / f"{_safe_symbol(symbol)}_{interval}_hmm.pkl"
    if model_type == "composite":
        return base / f"{_safe_symbol(symbol)}_{interval}_composite.json"
    return base / f"{_safe_symbol(symbol)}_{interval}_homc.json"


def _validate_model(model_type: str) -> None:
    if model_type not in VALID_MODELS:
        raise typer.BadParameter(f"--model must be one of {sorted(VALID_MODELS)}")


def _split_holdout(
    prices: pd.DataFrame, holdout_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the trailing `holdout_frac` of `prices` off as a held-out set.

    Returns (train_portion, holdout_portion). The holdout is contiguous and
    chronologically *after* the train portion, so it can be used to validate
    a strategy whose hyperparameters were tuned on the train portion only.

    A holdout_frac outside (0, 1) returns the full series and an empty
    holdout, which makes "no holdout" a no-op for callers.
    """
    if holdout_frac <= 0 or holdout_frac >= 1:
        return prices, prices.iloc[0:0]
    n = len(prices)
    split = int(n * (1 - holdout_frac))
    if split <= 0 or split >= n:
        return prices, prices.iloc[0:0]
    return prices.iloc[:split], prices.iloc[split:]


# ============================================================
# data
# ============================================================
@data_app.command("fetch")
def data_fetch(
    symbol: str,
    start: str = typer.Option(..., "--start"),
    end: str | None = typer.Option(None, "--end"),
    interval: str = typer.Option("1d", "--interval"),
) -> None:
    pipeline = _pipeline()
    df = pipeline.fetch(symbol, start=start, end=end, interval=interval)
    console.print(f"[green]Saved[/green] {len(df)} rows for [bold]{symbol}[/bold] ({interval})")


@data_app.command("refresh")
def data_refresh(symbol: str, interval: str = typer.Option("1d", "--interval")) -> None:
    pipeline = _pipeline()
    df = pipeline.refresh(symbol, interval=interval)
    console.print(f"[green]Refreshed[/green] {symbol} ({interval}) — {len(df)} rows total")


@data_app.command("list")
def data_list() -> None:
    rows = _store().list_datasets()
    if not rows:
        console.print("[yellow]No datasets stored.[/yellow]")
        return
    table = Table(title="Stored datasets")
    for col in ("symbol", "interval", "rows", "first", "last"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["symbol"], r["interval"], str(r["rows"]), str(r["first"]), str(r["last"]))
    console.print(table)


# ============================================================
# features helper
# ============================================================
def _features_for(symbol: str, interval: str, vol_window: int = 10) -> pd.DataFrame:
    df = _store().load(symbol, interval)
    if df.empty:
        raise typer.BadParameter(f"No data for {symbol} {interval} — run `signals data fetch` first.")
    feats = pd.DataFrame(index=df.index)
    feats["close"] = df["close"]
    feats["return_1d"] = log_returns(df["close"])
    # Column name kept as "volatility_20d" for backward compatibility with the
    # encoder's default feature name; the *window* itself is configurable.
    feats["volatility_20d"] = rolling_volatility(feats["return_1d"], window=vol_window)
    return feats.dropna()


def _build_model(model_type: str, n_states: int, order: int, n_iter: int, alpha: float = 0.01):
    if model_type == "hmm":
        return HiddenMarkovModel(n_states=n_states, n_iter=n_iter)
    if model_type == "composite":
        # n_states is interpreted as return_bins * volatility_bins; default 3×3 = 9.
        return CompositeMarkovChain(return_bins=3, volatility_bins=3, alpha=alpha)
    return HigherOrderMarkovChain(n_states=n_states, order=order, alpha=alpha)


def _fit_model(model, feats: pd.DataFrame):
    if isinstance(model, HiddenMarkovModel):
        model.fit(feats, feature_cols=["return_1d", "volatility_20d"], return_col="return_1d")
    elif isinstance(model, CompositeMarkovChain):
        model.fit(
            feats,
            return_feature="return_1d",
            volatility_feature="volatility_20d",
            return_col="return_1d",
        )
    else:
        model.fit(feats, feature_col="return_1d", return_col="return_1d")
    return model


def _load_model(symbol: str, interval: str, model_type: str):
    path = _model_path(symbol, interval, model_type)
    if not path.exists():
        raise typer.BadParameter(
            f"No trained {model_type.upper()} for {symbol} — run `signals model train --model {model_type}` first."
        )
    if model_type == "hmm":
        return HiddenMarkovModel.load(path)
    if model_type == "composite":
        return CompositeMarkovChain.load(path)
    return HigherOrderMarkovChain.load(path)


# ============================================================
# model
# ============================================================
@model_app.command("train")
def model_train(
    symbol: str,
    model: str = typer.Option("hmm", "--model", help="Model type: hmm or homc"),
    states: int = typer.Option(4, "--states", help="Number of (hidden or quantile) states"),
    order: int = typer.Option(3, "--order", help="HOMC order (only for --model homc)"),
    n_iter: int = typer.Option(200, "--n-iter", help="HMM EM iterations (only for --model hmm)"),
    window: int | None = typer.Option(None, "--window"),
    interval: str = typer.Option("1d", "--interval"),
) -> None:
    _validate_model(model)
    feats = _features_for(symbol, interval)
    if window:
        feats = feats.iloc[-window:]

    m = _build_model(model, states, order, n_iter, alpha=1.0)
    _fit_model(m, feats)

    out = _model_path(symbol, interval, model)
    m.save(out)
    console.print(f"[green]Trained[/green] {model.upper()} for {symbol} → {out}")
    if model == "hmm":
        ll = f"{m.log_likelihood_:.1f}" if m.log_likelihood_ is not None else "n/a"
        console.print(
            f"states={m.n_states}  bars={len(feats)}  converged={m.converged_}  log-likelihood={ll}"
        )
    elif model == "composite":
        console.print(
            f"states={m.n_states}  return_bins={m.return_bins}  volatility_bins={m.volatility_bins}  bars={len(feats)}"
        )
    else:
        console.print(
            f"states={m.n_states}  order={m.order}  "
            f"observed_histories={len(m.transitions_)}  bars={len(feats)}"
        )


@model_app.command("inspect")
def model_inspect(
    symbol: str,
    model: str = typer.Option("hmm", "--model"),
    interval: str = typer.Option("1d", "--interval"),
) -> None:
    _validate_model(model)
    m = _load_model(symbol, interval, model)
    n = m.n_states

    if model in ("hmm", "composite"):
        title = "HMM transition matrix" if model == "hmm" else "Composite chain transition matrix"
        table = Table(title=f"{title} — {symbol}")
        table.add_column("from \\ to")
        for j in range(n):
            table.add_column(m.label(j))
        for i in range(n):
            row = [m.label(i)] + [f"{m.transmat_[i, j]:.3f}" for j in range(n)]
            table.add_row(*row)
        console.print(table)

        steady = m.steady_state()
        st = Table(title="State characteristics")
        st.add_column("state")
        st.add_column("steady prob")
        st.add_column("avg return")
        st.add_column("# obs")
        if model == "hmm":
            for col in m.feature_cols_:
                st.add_column(f"μ {col}")
        for i in range(n):
            row = [
                m.label(i),
                f"{steady[i]:.3f}",
                f"{m.state_returns_[i]:+.4f}",
                str(int(m.state_counts_[i])),
            ]
            if model == "hmm":
                for j in range(len(m.feature_cols_)):
                    row.append(f"{m.feature_means_[i, j]:+.4f}")
            st.add_row(*row)
        console.print(st)
        console.print(f"Row sums: {m.transmat_.sum(axis=1).round(6).tolist()}")
        if model == "hmm":
            console.print(f"Steady sum: {steady.sum():.6f}  converged: {m.converged_}")
        else:
            console.print(f"Steady sum: {steady.sum():.6f}")
    else:
        # HOMC: show top rules in the paper's style
        rules = m.top_rules(k=15)
        table = Table(title=f"HOMC rules (top {len(rules)}) — order={m.order}, states={m.n_states}")
        table.add_column("history")
        table.add_column("→ most likely next")
        table.add_column("P(next)")
        table.add_column("E[r_next]")
        for r in rules:
            table.add_row(
                r["history"],
                r["most_likely_next"],
                f"{r['p_next']:.3f}",
                f"{r['expected_return'] * 1e4:+.1f} bps",
            )
        console.print(table)
        steady = m.steady_state()
        st = Table(title="Per-bin marginal characteristics")
        st.add_column("bin")
        st.add_column("steady prob")
        st.add_column("avg return")
        st.add_column("# obs")
        for i in range(n):
            st.add_row(
                m.label(i),
                f"{steady[i]:.3f}",
                f"{m.state_returns_[i]:+.4f}",
                str(int(m.state_counts_[i])),
            )
        console.print(st)
        console.print(f"Observed k-tuples: {len(m.transitions_)} / {n ** m.order} possible")


@model_app.command("plot")
def model_plot(
    symbol: str,
    model: str = typer.Option("hmm", "--model"),
    interval: str = typer.Option("1d", "--interval"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    _validate_model(model)
    import matplotlib.pyplot as plt

    if model in ("hmm", "composite"):
        m = _load_model(symbol, interval, model)
        mat = m.transmat_
    else:
        m = _load_model(symbol, interval, "homc")
        # Marginalize the k-tuple table to a 1st-order matrix for visualization
        mat = m.steady_state()  # not really a matrix; build T1 instead
        n = m.n_states
        T1 = pd.DataFrame(0.0, index=range(n), columns=range(n))
        for key, probs in m.transitions_.items():
            T1.loc[key[-1]] += probs
        T1 = T1.div(T1.sum(axis=1), axis=0).fillna(0).to_numpy()
        mat = T1

    n = m.n_states
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=mat.max() if mat.size else 1)
    labels = [m.label(i) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_title(f"{model.upper()} transitions — {symbol}")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    out_path = output or Path(f"{_safe_symbol(symbol)}_{model}_transitions.png")
    plt.savefig(out_path, dpi=120)
    console.print(f"[green]Saved[/green] heatmap → {out_path}")


# ============================================================
# signal
# ============================================================
@signal_app.command("now")
def signal_now(
    symbol: str,
    model: str = typer.Option("composite", "--model"),
    interval: str = typer.Option("1d", "--interval"),
    buy_bps: float = typer.Option(25.0, "--buy-bps"),
    sell_bps: float = typer.Option(-35.0, "--sell-bps"),
    target_scale_bps: float = typer.Option(20.0, "--target-scale-bps"),
    allow_short: bool = typer.Option(False, "--short/--no-short"),
) -> None:
    _validate_model(model)
    m = _load_model(symbol, interval, model)
    feats = _features_for(symbol, interval)
    current_state = m.predict_state(feats)
    gen = SignalGenerator(
        model=m,
        buy_threshold_bps=buy_bps,
        sell_threshold_bps=sell_bps,
        target_scale_bps=target_scale_bps,
        allow_short=allow_short,
    )
    decision = gen.generate(current_state)
    last_ts = feats.index[-1]

    # Persist signal (state stored as int when possible)
    state_int = int(current_state) if isinstance(current_state, (int, np.integer)) else -1
    _store().record_signal(
        symbol=symbol,
        ts=last_ts,
        signal=decision.signal.value,
        confidence=decision.confidence,
        state=state_int,
        expected_return=decision.expected_return,
    )
    console.print(
        f"[bold]{symbol}[/bold] ({model.upper()}) @ {last_ts}: "
        f"[bold]{decision.signal.value}[/bold]  "
        f"state=[cyan]{decision.state_label}[/cyan]  "
        f"E[r_next]={decision.expected_return * 1e4:+.1f} bps  "
        f"confidence={decision.confidence:.2f}  "
        f"target={decision.target_position:+.2f}"
    )


@signal_app.command("next")
def signal_next(
    symbol: str,
    model: str = typer.Option("composite", "--model"),
    interval: str = typer.Option("1d", "--interval"),
    buy_bps: float = typer.Option(25.0, "--buy-bps"),
    sell_bps: float = typer.Option(-35.0, "--sell-bps"),
    target_scale_bps: float = typer.Option(20.0, "--target-scale-bps"),
    allow_short: bool = typer.Option(False, "--short/--no-short"),
    train_window: int = typer.Option(252, "--train-window"),
    vol_window: int = typer.Option(10, "--vol-window"),
    refresh: bool = typer.Option(True, "--refresh/--no-refresh"),
) -> None:
    """One-shot 'what should I do tomorrow?' workflow.

    Refreshes the latest bars, retrains the model on a rolling window
    (so it never gets stale), and prints the action for the *next* bar's
    open along with size, expected return, and the current state.
    """
    _validate_model(model)

    # 1. Refresh data
    if refresh:
        try:
            df = _pipeline().refresh(symbol, interval=interval)
            console.print(f"[dim]refreshed {symbol} ({interval}) — {len(df)} bars total[/dim]")
        except Exception as e:
            console.print(f"[yellow]refresh failed:[/yellow] {e} (using stored data)")

    feats = _features_for(symbol, interval, vol_window=vol_window)
    if len(feats) < train_window:
        raise typer.BadParameter(
            f"Need >= {train_window} bars to train; got {len(feats)}. "
            f"Run `signals data fetch {symbol} --start <earlier date>` first."
        )

    # 2. Train fresh on the most recent train_window bars (no stale model state)
    train_slice = feats.iloc[-train_window:]
    fresh = _build_model(model, n_states=9, order=3, n_iter=200, alpha=1.0)
    _fit_model(fresh, train_slice)
    out = _model_path(symbol, interval, model)
    fresh.save(out)

    # 3. Decode current state from the same window
    current_state = fresh.predict_state(train_slice)

    # 4. Generate signal with the optimized strategy params
    gen = SignalGenerator(
        model=fresh,
        buy_threshold_bps=buy_bps,
        sell_threshold_bps=sell_bps,
        target_scale_bps=target_scale_bps,
        allow_short=allow_short,
    )
    decision = gen.generate(current_state)

    last_ts = feats.index[-1]
    last_close = float(feats["close"].iloc[-1])

    # 5. Persist for the audit trail
    state_int = int(current_state) if isinstance(current_state, (int, np.integer)) else -1
    _store().record_signal(
        symbol=symbol,
        ts=last_ts,
        signal=decision.signal.value,
        confidence=decision.confidence,
        state=state_int,
        expected_return=decision.expected_return,
    )

    # 6. Print the actionable summary
    action_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}[decision.signal.value]
    table = Table(title=f"Tomorrow's action — {symbol}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("Last bar", str(last_ts))
    table.add_row("Last close", f"${last_close:,.2f}")
    table.add_row("Model", f"{model.upper()} ({train_window}-bar window)")
    table.add_row("Current state", f"{decision.state_label} (idx {state_int})")
    table.add_row("E\\[r_next]", f"{decision.expected_return * 1e4:+.1f} bps")
    table.add_row("Confidence", f"{decision.confidence:.2f}")
    table.add_row("Action", f"[{action_color}][bold]{decision.signal.value}[/bold][/{action_color}]")
    table.add_row("Target position", f"{decision.target_position:+.0%}")
    table.add_row("Thresholds", f"buy ≥ {buy_bps:.0f} bps, sell ≤ {sell_bps:.0f} bps")
    console.print(table)
    console.print(
        f"[dim]Execute at the open of the next {interval} bar after {last_ts}.[/dim]"
    )


@signal_app.command("history")
def signal_history(symbol: str, days: int = typer.Option(30, "--days")) -> None:
    df = _store().recent_signals(symbol, days)
    if df.empty:
        console.print("[yellow]No recorded signals.[/yellow]")
        return
    table = Table(title=f"Recent signals — {symbol}")
    for col in df.columns:
        table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row.tolist()])
    console.print(table)


# ============================================================
# backtest
# ============================================================
def _build_config(
    model: str,
    states: int,
    order: int,
    train_window: int,
    retrain_freq: int,
    *,
    buy_bps: float | None = None,
    sell_bps: float | None = None,
    target_scale_bps: float = 20.0,
    allow_short: bool = False,
    max_long: float = 1.0,
    max_short: float = 1.0,
    stop_loss_pct: float = 0.0,
    stop_cooldown_bars: int = 5,
    min_trade_fraction: float = 0.20,
    hold_preserves_position: bool = True,
    return_bins: int = 3,
    volatility_bins: int = 3,
    vol_window: int = 10,
    laplace_alpha: float = 0.01,
) -> BacktestConfig:
    return BacktestConfig(
        model_type=model,
        train_window=train_window,
        retrain_freq=retrain_freq,
        n_states=states,
        return_bins=return_bins,
        volatility_bins=volatility_bins,
        order=order,
        vol_window=vol_window,
        laplace_alpha=laplace_alpha,
        initial_cash=SETTINGS.backtest.initial_cash,
        commission_bps=SETTINGS.backtest.commission_bps,
        slippage_bps=SETTINGS.backtest.slippage_bps,
        buy_threshold_bps=buy_bps if buy_bps is not None else SETTINGS.signal.buy_threshold_bps,
        sell_threshold_bps=sell_bps if sell_bps is not None else SETTINGS.signal.sell_threshold_bps,
        target_scale_bps=target_scale_bps,
        allow_short=allow_short,
        max_long=max_long,
        max_short=max_short,
        stop_loss_pct=stop_loss_pct,
        stop_cooldown_bars=stop_cooldown_bars,
        min_trade_fraction=min_trade_fraction,
        hold_preserves_position=hold_preserves_position,
    )


def _print_backtest_table(result: BacktestResult, model_label: str) -> None:
    m, b = result.metrics, result.benchmark_metrics
    table = Table(title=f"Backtest — {result.symbol} ({model_label}) {result.start.date()} → {result.end.date()}")
    table.add_column("Metric")
    table.add_column("Strategy")
    table.add_column("Buy & Hold")
    rows = [
        ("Final equity", f"${m.final_equity:,.2f}", f"${b.final_equity:,.2f}"),
        ("CAGR", f"{m.cagr * 100:.2f}%", f"{b.cagr * 100:.2f}%"),
        ("Sharpe", f"{m.sharpe:.2f}", f"{b.sharpe:.2f}"),
        ("Max DD", f"{m.max_drawdown * 100:.2f}%", f"{b.max_drawdown * 100:.2f}%"),
        ("Calmar", f"{m.calmar:.2f}", f"{b.calmar:.2f}"),
        ("Win rate", f"{m.win_rate * 100:.1f}%", "—"),
        ("Profit factor", f"{m.profit_factor:.2f}", "—"),
        ("# trades", str(m.n_trades), "—"),
    ]
    for r in rows:
        table.add_row(*r)
    console.print(table)


def _persist_backtest(result: BacktestResult, model_type: str) -> int:
    m = result.metrics
    return _store().record_backtest(
        {
            "symbol": f"{result.symbol} [{model_type}]",
            "start_date": str(result.start.date()),
            "end_date": str(result.end.date()),
            "n_states": result.config.n_states,
            "train_window": result.config.train_window,
            "retrain_freq": result.config.retrain_freq,
            "sharpe": m.sharpe,
            "cagr": m.cagr,
            "max_drawdown": m.max_drawdown,
            "win_rate": m.win_rate,
            "profit_factor": m.profit_factor,
            "calmar": m.calmar,
            "final_equity": m.final_equity,
            "n_trades": m.n_trades,
        }
    )


@backtest_app.command("run")
def backtest_run(
    symbol: str,
    model: str = typer.Option("composite", "--model"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    interval: str = typer.Option("1d", "--interval"),
    states: int = typer.Option(9, "--states"),
    order: int = typer.Option(3, "--order"),
    train_window: int = typer.Option(252, "--train-window"),
    retrain_freq: int = typer.Option(21, "--retrain-freq"),
    vol_window: int = typer.Option(10, "--vol-window"),
    laplace_alpha: float = typer.Option(0.01, "--alpha"),
    buy_bps: float = typer.Option(25.0, "--buy-bps"),
    sell_bps: float = typer.Option(-35.0, "--sell-bps"),
    target_scale_bps: float = typer.Option(20.0, "--target-scale-bps"),
    allow_short: bool = typer.Option(False, "--short/--no-short"),
    max_long: float = typer.Option(1.0, "--max-long"),
    max_short: float = typer.Option(1.0, "--max-short"),
    stop_loss_pct: float = typer.Option(0.0, "--stop-loss"),
    stop_cooldown: int = typer.Option(5, "--stop-cooldown"),
    min_trade: float = typer.Option(0.20, "--min-trade"),
    holdout_frac: float = typer.Option(
        0.0, "--holdout-frac",
        help="Reserve trailing fraction of data as a held-out validation set "
             "(e.g. 0.2 for the last 20%). Strategy is run on the train portion; "
             "the held-out window is then evaluated separately for validation.",
    ),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
) -> None:
    _validate_model(model)
    prices = _store().load(symbol, interval)
    if prices.empty:
        raise typer.BadParameter(f"No data for {symbol}; run `signals data fetch` first.")

    # Apply --start / --end first, then carve off the holdout from the remainder.
    if start is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start, tz=prices.index.tz or "UTC")]
    if end is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(end, tz=prices.index.tz or "UTC")]
    train_prices, holdout_prices = _split_holdout(prices, holdout_frac)
    if not holdout_prices.empty:
        console.print(
            f"[dim]holdout reserved: {holdout_prices.index[0].date()} → "
            f"{holdout_prices.index[-1].date()} ({len(holdout_prices)} bars)[/dim]"
        )

    cfg = _build_config(
        model, states, order, train_window, retrain_freq,
        buy_bps=buy_bps, sell_bps=sell_bps,
        target_scale_bps=target_scale_bps,
        allow_short=allow_short, max_long=max_long, max_short=max_short,
        stop_loss_pct=stop_loss_pct, stop_cooldown_bars=stop_cooldown,
        min_trade_fraction=min_trade,
        vol_window=vol_window, laplace_alpha=laplace_alpha,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(train_prices, symbol=symbol)
    _print_backtest_table(result, model.upper())
    run_id = _persist_backtest(result, model)
    console.print(f"[green]Recorded[/green] backtest run [bold]{run_id}[/bold]")

    # Validate on the held-out portion using the same config.
    if not holdout_prices.empty:
        # The engine needs train_window bars before the holdout to warm up,
        # so feed it the trailing slice of train_prices stitched onto the holdout.
        warmup = train_prices.iloc[-(train_window + cfg.vol_window + 5):]
        holdout_with_warmup = pd.concat([warmup, holdout_prices])
        holdout_with_warmup = holdout_with_warmup[~holdout_with_warmup.index.duplicated()]
        holdout_result = BacktestEngine(cfg).run(holdout_with_warmup, symbol=symbol)
        # Trim equity curve to the holdout window for fair reporting.
        holdout_eq = holdout_result.equity_curve.loc[
            holdout_result.equity_curve.index >= holdout_prices.index[0]
        ]
        from signals.backtest.metrics import compute_metrics

        if not holdout_eq.empty:
            normalized = (holdout_eq / holdout_eq.iloc[0]) * cfg.initial_cash
            holdout_metrics = compute_metrics(
                normalized, holdout_result.trades, risk_free_rate=cfg.risk_free_rate
            )
            console.print()
            console.print("[bold yellow]Held-out validation[/bold yellow]")
            ht = Table(title=f"Holdout — {symbol} ({model.upper()})")
            ht.add_column("Metric")
            ht.add_column("Train (in-sample)")
            ht.add_column("Holdout (out-of-sample)")
            train_m = result.metrics
            rows = [
                ("CAGR", f"{train_m.cagr * 100:.2f}%", f"{holdout_metrics.cagr * 100:.2f}%"),
                ("Sharpe", f"{train_m.sharpe:.2f}", f"{holdout_metrics.sharpe:.2f}"),
                ("Max DD", f"{train_m.max_drawdown * 100:.2f}%", f"{holdout_metrics.max_drawdown * 100:.2f}%"),
                ("Calmar", f"{train_m.calmar:.2f}", f"{holdout_metrics.calmar:.2f}"),
                ("# trades", str(train_m.n_trades), str(holdout_metrics.n_trades)),
            ]
            for r in rows:
                ht.add_row(*r)
            console.print(ht)

    if plot and not result.equity_curve.empty:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        result.equity_curve.plot(ax=ax, label=f"Strategy ({model.upper()})", linewidth=2)
        result.benchmark_curve.plot(ax=ax, label="Buy & Hold", linewidth=1, alpha=0.7)
        ax.set_title(f"{symbol} — {model.upper()} equity curve")
        ax.set_ylabel("Equity ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_path = Path(f"{_safe_symbol(symbol)}_{model}_backtest_{run_id}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        console.print(f"[green]Saved[/green] equity curve → {out_path}")


@backtest_app.command("compare")
def backtest_compare(
    symbol: str,
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    interval: str = typer.Option("1d", "--interval"),
    hmm_states: int = typer.Option(4, "--hmm-states"),
    homc_states: int = typer.Option(5, "--homc-states"),
    homc_order: int = typer.Option(3, "--homc-order"),
    train_window: int = typer.Option(252, "--train-window"),
    retrain_freq: int = typer.Option(21, "--retrain-freq"),
    vol_window: int = typer.Option(10, "--vol-window"),
    laplace_alpha: float = typer.Option(0.01, "--alpha"),
    buy_bps: float = typer.Option(25.0, "--buy-bps"),
    sell_bps: float = typer.Option(-35.0, "--sell-bps"),
    target_scale_bps: float = typer.Option(20.0, "--target-scale-bps"),
    allow_short: bool = typer.Option(False, "--short/--no-short"),
    max_long: float = typer.Option(1.0, "--max-long"),
    max_short: float = typer.Option(1.0, "--max-short"),
    stop_loss_pct: float = typer.Option(0.0, "--stop-loss"),
    stop_cooldown: int = typer.Option(5, "--stop-cooldown"),
    min_trade: float = typer.Option(0.20, "--min-trade"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
) -> None:
    """Run composite + HMM + HOMC backtests with the same strategy params."""
    prices = _store().load(symbol, interval)
    if prices.empty:
        raise typer.BadParameter(f"No data for {symbol}; run `signals data fetch` first.")

    common = dict(
        train_window=train_window, retrain_freq=retrain_freq,
        buy_bps=buy_bps, sell_bps=sell_bps,
        target_scale_bps=target_scale_bps,
        allow_short=allow_short, max_long=max_long, max_short=max_short,
        stop_loss_pct=stop_loss_pct, stop_cooldown_bars=stop_cooldown,
        min_trade_fraction=min_trade,
        vol_window=vol_window, laplace_alpha=laplace_alpha,
    )

    console.print("[bold]Running Composite backtest[/bold] (3×3)...")
    comp_cfg = _build_config("composite", states=9, order=1, **common)
    comp_result = BacktestEngine(comp_cfg).run(prices, symbol=symbol, start=start, end=end)

    console.print(f"[bold]Running HMM backtest[/bold] ({hmm_states} states)...")
    hmm_cfg = _build_config("hmm", states=hmm_states, order=1, **common)
    hmm_result = BacktestEngine(hmm_cfg).run(prices, symbol=symbol, start=start, end=end)

    console.print(f"[bold]Running HOMC backtest[/bold] ({homc_states} states, order {homc_order})...")
    homc_cfg = _build_config("homc", states=homc_states, order=homc_order, **common)
    homc_result = BacktestEngine(homc_cfg).run(prices, symbol=symbol, start=start, end=end)

    c_m = comp_result.metrics
    h_m = hmm_result.metrics
    o_m = homc_result.metrics
    bh = hmm_result.benchmark_metrics

    table = Table(
        title=f"3-way comparison — {symbol} ({hmm_result.start.date()} → {hmm_result.end.date()})"
    )
    table.add_column("Metric")
    table.add_column("Composite (3×3)")
    table.add_column(f"HMM ({hmm_states}s)")
    table.add_column(f"HOMC ({homc_states}s, o{homc_order})")
    table.add_column("Buy & Hold")
    rows = [
        ("Final equity",
         f"${c_m.final_equity:,.2f}", f"${h_m.final_equity:,.2f}", f"${o_m.final_equity:,.2f}", f"${bh.final_equity:,.2f}"),
        ("CAGR",
         f"{c_m.cagr * 100:.2f}%", f"{h_m.cagr * 100:.2f}%", f"{o_m.cagr * 100:.2f}%", f"{bh.cagr * 100:.2f}%"),
        ("Sharpe",
         f"{c_m.sharpe:.2f}", f"{h_m.sharpe:.2f}", f"{o_m.sharpe:.2f}", f"{bh.sharpe:.2f}"),
        ("Max DD",
         f"{c_m.max_drawdown * 100:.2f}%", f"{h_m.max_drawdown * 100:.2f}%", f"{o_m.max_drawdown * 100:.2f}%", f"{bh.max_drawdown * 100:.2f}%"),
        ("Calmar",
         f"{c_m.calmar:.2f}", f"{h_m.calmar:.2f}", f"{o_m.calmar:.2f}", f"{bh.calmar:.2f}"),
        ("Win rate",
         f"{c_m.win_rate * 100:.1f}%", f"{h_m.win_rate * 100:.1f}%", f"{o_m.win_rate * 100:.1f}%", "—"),
        ("Profit factor",
         f"{c_m.profit_factor:.2f}", f"{h_m.profit_factor:.2f}", f"{o_m.profit_factor:.2f}", "—"),
        ("# trades",
         str(c_m.n_trades), str(h_m.n_trades), str(o_m.n_trades), "—"),
    ]
    for r in rows:
        table.add_row(*r)
    console.print(table)

    comp_id = _persist_backtest(comp_result, "composite")
    hmm_id = _persist_backtest(hmm_result, "hmm")
    homc_id = _persist_backtest(homc_result, "homc")
    console.print(
        f"[green]Recorded[/green] runs Composite=[bold]{comp_id}[/bold], "
        f"HMM=[bold]{hmm_id}[/bold], HOMC=[bold]{homc_id}[/bold]"
    )

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 6))
        comp_result.equity_curve.plot(ax=ax, label="Composite (3×3)", linewidth=2)
        hmm_result.equity_curve.plot(ax=ax, label=f"HMM ({hmm_states}s)", linewidth=2)
        homc_result.equity_curve.plot(ax=ax, label=f"HOMC ({homc_states}s, o{homc_order})", linewidth=2)
        hmm_result.benchmark_curve.plot(ax=ax, label="Buy & Hold", linewidth=1, alpha=0.6)
        ax.set_title(f"{symbol} — 3-way model comparison")
        ax.set_ylabel("Equity ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_path = Path(f"{_safe_symbol(symbol)}_compare.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        console.print(f"[green]Saved[/green] comparison plot → {out_path}")


@backtest_app.command("sweep")
def backtest_sweep(
    symbol: str,
    model: str = typer.Option("composite", "--model"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    interval: str = typer.Option("1d", "--interval"),
    states: int = typer.Option(9, "--states"),
    order: int = typer.Option(3, "--order"),
    train_window: int = typer.Option(504, "--train-window"),
    retrain_freq: int = typer.Option(21, "--retrain-freq"),
    buy_grid: str = typer.Option("5,10,15,20,30", "--buy-grid", help="comma-separated buy threshold bps"),
    sell_grid: str = typer.Option("-5,-10,-15,-20,-30", "--sell-grid"),
    stop_grid: str = typer.Option("0,0.05,0.08,0.12", "--stop-grid"),
    target_scale_bps: float = typer.Option(50.0, "--target-scale-bps"),
    allow_short: bool = typer.Option(True, "--short/--no-short"),
    rank_by: str = typer.Option("calmar", "--rank-by", help="calmar | sharpe | cagr | final"),
    top: int = typer.Option(10, "--top"),
    holdout_frac: float = typer.Option(
        0.0, "--holdout-frac",
        help="Reserve trailing fraction of data as a held-out validation set. "
             "Top configs are re-run on the holdout for an honest out-of-sample read.",
    ),
) -> None:
    """Grid-search buy/sell thresholds and stop-loss; print top configs."""
    _validate_model(model)
    prices = _store().load(symbol, interval)
    if prices.empty:
        raise typer.BadParameter(f"No data for {symbol}; run `signals data fetch` first.")

    # Apply --start / --end first, then carve off the holdout from the remainder.
    if start is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start, tz=prices.index.tz or "UTC")]
    if end is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(end, tz=prices.index.tz or "UTC")]
    train_prices, holdout_prices = _split_holdout(prices, holdout_frac)
    if not holdout_prices.empty:
        console.print(
            f"[dim]holdout reserved: {holdout_prices.index[0].date()} → "
            f"{holdout_prices.index[-1].date()} ({len(holdout_prices)} bars)[/dim]"
        )

    buy_vals = [float(x) for x in buy_grid.split(",")]
    sell_vals = [float(x) for x in sell_grid.split(",")]
    stop_vals = [float(x) for x in stop_grid.split(",")]

    rank_keys = {
        "calmar": lambda m: m.calmar,
        "sharpe": lambda m: m.sharpe,
        "cagr": lambda m: m.cagr,
        "final": lambda m: m.final_equity,
    }
    if rank_by not in rank_keys:
        raise typer.BadParameter(f"--rank-by must be one of {sorted(rank_keys)}")
    keyf = rank_keys[rank_by]

    n_total = len(buy_vals) * len(sell_vals) * len(stop_vals)
    console.print(f"[bold]Sweeping[/bold] {n_total} configs ({model.upper()})...")

    results: list[dict] = []
    for b in buy_vals:
        for s in sell_vals:
            if s >= b:
                continue
            for stp in stop_vals:
                cfg = _build_config(
                    model, states, order, train_window, retrain_freq,
                    buy_bps=b, sell_bps=s,
                    target_scale_bps=target_scale_bps,
                    allow_short=allow_short,
                    stop_loss_pct=stp,
                )
                try:
                    res = BacktestEngine(cfg).run(train_prices, symbol=symbol)
                except Exception as e:
                    log_local = get_logger("sweep")
                    log_local.warning("config failed (b=%s s=%s stop=%s): %s", b, s, stp, e)
                    continue
                m = res.metrics
                results.append({
                    "buy_bps": b, "sell_bps": s, "stop": stp,
                    "final": m.final_equity, "cagr": m.cagr, "sharpe": m.sharpe,
                    "mdd": m.max_drawdown, "calmar": m.calmar, "trades": m.n_trades,
                    "_metrics": m,
                    "_cfg": cfg,
                    "_n_obs": len(res.equity_curve),
                })

    if not results:
        console.print("[red]No successful runs[/red]")
        return

    results.sort(key=lambda r: keyf(r["_metrics"]), reverse=True)

    # Deflated Sharpe corrects for the multiple-testing problem inherent in
    # grid search: a sweep over N configs will produce a "best" Sharpe even
    # against pure noise. DSR is the probability that the observed Sharpe is
    # better than the expected max of N IID Sharpe estimates under H0 (true
    # SR=0). DSR > 0.95 is the conventional bar for "this isn't just noise".
    n_trials = len(results)
    bench_metrics = BacktestEngine(
        _build_config(model, states, order, train_window, retrain_freq)
    ).run(train_prices, symbol=symbol).benchmark_metrics

    table = Table(title=f"Sweep top {top} by {rank_by} — {symbol} ({model.upper()}) — N={n_trials} trials")
    for col in ("buy_bps", "sell_bps", "stop", "final", "CAGR", "Sharpe", "DSR", "Max DD", "Calmar", "trades"):
        table.add_column(col)
    for r in results[:top]:
        dsr = deflated_sharpe_ratio(
            sharpe=r["sharpe"], n_trials=n_trials, n_observations=r["_n_obs"]
        )
        dsr_color = "green" if dsr >= 0.95 else ("yellow" if dsr >= 0.80 else "red")
        table.add_row(
            f"{r['buy_bps']:.0f}",
            f"{r['sell_bps']:.0f}",
            f"{r['stop'] * 100:.0f}%",
            f"${r['final']:,.0f}",
            f"{r['cagr'] * 100:.1f}%",
            f"{r['sharpe']:.2f}",
            f"[{dsr_color}]{dsr:.2f}[/{dsr_color}]",
            f"{r['mdd'] * 100:.1f}%",
            f"{r['calmar']:.2f}",
            str(r['trades']),
        )
    console.print(table)
    console.print(
        "[dim]DSR = deflated Sharpe probability. Bailey & López de Prado (2014). "
        "DSR ≥ 0.95 is the conventional bar for 'survives the multi-trial deflation'.[/dim]"
    )
    console.print(
        f"[dim]Buy & hold benchmark: ${bench_metrics.final_equity:,.0f}, "
        f"CAGR {bench_metrics.cagr * 100:.1f}%, MDD {bench_metrics.max_drawdown * 100:.1f}%, "
        f"Calmar {bench_metrics.calmar:.2f}[/dim]"
    )

    # Re-run the top config on the held-out portion for an honest OOS read.
    if not holdout_prices.empty:
        from signals.backtest.metrics import compute_metrics

        best = results[0]
        best_cfg = best["_cfg"]
        warmup = train_prices.iloc[-(best_cfg.train_window + best_cfg.vol_window + 5):]
        holdout_with_warmup = pd.concat([warmup, holdout_prices])
        holdout_with_warmup = holdout_with_warmup[~holdout_with_warmup.index.duplicated()]
        try:
            holdout_res = BacktestEngine(best_cfg).run(holdout_with_warmup, symbol=symbol)
            holdout_eq = holdout_res.equity_curve.loc[
                holdout_res.equity_curve.index >= holdout_prices.index[0]
            ]
            if not holdout_eq.empty:
                normalized = (holdout_eq / holdout_eq.iloc[0]) * best_cfg.initial_cash
                holdout_metrics = compute_metrics(
                    normalized, holdout_res.trades, risk_free_rate=best_cfg.risk_free_rate
                )
                console.print()
                console.print(
                    f"[bold yellow]Best config validated on holdout[/bold yellow] "
                    f"(buy={best['buy_bps']:.0f} sell={best['sell_bps']:.0f} stop={best['stop'] * 100:.0f}%)"
                )
                ht = Table(title=f"Holdout — {symbol} ({model.upper()})")
                ht.add_column("Metric")
                ht.add_column("Train (in-sample)")
                ht.add_column("Holdout (out-of-sample)")
                tm = best["_metrics"]
                rows = [
                    ("CAGR", f"{tm.cagr * 100:.2f}%", f"{holdout_metrics.cagr * 100:.2f}%"),
                    ("Sharpe", f"{tm.sharpe:.2f}", f"{holdout_metrics.sharpe:.2f}"),
                    ("Max DD", f"{tm.max_drawdown * 100:.2f}%", f"{holdout_metrics.max_drawdown * 100:.2f}%"),
                    ("Calmar", f"{tm.calmar:.2f}", f"{holdout_metrics.calmar:.2f}"),
                    ("# trades", str(tm.n_trades), str(holdout_metrics.n_trades)),
                ]
                for r in rows:
                    ht.add_row(*r)
                console.print(ht)
        except Exception as e:
            console.print(f"[red]Holdout validation failed:[/red] {e}")


@backtest_app.command("portfolio")
def backtest_portfolio(
    spec: list[str] = typer.Argument(
        ...,
        help="Allocations as 'SYMBOL:WEIGHT:MODEL' triples, e.g. "
             "'BTC-USD:0.4:hybrid ^GSPC:0.6:bh'. 'bh' means buy-and-hold.",
    ),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    interval: str = typer.Option("1d", "--interval"),
    rebalance: str = typer.Option("daily", "--rebalance", help="'daily' or 'window'"),
    train_window: int = typer.Option(1000, "--train-window"),
    states: int = typer.Option(5, "--states"),
    order: int = typer.Option(5, "--order"),
    vol_quantile: float = typer.Option(0.70, "--vol-quantile"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
) -> None:
    """Run a multi-asset portfolio backtest.

    Each allocation is given as 'SYMBOL:WEIGHT:MODEL' where MODEL is one
    of {composite, hmm, homc, hybrid, trend, golden_cross, bh}. 'bh'
    means buy-and-hold (no model, no training). Weights must sum to 1.0.

    Example:
        signals backtest portfolio BTC-USD:0.4:hybrid ^GSPC:0.6:bh \\
            --start 2018-01-01 --end 2024-12-31 --rebalance daily

    The Tier-2 finding: 40/60 BTC hybrid / S&P B&H with daily rebalance
    produces median Sharpe 2.44 at seed 42 and averages Sharpe +16% over
    BTC-alone across 4 seeds. See scripts/BTC_SP500_PORTFOLIO_RESULTS.md.
    """
    from signals.backtest.metrics import compute_metrics
    from signals.backtest.portfolio_blend import (
        PortfolioAllocation,
        run_portfolio_backtest,
    )

    if rebalance not in ("daily", "window"):
        raise typer.BadParameter("--rebalance must be 'daily' or 'window'")

    allocations: list[PortfolioAllocation] = []
    for token in spec:
        parts = token.split(":")
        if len(parts) != 3:
            raise typer.BadParameter(
                f"spec must be 'SYMBOL:WEIGHT:MODEL'; got {token!r}"
            )
        sym, weight_s, model_s = parts
        try:
            weight = float(weight_s)
        except ValueError as e:
            raise typer.BadParameter(f"invalid weight {weight_s!r}") from e

        if model_s == "bh":
            cfg = None
        elif model_s == "hybrid":
            cfg = BacktestConfig(
                model_type="hybrid",
                train_window=train_window,
                retrain_freq=21,
                n_states=states,
                order=order,
                return_bins=3,
                volatility_bins=3,
                vol_window=10,
                laplace_alpha=0.01,
                hybrid_routing_strategy="vol",
                hybrid_vol_quantile=vol_quantile,
            )
        elif model_s in ("composite", "hmm", "homc", "trend", "golden_cross"):
            cfg = BacktestConfig(
                model_type=model_s,
                train_window=train_window,
                retrain_freq=21,
                n_states=states,
                order=order,
            )
        else:
            raise typer.BadParameter(
                f"unknown model {model_s!r}; must be composite/hmm/homc/"
                f"hybrid/trend/golden_cross/bh"
            )
        allocations.append(PortfolioAllocation(symbol=sym, cfg=cfg, weight=weight))

    total_weight = sum(a.weight for a in allocations)
    if not (0.999 <= total_weight <= 1.001):
        raise typer.BadParameter(f"weights must sum to 1.0; got {total_weight}")

    prices_by_symbol: dict[str, pd.DataFrame] = {}
    for alloc in allocations:
        prices = _store().load(alloc.symbol, interval)
        if prices.empty:
            raise typer.BadParameter(
                f"No data for {alloc.symbol}; run `signals data fetch` first."
            )
        prices_by_symbol[alloc.symbol] = prices

    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    console.print(
        f"[bold]Running portfolio[/bold] "
        f"({', '.join(f'{a.symbol}={a.weight*100:.0f}%' for a in allocations)}) "
        f"with {rebalance} rebalance"
    )
    port_eq = run_portfolio_backtest(
        allocations=allocations,
        prices_by_symbol=prices_by_symbol,
        rebalance=rebalance,
        start=start_ts,
        end=end_ts,
    )
    if port_eq.empty:
        console.print("[red]Empty portfolio equity curve[/red]")
        raise typer.Exit(1)

    metrics = compute_metrics(port_eq, [])
    table = Table(title="Portfolio backtest result")
    table.add_column("Metric")
    table.add_column("Value")
    rows = [
        ("Start", str(port_eq.index[0].date()) if hasattr(port_eq.index[0], "date") else str(port_eq.index[0])),
        ("End", str(port_eq.index[-1].date()) if hasattr(port_eq.index[-1], "date") else str(port_eq.index[-1])),
        ("Bars", str(len(port_eq))),
        ("Final equity", f"${metrics.final_equity:,.2f}"),
        ("CAGR", f"{metrics.cagr * 100:.2f}%"),
        ("Sharpe", f"{metrics.sharpe:.2f}"),
        ("Max DD", f"{metrics.max_drawdown * 100:.2f}%"),
        ("Calmar", f"{metrics.calmar:.2f}"),
    ]
    for r in rows:
        table.add_row(*r)
    console.print(table)

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        port_eq.plot(ax=ax, label="Portfolio", linewidth=2)
        ax.set_title(
            f"Portfolio: {', '.join(f'{a.symbol} {a.weight*100:.0f}%' for a in allocations)}"
        )
        ax.set_ylabel("Equity ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_path = Path(
            "portfolio_" + "_".join(_safe_symbol(a.symbol) for a in allocations) + ".png"
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        console.print(f"[green]Saved[/green] plot → {out_path}")


@backtest_app.command("list")
def backtest_list() -> None:
    df = _store().list_backtests()
    if df.empty:
        console.print("[yellow]No backtests recorded.[/yellow]")
        return
    table = Table(title="Backtest runs")
    cols = ["id", "symbol", "start_date", "end_date", "sharpe", "cagr", "max_drawdown", "final_equity", "n_trades"]
    for c in cols:
        table.add_column(c)
    for _, row in df.iterrows():
        table.add_row(*[str(row[c]) for c in cols])
    console.print(table)


@backtest_app.command("show")
def backtest_show(run_id: int) -> None:
    df = _store().get_backtest(run_id)
    if df.empty:
        console.print(f"[red]No backtest with id {run_id}[/red]")
        raise typer.Exit(1)
    row = df.iloc[0]
    table = Table(title=f"Backtest #{run_id}")
    table.add_column("field")
    table.add_column("value")
    for col in df.columns:
        table.add_row(col, str(row[col]))
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
