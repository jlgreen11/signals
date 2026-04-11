"""Trace a single random 180-day window of the 4-asset portfolio.

Pulls per-day detail for the production recommendation from Round 4:
equal-weight basket of (BTC hybrid, ^GSPC B&H, TLT B&H, GLD B&H) over
a single random 180-bar window drawn from the shared calendar (days
on which all 4 assets trade — i.e. equity weekdays, because BTC has
its own weekend bars that the others don't).

Outputs a day-by-day table with columns:

    date        — trading day
    btc_sig     — hybrid signal (BUY / HOLD / SELL)
    btc_regime  — state label emitted by the hybrid
    btc_target  — signed position fraction (0.00 = flat, 1.00 = fully long)
    btc_ret     — BTC leg daily return (from the hybrid equity curve)
    sp_ret      — ^GSPC daily return (buy & hold)
    tlt_ret     — TLT daily return (buy & hold)
    gld_ret     — GLD daily return (buy & hold)
    port_ret    — equal-weighted portfolio daily return
    port_value  — portfolio value (start = $10,000)
    appreciation— cumulative return since window start

Writes the full 180-row table to
    scripts/data/portfolio_trace_180d.parquet
    scripts/data/portfolio_trace_180d.md

Prints a summary + compact view to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.engine import (
    BTC_HYBRID_PRODUCTION,
    BacktestConfig,
    BacktestEngine,
)
from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL_BTC = "BTC-USD"
SYMBOL_SP = "^GSPC"
SYMBOL_TLT = "TLT"
SYMBOL_GLD = "GLD"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")

WINDOW_LEN = 180        # trading days, equity calendar
WARMUP_PAD = 5
SEED = 42


def _load(store: DataStore, sym: str) -> pd.DataFrame:
    df = store.load(sym, "1d").sort_index()
    return df.loc[(df.index >= START) & (df.index <= END)]


def _btc_hybrid_run(
    btc_prices: pd.DataFrame, eval_start_ts: pd.Timestamp, eval_end_ts: pd.Timestamp
):
    """Run the BTC hybrid over a warmup + eval slice and return
    (signals_df, equity_curve) both restricted to the eval window."""
    base = dict(BTC_HYBRID_PRODUCTION)
    base["risk_free_rate"] = historical_usd_rate("2018-2024")
    cfg = BacktestConfig(**base)

    # Build the input slice with enough warmup for the hybrid to fit
    # (train_window + vol_window + pad) bars before eval_start_ts.
    start_i = btc_prices.index.searchsorted(eval_start_ts)
    slice_start = max(0, start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD)
    end_i = btc_prices.index.searchsorted(eval_end_ts, side="right")
    engine_input = btc_prices.iloc[slice_start:end_i]

    result = BacktestEngine(cfg).run(engine_input, symbol=SYMBOL_BTC)
    sigs = result.signals
    eq = result.equity_curve

    # Restrict to eval window
    sigs_win = sigs.loc[sigs.index >= eval_start_ts]
    eq_win = eq.loc[eq.index >= eval_start_ts]
    return sigs_win, eq_win


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    btc = _load(store, SYMBOL_BTC)
    sp = _load(store, SYMBOL_SP)
    tlt = _load(store, SYMBOL_TLT)
    gld = _load(store, SYMBOL_GLD)

    # Shared calendar: intersection of the 3 buy-and-hold assets'
    # indices (BTC has extra weekend bars we'll merge onto this).
    shared_idx = sp.index.intersection(tlt.index).intersection(gld.index)
    shared_idx = shared_idx.sort_values()

    # We also need a BTC warmup of at least 750 + 10 + 5 = 765 bars
    # BEFORE the eval start. In the shared (weekday) calendar that's
    # roughly 765 / (365/252) ≈ 528 trading days of buffer. Be safe
    # and use 600.
    buffer_days = 600

    # Pick a single random 180-weekday window via the shared sampler,
    # restricted to [buffer_days, len(shared_idx) - WINDOW_LEN - 1].
    min_start = buffer_days
    max_start = len(shared_idx) - WINDOW_LEN - 1
    starts = draw_non_overlapping_starts(
        seed=SEED,
        min_start=min_start,
        max_start=max_start,
        window_len=WINDOW_LEN,
        n_windows=1,
    )
    s = starts[0]
    eval_start_ts = shared_idx[s]
    eval_end_ts = shared_idx[s + WINDOW_LEN - 1]

    print(f"Random 180-day window (seed={SEED}):")
    print(f"  Start: {eval_start_ts.date()}")
    print(f"  End:   {eval_end_ts.date()}")
    print(f"  Bars:  {WINDOW_LEN}  (shared equity calendar)")
    print()

    # Run BTC hybrid
    btc_sigs, btc_eq = _btc_hybrid_run(btc, eval_start_ts, eval_end_ts)

    # Rebase BTC equity to 1.0 at window start so per-leg returns compose
    # cleanly
    btc_eq = btc_eq / btc_eq.iloc[0]

    # Collect per-day data on the shared calendar only
    window_idx = shared_idx[(shared_idx >= eval_start_ts) & (shared_idx <= eval_end_ts)]

    sp_win = sp.loc[window_idx]
    tlt_win = tlt.loc[window_idx]
    gld_win = gld.loc[window_idx]

    # Rebase each buy-and-hold leg to 1.0 at window start
    sp_eq = sp_win["close"] / sp_win["close"].iloc[0]
    tlt_eq = tlt_win["close"] / tlt_win["close"].iloc[0]
    gld_eq = gld_win["close"] / gld_win["close"].iloc[0]

    # BTC equity on the shared calendar (forward-fill weekend bars into
    # Monday so we have a value on every equity trading day)
    btc_eq_daily = btc_eq.reindex(window_idx, method="ffill")
    btc_close = btc["close"].reindex(window_idx, method="ffill")

    # Daily returns on each leg
    btc_leg_ret = btc_eq_daily.pct_change().fillna(0.0)
    sp_leg_ret = sp_eq.pct_change().fillna(0.0)
    tlt_leg_ret = tlt_eq.pct_change().fillna(0.0)
    gld_leg_ret = gld_eq.pct_change().fillna(0.0)

    # Equal-weight portfolio daily return + compounding equity
    port_ret = 0.25 * (btc_leg_ret + sp_leg_ret + tlt_leg_ret + gld_leg_ret)
    port_equity = (1.0 + port_ret).cumprod()
    port_equity.iloc[0] = 1.0  # rebase so day 1 = $10,000 exactly
    port_value = port_equity * 10_000.0
    appreciation = (port_equity - 1.0)

    # BTC signals: the engine emits one row per bar. Reindex to the
    # shared calendar (ffill) so every equity day has a signal snapshot.
    btc_sigs_on_calendar = btc_sigs[
        ["signal", "state_label", "target_position"]
    ].reindex(window_idx, method="ffill")

    # Assemble the output table
    out = pd.DataFrame({
        "date": [t.strftime("%Y-%m-%d") for t in window_idx],
        "btc_close": btc_close.values,
        "btc_sig": btc_sigs_on_calendar["signal"].values,
        "btc_regime": btc_sigs_on_calendar["state_label"].values,
        "btc_target": btc_sigs_on_calendar["target_position"].values,
        "btc_leg_ret": btc_leg_ret.values,
        "sp_ret": sp_leg_ret.values,
        "tlt_ret": tlt_leg_ret.values,
        "gld_ret": gld_leg_ret.values,
        "port_ret": port_ret.values,
        "port_value": port_value.values,
        "appreciation": appreciation.values,
    })
    out["dow"] = [t.day_name()[:3] for t in window_idx]

    out_parquet = Path(__file__).parent / "data" / "portfolio_trace_180d.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet)
    print(f"[wrote] {out_parquet}")

    # Summary stats
    start_value = 10_000.0
    end_value = float(out["port_value"].iloc[-1])
    total_return = end_value / start_value - 1
    n_days = len(out)
    years = n_days / 252.0
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else 0.0

    m = compute_metrics(
        out.set_index(pd.DatetimeIndex(out["date"]).tz_localize("UTC"))["port_value"],
        [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=252.0,
    )
    signal_mix = out["btc_sig"].value_counts(dropna=False).to_dict()
    long_days = int((out["btc_target"] > 0.01).sum())
    flat_days = int((out["btc_target"] <= 0.01).sum())

    print("\n" + "=" * 72)
    print("Summary — 180 trading days, equal-weight 4-asset portfolio")
    print("=" * 72)
    print(f"  Start value        : $ {start_value:>10,.2f}")
    print(f"  End value          : $ {end_value:>10,.2f}")
    print(f"  Total return       :   {total_return:>+10.2%}")
    print(f"  CAGR (annualized)  :   {cagr:>+10.2%}")
    print(f"  Sharpe (252/yr)    :   {m.sharpe:>+10.3f}")
    print(f"  Max drawdown       :   {m.max_drawdown:>+10.2%}")
    print(f"  Trading days       :   {n_days:>10d}")
    print(f"  BTC long days      :   {long_days:>10d}  ({long_days/n_days:.0%})")
    print(f"  BTC flat days      :   {flat_days:>10d}  ({flat_days/n_days:.0%})")
    print(f"  BTC signal mix     :   {signal_mix}")

    print("\nPer-leg contribution over the window:")
    print(f"  BTC leg total return   : {(btc_eq_daily.iloc[-1] - 1):>+8.2%}")
    print(f"  ^GSPC buy-hold return  : {(sp_eq.iloc[-1] - 1):>+8.2%}")
    print(f"  TLT buy-hold return    : {(tlt_eq.iloc[-1] - 1):>+8.2%}")
    print(f"  GLD buy-hold return    : {(gld_eq.iloc[-1] - 1):>+8.2%}")

    # Write a compact markdown table too
    out_md = Path(__file__).parent / "data" / "portfolio_trace_180d.md"
    md_lines = [
        "# 180-day portfolio trace — 4-asset equal-weight",
        "",
        f"**Window**: {eval_start_ts.date()} → {eval_end_ts.date()}  "
        f"({n_days} trading days)",
        f"**Start value**: $ {start_value:,.2f}",
        f"**End value**: $ {end_value:,.2f}  "
        f"(**{total_return:+.2%}** total, {cagr:+.2%} CAGR)",
        f"**Sharpe (252/yr)**: {m.sharpe:+.3f}  "
        f"**Max DD**: {m.max_drawdown:+.2%}",
        "",
        "**Per-leg contribution over window**:",
        f"- BTC leg (hybrid): {(btc_eq_daily.iloc[-1] - 1):+.2%}",
        f"- ^GSPC (B&H):     {(sp_eq.iloc[-1] - 1):+.2%}",
        f"- TLT (B&H):       {(tlt_eq.iloc[-1] - 1):+.2%}",
        f"- GLD (B&H):       {(gld_eq.iloc[-1] - 1):+.2%}",
        "",
        "**BTC signal mix**:",
        f"- long days: {long_days}/{n_days} ({long_days/n_days:.0%})",
        f"- flat days: {flat_days}/{n_days} ({flat_days/n_days:.0%})",
        "",
        "## Daily detail (every day)",
        "",
        "| date | dow | btc_close | sig | target | btc_ret | sp_ret | tlt_ret | gld_ret | port_ret | port_value | appr |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in out.iterrows():
        md_lines.append(
            f"| {r['date']} | {r['dow']} | "
            f"{r['btc_close']:,.0f} | {r['btc_sig']} | "
            f"{r['btc_target']:.2f} | "
            f"{r['btc_leg_ret']:+.2%} | {r['sp_ret']:+.2%} | "
            f"{r['tlt_ret']:+.2%} | {r['gld_ret']:+.2%} | "
            f"{r['port_ret']:+.2%} | "
            f"${r['port_value']:,.0f} | {r['appreciation']:+.2%} |"
        )
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"\n[wrote] {out_md}")

    # Compact view: first 5, last 5, and every signal-change day
    print("\n" + "=" * 72)
    print("First 5 days")
    print("=" * 72)
    _print_compact(out.head(5))

    # Signal change days (when btc_target differs from previous day by
    # more than 0.05)
    target_change = out["btc_target"].diff().abs()
    change_days = out[target_change > 0.05]
    if len(change_days) > 0:
        print(f"\n{'=' * 72}")
        print(f"BTC position-change days ({len(change_days)} total, >5% delta)")
        print("=" * 72)
        _print_compact(change_days.head(20))

    print(f"\n{'=' * 72}")
    print("Last 5 days")
    print("=" * 72)
    _print_compact(out.tail(5))


def _print_compact(df: pd.DataFrame) -> None:
    print(
        f"{'date':<11}{'dow':<5}{'btc_close':>10}{'sig':>6}"
        f"{'target':>8}{'btc_ret':>9}{'sp_ret':>9}{'tlt_ret':>9}"
        f"{'gld_ret':>9}{'port_ret':>10}{'value':>12}{'appr':>9}"
    )
    for _, r in df.iterrows():
        print(
            f"{r['date']:<11}{r['dow']:<5}"
            f"{r['btc_close']:>10,.0f}{r['btc_sig']:>6}"
            f"{r['btc_target']:>8.2f}{r['btc_leg_ret']:>+9.2%}"
            f"{r['sp_ret']:>+9.2%}{r['tlt_ret']:>+9.2%}{r['gld_ret']:>+9.2%}"
            f"{r['port_ret']:>+10.2%}${r['port_value']:>10,.0f}"
            f"{r['appreciation']:>+9.2%}"
        )


if __name__ == "__main__":
    main()
