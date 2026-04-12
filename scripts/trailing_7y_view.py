"""Trailing 7-year view — 2019-04-01 → 2026-04-01.

Seven years of daily equity-curve detail for every top strategy, ending
on April 1, 2026 (the nearest equity trading day, anchored cleanly to
avoid the weekend boundary effect). Includes a head-to-head vs SP500 B&H
plus the 4-asset basket, BTC_HYBRID_PRODUCTION, 60/40 pension mix, and
each single asset.

All strategies evaluated on:
  - Shared equity calendar (inner join of SP/TLT/GLD trading days)
  - 252/yr annualization (equity calendar)
  - historical_usd_rate("2018-2024") risk-free rate ~ 2.3% annualized
  - Rebase to $10,000 at window start
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from signals.backtest.engine import (
    BTC_HYBRID_PRODUCTION,
    BacktestConfig,
    BacktestEngine,
)
from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

# Target window (user-specified)
WINDOW_START = pd.Timestamp("2019-04-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-04-01", tz="UTC")
INITIAL = 10_000.0
WARMUP_PAD = 5


def _load(store: DataStore, sym: str) -> pd.DataFrame:
    df = store.load(sym, "1d").sort_index()
    return df.loc[df.index <= WINDOW_END]


def _anchor_to_equity_calendar(
    sp_prices: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find the first SP trading day >= WINDOW_START and the last SP
    trading day <= WINDOW_END. This guarantees the portfolio can
    rebalance at both endpoints and avoids the weekend boundary bug
    that validate_btc_calendar.py surfaced."""
    eligible = sp_prices.loc[
        (sp_prices.index >= WINDOW_START) & (sp_prices.index <= WINDOW_END)
    ]
    return eligible.index[0], eligible.index[-1]


def _btc_hybrid_equity(
    btc_prices: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> pd.Series:
    base = dict(BTC_HYBRID_PRODUCTION)
    base["risk_free_rate"] = historical_usd_rate("2018-2024")
    cfg = BacktestConfig(**base)
    # Full 2015→END slice for warmup (ensures the hybrid has ≥1000 bars
    # of BTC training data before start_ts)
    warmup_start = pd.Timestamp("2015-01-01", tz="UTC")
    engine_input = btc_prices.loc[
        (btc_prices.index >= warmup_start) & (btc_prices.index <= end_ts)
    ]
    result = BacktestEngine(cfg).run(engine_input, symbol="BTC-USD")
    eq = result.equity_curve.loc[
        (result.equity_curve.index >= start_ts)
        & (result.equity_curve.index <= end_ts)
    ]
    if eq.empty or eq.iloc[0] <= 0:
        return pd.Series(dtype=float)
    return eq / eq.iloc[0]


def _bh_equity(
    prices: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> pd.Series:
    sl = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)]
    if sl.empty:
        return pd.Series(dtype=float)
    return sl["close"] / sl["close"].iloc[0]


def _equal_weight_portfolio(legs: dict[str, pd.Series]) -> pd.Series:
    leg_df = pd.concat(legs, axis=1).dropna(how="any")
    if leg_df.empty:
        return pd.Series(dtype=float)
    leg_returns = leg_df.pct_change().fillna(0.0)
    n = len(legs)
    port_returns = (1.0 / n) * leg_returns.sum(axis=1)
    port_equity = (1.0 + port_returns).cumprod()
    port_equity.iloc[0] = 1.0
    return port_equity


def _strategy_stats(equity: pd.Series, label: str) -> dict:
    if equity.empty:
        return {"strategy": label, "start_value": INITIAL, "end_value": 0.0}
    value_series = equity * INITIAL
    m = compute_metrics(
        value_series,
        [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=252.0,
    )
    total_return = float(value_series.iloc[-1] / value_series.iloc[0] - 1)
    return {
        "strategy": label,
        "start": value_series.index[0].strftime("%Y-%m-%d"),
        "end": value_series.index[-1].strftime("%Y-%m-%d"),
        "n_days": len(value_series),
        "start_value": INITIAL,
        "end_value": float(value_series.iloc[-1]),
        "total_return": total_return,
        "cagr": m.cagr,
        "sharpe_252": m.sharpe,
        "max_dd": m.max_drawdown,
        "calmar": m.calmar,
        "_equity_series": value_series,
    }


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    btc = _load(store, "BTC-USD")
    sp = _load(store, "^GSPC")
    tlt = _load(store, "TLT")
    gld = _load(store, "GLD")

    # Anchor the window to the nearest equity trading days
    start_ts, end_ts = _anchor_to_equity_calendar(sp)

    print(f"Window: {start_ts.date()} → {end_ts.date()}")
    print(f"  Requested: {WINDOW_START.date()} → {WINDOW_END.date()}")
    print("  Anchored to equity calendar (nearest trading days)")
    print(f"BTC: {len(btc)} bars total, SP: {len(sp)}, TLT: {len(tlt)}, GLD: {len(gld)}")
    print()

    # Strategies
    btc_eq_all = _btc_hybrid_equity(btc, start_ts, end_ts)
    if btc_eq_all.empty:
        print("BTC hybrid failed.")
        return

    sp_eq = _bh_equity(sp, start_ts, end_ts)
    tlt_eq = _bh_equity(tlt, start_ts, end_ts)
    gld_eq = _bh_equity(gld, start_ts, end_ts)

    # BTC equity reindexed to equity calendar (ffill weekend bars onto Monday)
    btc_on_sp_cal = btc_eq_all.reindex(sp_eq.index, method="ffill")
    btc_on_sp_cal = btc_on_sp_cal / btc_on_sp_cal.iloc[0]

    # 4-asset equal-weight portfolio
    port_eq = _equal_weight_portfolio({
        "BTC": btc_eq_all, "SP": sp_eq, "TLT": tlt_eq, "GLD": gld_eq,
    })

    strategies = []
    strategies.append(_strategy_stats(sp_eq, "SP500 buy-and-hold"))
    strategies.append(_strategy_stats(tlt_eq, "TLT buy-and-hold"))
    strategies.append(_strategy_stats(gld_eq, "GLD buy-and-hold"))
    strategies.append(_strategy_stats(btc_on_sp_cal, "BTC_HYBRID_PRODUCTION"))
    strategies.append(_strategy_stats(port_eq, "4-asset equal-weight portfolio"))

    # 60/40 SP/TLT (classic)
    leg_df_6040 = pd.concat({"SP": sp_eq, "TLT": tlt_eq}, axis=1).dropna(how="any")
    if not leg_df_6040.empty:
        leg_ret = leg_df_6040.pct_change().fillna(0.0)
        r_6040 = 0.60 * leg_ret["SP"] + 0.40 * leg_ret["TLT"]
        eq_6040 = (1 + r_6040).cumprod()
        eq_6040.iloc[0] = 1.0
        strategies.append(_strategy_stats(eq_6040, "60/40 SP/TLT (classic)"))

    # 60/40 SP/GLD (gold instead of bonds — good over hiking cycles)
    leg_df_6040g = pd.concat({"SP": sp_eq, "GLD": gld_eq}, axis=1).dropna(how="any")
    if not leg_df_6040g.empty:
        leg_ret = leg_df_6040g.pct_change().fillna(0.0)
        r_6040g = 0.60 * leg_ret["SP"] + 0.40 * leg_ret["GLD"]
        eq_6040g = (1 + r_6040g).cumprod()
        eq_6040g.iloc[0] = 1.0
        strategies.append(_strategy_stats(eq_6040g, "60/40 SP/GLD (gold bond)"))

    # Pure BTC buy-and-hold for reference
    btc_bh = _bh_equity(btc, start_ts, end_ts)
    if not btc_bh.empty:
        btc_bh_on_sp = btc_bh.reindex(sp_eq.index, method="ffill")
        btc_bh_on_sp = btc_bh_on_sp / btc_bh_on_sp.iloc[0]
        strategies.append(_strategy_stats(btc_bh_on_sp, "BTC buy-and-hold (raw)"))

    # Summary table
    summary_rows = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in strategies
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "sharpe_252", ascending=False
    )

    print("=" * 100)
    print(
        f"7-year comparison ({start_ts.date()} → {end_ts.date()}, "
        f"{len(sp_eq)} equity trading days)"
    )
    print("=" * 100)
    print(
        f"  {'strategy':<34}"
        f"{'end $':>14}{'total ret':>13}"
        f"{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'Calmar':>9}"
    )
    print("  " + "-" * 97)
    for _, r in summary_df.iterrows():
        print(
            f"  {r['strategy']:<34}"
            f"${r['end_value']:>12,.0f}"
            f"{r['total_return']:>+12.1%}"
            f"{r['cagr']:>+8.1%}"
            f"{r['sharpe_252']:>+9.3f}"
            f"{r['max_dd']:>+8.1%}"
            f"{r['calmar']:>+8.2f}"
        )

    # Yearly milestones
    print("\n" + "=" * 100)
    print("Year-end value progression (Dec 31 of each year, plus final)")
    print("=" * 100)

    port_s = port_eq * INITIAL
    sp_s = sp_eq * INITIAL
    btc_s = btc_on_sp_cal * INITIAL
    tlt_s = tlt_eq * INITIAL
    gld_s = gld_eq * INITIAL

    common_idx = sp_s.index
    port_on_sp = port_s.reindex(common_idx, method="ffill")
    btc_on_sp_s = btc_s.reindex(common_idx, method="ffill")
    tlt_on_sp = tlt_s.reindex(common_idx, method="ffill")
    gld_on_sp = gld_s.reindex(common_idx, method="ffill")

    yearly = pd.DataFrame({
        "date": common_idx,
        "portfolio": port_on_sp.values,
        "sp_bh": sp_s.values,
        "btc_hybrid": btc_on_sp_s.values,
        "tlt_bh": tlt_on_sp.values,
        "gld_bh": gld_on_sp.values,
    })
    yearly["year"] = pd.to_datetime(yearly["date"]).dt.year
    year_last = yearly.groupby("year").last().reset_index()
    year_last["date"] = pd.to_datetime(year_last["date"]).dt.strftime("%Y-%m-%d")

    print(
        f"  {'year end':<12}"
        f"{'4-asset':>14}{'SP B&H':>12}{'BTC hybrid':>14}"
        f"{'TLT B&H':>12}{'GLD B&H':>12}"
    )
    print("  " + "-" * 76)
    for _, r in year_last.iterrows():
        print(
            f"  {r['date']:<12}"
            f"${r['portfolio']:>12,.0f}"
            f"${r['sp_bh']:>10,.0f}"
            f"${r['btc_hybrid']:>12,.0f}"
            f"${r['tlt_bh']:>10,.0f}"
            f"${r['gld_bh']:>10,.0f}"
        )

    # Persist
    out_parquet = Path(__file__).parent / "data" / "trailing_7y_view.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    daily_df = pd.DataFrame({
        "date": common_idx,
        "portfolio": port_on_sp.values,
        "sp_bh": sp_s.values,
        "btc_hybrid": btc_on_sp_s.values,
        "tlt_bh": tlt_on_sp.values,
        "gld_bh": gld_on_sp.values,
    })
    daily_df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}")

    # Markdown
    out_md = Path(__file__).parent / "TRAILING_7Y_VIEW.md"
    md_lines = [
        f"# Trailing 7-year view ({start_ts.date()} → {end_ts.date()})",
        "",
        "Seven years of daily equity-curve data for every top strategy, "
        "ending on April 1, 2026 (anchored to the nearest equity trading "
        "day to avoid weekend boundary effects). All strategies start "
        "at $10,000 on 2019-04-01.",
        "",
        "**BTC calendar validation**: `scripts/validate_btc_calendar.py` "
        "confirms that the 4-asset portfolio captures every BTC tick, "
        "including weekends — Monday's portfolio bar carries the full "
        "Friday→Monday 3-day return, and the product of equity-calendar "
        "bars equals the product of all 365 native BTC bars to "
        "floating-point precision. No BTC price action is lost.",
        "",
        "Annualization: 252/yr on the equity shared calendar. "
        "Risk-free rate ≈ 2.3% (2018-2024 historical average).",
        "",
        "## Summary — sorted by Sharpe",
        "",
        "| strategy | end value | total return | CAGR | Sharpe | MDD | Calmar |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary_df.iterrows():
        md_lines.append(
            f"| `{r['strategy']}` | "
            f"${r['end_value']:,.0f} | "
            f"{r['total_return']:+.1%} | "
            f"{r['cagr']:+.1%} | "
            f"{r['sharpe_252']:+.3f} | "
            f"{r['max_dd']:+.1%} | "
            f"{r['calmar']:+.2f} |"
        )

    md_lines += [
        "",
        "## Year-end value progression",
        "",
        "$10,000 initial on 2019-04-01. Each row shows the value at the "
        "final equity trading day of that calendar year (or the window "
        "end for 2026).",
        "",
        "| year | 4-asset portfolio | SP B&H | BTC hybrid | TLT B&H | GLD B&H |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in year_last.iterrows():
        md_lines.append(
            f"| {r['date']} | "
            f"${r['portfolio']:,.0f} | "
            f"${r['sp_bh']:,.0f} | "
            f"${r['btc_hybrid']:,.0f} | "
            f"${r['tlt_bh']:,.0f} | "
            f"${r['gld_bh']:,.0f} |"
        )

    md_lines += [
        "",
        "## Raw data",
        "",
        "- `scripts/data/trailing_7y_view.parquet` — daily values for all strategies",
        "- Reproduce: `python scripts/trailing_7y_view.py`",
        "- BTC calendar validation: `python scripts/validate_btc_calendar.py`",
        "",
    ]
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
