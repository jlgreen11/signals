"""Trailing 3-year head-to-head: 4-asset portfolio vs SP500 B&H vs BTC_HYBRID_PRODUCTION.

Single-window, end-anchored comparison. Runs each strategy over the
most recent 3 calendar years of available data (2022-01-01 → 2024-12-31
= ~756 trading days) and reports start/end value, CAGR, Sharpe, max
drawdown, and monthly equity milestones.

This is a long-horizon look — 10-seed random-window evals said the
4-asset basket ties SP B&H on 6-month Sharpe. A 3-year window tells
a different story because BTC's CAGR has more bars to compound.

Output:
  scripts/TRAILING_3Y_VIEW.md       — human-readable table + narrative
  scripts/data/trailing_3y_view.parquet — daily portfolio values
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

START = pd.Timestamp("2022-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")
INITIAL = 10_000.0
WARMUP_PAD = 5


def _load(store: DataStore, sym: str) -> pd.DataFrame:
    df = store.load(sym, "1d").sort_index()
    return df.loc[df.index <= END]


def _btc_hybrid_equity(btc_prices: pd.DataFrame) -> pd.Series:
    """Walk-forward the BTC hybrid over the full 2015-2024 range, then
    restrict to the trailing 3-year window. This ensures the hybrid
    sees its required ~765 bars of warmup before 2022-01-01."""
    base = dict(BTC_HYBRID_PRODUCTION)
    base["risk_free_rate"] = historical_usd_rate("2018-2024")
    cfg = BacktestConfig(**base)

    # Engine input: everything from 2015 to END so the walk-forward
    # has full history available.
    start_buffer = pd.Timestamp("2015-01-01", tz="UTC")
    engine_input = btc_prices.loc[
        (btc_prices.index >= start_buffer) & (btc_prices.index <= END)
    ]
    result = BacktestEngine(cfg).run(engine_input, symbol="BTC-USD")

    # Restrict to the trailing 3-year window
    eq = result.equity_curve.loc[result.equity_curve.index >= START]
    if eq.empty or eq.iloc[0] <= 0:
        return pd.Series(dtype=float)
    # Rebase so day 1 of the window = 1.0
    return eq / eq.iloc[0]


def _bh_equity(prices: pd.DataFrame, start_ts: pd.Timestamp) -> pd.Series:
    sl = prices.loc[(prices.index >= start_ts) & (prices.index <= END)]
    if sl.empty:
        return pd.Series(dtype=float)
    return sl["close"] / sl["close"].iloc[0]


def _equal_weight_portfolio(
    btc_eq: pd.Series,
    sp_eq: pd.Series,
    tlt_eq: pd.Series,
    gld_eq: pd.Series,
) -> pd.Series:
    leg_df = pd.concat(
        {"BTC": btc_eq, "SP": sp_eq, "TLT": tlt_eq, "GLD": gld_eq},
        axis=1,
    ).dropna(how="any")
    if leg_df.empty:
        return pd.Series(dtype=float)
    leg_returns = leg_df.pct_change().fillna(0.0)
    port_returns = 0.25 * leg_returns.sum(axis=1)
    port_equity = (1.0 + port_returns).cumprod()
    port_equity.iloc[0] = 1.0
    return port_equity


def _strategy_stats(equity: pd.Series, label: str) -> dict:
    """Compute stats for an equity curve starting at 1.0."""
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
    n_days = len(value_series)
    return {
        "strategy": label,
        "start": value_series.index[0].strftime("%Y-%m-%d"),
        "end": value_series.index[-1].strftime("%Y-%m-%d"),
        "n_days": n_days,
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

    print(f"Window: {START.date()} → {END.date()}")
    print(f"BTC: {len(btc)} bars total, SP: {len(sp)}, TLT: {len(tlt)}, GLD: {len(gld)}")
    print()

    # Run each strategy
    btc_eq_all = _btc_hybrid_equity(btc)
    if btc_eq_all.empty:
        print("BTC hybrid failed.")
        return

    sp_eq = _bh_equity(sp, START)
    tlt_eq = _bh_equity(tlt, START)
    gld_eq = _bh_equity(gld, START)

    # Align BTC to the equity calendar for portfolio construction
    port_eq = _equal_weight_portfolio(btc_eq_all, sp_eq, tlt_eq, gld_eq)

    # Strategy results
    strategies = []
    strategies.append(_strategy_stats(sp_eq, "SP500 buy-and-hold"))
    strategies.append(_strategy_stats(tlt_eq, "TLT buy-and-hold"))
    strategies.append(_strategy_stats(gld_eq, "GLD buy-and-hold"))
    # BTC hybrid on equity calendar (reindex to SP's index, ffill)
    btc_eq_equity_cal = btc_eq_all.reindex(sp_eq.index, method="ffill")
    btc_eq_equity_cal = btc_eq_equity_cal / btc_eq_equity_cal.iloc[0]
    strategies.append(_strategy_stats(btc_eq_equity_cal, "BTC_HYBRID_PRODUCTION"))
    strategies.append(_strategy_stats(port_eq, "4-asset equal-weight portfolio"))

    # Also a 60/40 SP/TLT for comparison (classic pension mix)
    leg_df_6040 = pd.concat({"SP": sp_eq, "TLT": tlt_eq}, axis=1).dropna(how="any")
    if not leg_df_6040.empty:
        leg_ret = leg_df_6040.pct_change().fillna(0.0)
        r_6040 = 0.60 * leg_ret["SP"] + 0.40 * leg_ret["TLT"]
        eq_6040 = (1 + r_6040).cumprod()
        eq_6040.iloc[0] = 1.0
        strategies.append(_strategy_stats(eq_6040, "60/40 SP/TLT (classic)"))

    # Summary table
    summary_cols = [
        "strategy", "end_value", "total_return",
        "cagr", "sharpe_252", "max_dd", "calmar",
    ]
    summary_rows = [{c: s[c] for c in summary_cols} for s in strategies]
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "sharpe_252", ascending=False
    )

    print("=" * 88)
    print(f"Trailing {START.year}-{END.year} comparison (~3 years, single window)")
    print("=" * 88)
    print(
        f"  {'strategy':<34}"
        f"{'end $':>12}{'total ret':>12}"
        f"{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'Calmar':>9}"
    )
    print("  " + "-" * 86)
    for _, r in summary_df.iterrows():
        print(
            f"  {r['strategy']:<34}"
            f"${r['end_value']:>10,.0f}"
            f"{r['total_return']:>+11.1%}"
            f"{r['cagr']:>+8.1%}"
            f"{r['sharpe_252']:>+9.3f}"
            f"{r['max_dd']:>+8.1%}"
            f"{r['calmar']:>+8.2f}"
        )

    # Monthly milestones for the top 3 strategies
    print("\n" + "=" * 88)
    print("Quarterly portfolio value — $10,000 → where you ended each quarter")
    print("=" * 88)

    # Pick the monthly-ish resampled points
    port_s = port_eq * INITIAL
    sp_s = sp_eq * INITIAL
    btc_s = btc_eq_equity_cal * INITIAL

    # Align all 3 to SP's index and forward-fill
    common_idx = sp_s.index
    port_on_sp = port_s.reindex(common_idx, method="ffill")
    btc_on_sp = btc_s.reindex(common_idx, method="ffill")

    # Quarterly samples
    q = pd.DataFrame({
        "date": common_idx,
        "portfolio": port_on_sp.values,
        "sp_bh": sp_s.values,
        "btc_hybrid": btc_on_sp.values,
    })
    q["month"] = pd.to_datetime(q["date"]).dt.to_period("Q")
    q_last = q.groupby("month").last().reset_index()
    q_last["date"] = pd.to_datetime(q_last["date"]).dt.strftime("%Y-%m-%d")

    print(
        f"  {'qtr end':<12}"
        f"{'4-asset port':>18}{'SP B&H':>14}{'BTC hybrid':>14}"
    )
    print("  " + "-" * 58)
    for _, r in q_last.iterrows():
        print(
            f"  {r['date']:<12}"
            f"${r['portfolio']:>16,.0f}"
            f"${r['sp_bh']:>12,.0f}"
            f"${r['btc_hybrid']:>12,.0f}"
        )

    # Persist
    out_parquet = Path(__file__).parent / "data" / "trailing_3y_view.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    daily_df = pd.DataFrame({
        "date": common_idx,
        "portfolio": port_on_sp.values,
        "sp_bh": sp_s.values,
        "btc_hybrid": btc_on_sp.values,
    })
    daily_df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}")

    # Markdown output
    out_md = Path(__file__).parent / "TRAILING_3Y_VIEW.md"
    md_lines = [
        f"# Trailing 3-year view ({START.date()} → {END.date()})",
        "",
        "Single-window, end-anchored comparison of the project's top "
        "strategies against SP500 buy-and-hold over the most recent "
        "3 calendar years of available data. Everything starts at $10,000 "
        "on 2022-01-03 (first equity trading day ≥ 2022-01-01) and ends "
        "on 2024-12-31.",
        "",
        "Annualization: 252/yr (equity shared calendar). Risk-free rate "
        "≈ 2.3% (2018-2024 average).",
        "",
        "## Summary",
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
        "## Quarterly portfolio value",
        "",
        "$10,000 initial → quarter-end value for each strategy.",
        "",
        "| quarter end | 4-asset portfolio | SP B&H | BTC hybrid |",
        "|---|---:|---:|---:|",
    ]
    for _, r in q_last.iterrows():
        md_lines.append(
            f"| {r['date']} | "
            f"${r['portfolio']:,.0f} | "
            f"${r['sp_bh']:,.0f} | "
            f"${r['btc_hybrid']:,.0f} |"
        )

    md_lines += [
        "",
        "## Interpretation",
        "",
        "Read the Sharpe column as the risk-adjusted winner and CAGR as "
        "the absolute-return winner. A single 3-year window is NOT "
        "statistically robust (N=1), so treat this as a narrative "
        "snapshot rather than evidence. For multi-seed averages see "
        "`scripts/PORTFOLIO_VS_SP_BH.md` and `scripts/BROAD_COMPARISON.md`.",
        "",
        "## Raw data",
        "",
        "- `scripts/data/trailing_3y_view.parquet` — daily portfolio values "
        "for all 3 headline strategies",
        "- Reproduce: `python scripts/trailing_3y_view.py`",
        "",
    ]
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
