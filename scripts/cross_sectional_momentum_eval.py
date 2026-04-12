"""Cross-sectional momentum evaluation — 20 major US stocks.

Implements the classic Jegadeesh & Titman (1993) "12-1 month" cross-sectional
momentum strategy: rank 20 large-cap US stocks by trailing 12-month return
(skipping the most recent month), go equal-weight long the top 5, rebalance
monthly. Compares against equal-weight buy-and-hold of all 20 and SP500 B&H.

Primary evaluation window: 2019-04-01 -> 2026-04-01 (trailing 7 years).
Secondary: 5-seed multi-seed eval with 12 non-overlapping 6-month windows.

Output:
  scripts/data/cross_sectional_momentum.parquet — raw equity curves
  scripts/CROSS_SECTIONAL_MOMENTUM_RESULTS.md   — comparison table
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore
from signals.model.momentum import CrossSectionalMomentum

# ── Configuration ──────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "XOM", "LLY",
    "AVGO", "COST", "NFLX", "AMD", "ADBE",
]

WINDOW_START = pd.Timestamp("2019-04-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-04-01", tz="UTC")
INITIAL = 10_000.0

RF = historical_usd_rate("2018-2024")
PPY = 252.0

SIX_MONTHS = 126
N_WINDOWS = 12
SEEDS = [42, 7, 100, 999, 1337]


# ── Helpers ────────────────────────────────────────────────────────────────

def load_prices(store: DataStore) -> dict[str, pd.DataFrame]:
    """Load daily OHLCV for all tickers; drop any that are empty."""
    prices: dict[str, pd.DataFrame] = {}
    for sym in TICKERS:
        df = store.load(sym, "1d").sort_index()
        if df.empty:
            print(f"  [WARN] no data for {sym}, skipping")
            continue
        prices[sym] = df
    print(f"  Loaded {len(prices)}/{len(TICKERS)} tickers")
    return prices


def equal_weight_bh(
    prices_dict: dict[str, pd.DataFrame],
    start: str,
    end: str,
    initial_cash: float = 10000.0,
) -> pd.Series:
    """Equal-weight buy-and-hold of all stocks in the universe.

    Buys at the start, holds to the end. No rebalancing. Returns equity curve.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    # Build common date index
    all_dates: set[pd.Timestamp] = set()
    for df in prices_dict.values():
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        all_dates.update(df.index[mask])
    trading_dates = sorted(all_dates)

    if not trading_dates:
        return pd.Series(dtype=float)

    # Find stocks that have data on the first trading day
    first_date = trading_dates[0]
    available: dict[str, float] = {}
    for sym, df in prices_dict.items():
        if first_date in df.index:
            available[sym] = float(df.loc[first_date, "close"])

    if not available:
        return pd.Series(dtype=float)

    # Buy equal weight on first day
    per_stock = initial_cash / len(available)
    holdings: dict[str, float] = {}
    cash = initial_cash
    for sym, price in available.items():
        shares = per_stock / price
        holdings[sym] = shares
        cash -= shares * price

    # Walk forward
    equity_points: list[tuple[pd.Timestamp, float]] = []
    for date in trading_dates:
        equity = cash
        for sym, shares in holdings.items():
            if date in prices_dict[sym].index:
                equity += shares * float(prices_dict[sym].loc[date, "close"])
        equity_points.append((date, equity))

    ts, eq = zip(*equity_points, strict=True)
    return pd.Series(eq, index=pd.DatetimeIndex(ts), name="equity")


def sp500_bh(
    store: DataStore, start: str, end: str, initial_cash: float = 10000.0,
) -> pd.Series:
    """SP500 buy-and-hold equity curve."""
    df = store.load("GSPC", "1d").sort_index()
    if df.empty:
        # Try alternate name
        df = store.load("^GSPC", "1d").sort_index()
    if df.empty:
        print("  [WARN] no SP500 (^GSPC/GSPC) data found")
        return pd.Series(dtype=float)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    mask = (df.index >= start_ts) & (df.index <= end_ts)
    df = df.loc[mask]

    if df.empty:
        return pd.Series(dtype=float)

    # Rebase to initial_cash
    rebase = initial_cash / float(df["close"].iloc[0])
    equity = df["close"] * rebase
    equity.name = "equity"
    return equity


def trim_to_common(
    curves: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Trim all equity curves to their common date range."""
    if not curves:
        return curves
    common_start = max(c.index[0] for c in curves.values() if len(c) > 0)
    common_end = min(c.index[-1] for c in curves.values() if len(c) > 0)
    result = {}
    for name, c in curves.items():
        trimmed = c.loc[(c.index >= common_start) & (c.index <= common_end)]
        if len(trimmed) > 0:
            # Rebase to start at INITIAL
            result[name] = trimmed * (INITIAL / trimmed.iloc[0])
    return result


def metrics_row(name: str, equity: pd.Series) -> dict:
    """Compute metrics and return as a flat dict."""
    m = compute_metrics(equity, trades=[], risk_free_rate=RF, periods_per_year=PPY)
    return {
        "strategy": name,
        "sharpe": m.sharpe,
        "cagr": m.cagr,
        "max_drawdown": m.max_drawdown,
        "calmar": m.calmar,
        "final_equity": m.final_equity,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    store = DataStore(SETTINGS.data.dir)

    print("=" * 70)
    print("Cross-Sectional Momentum Evaluation")
    print("=" * 70)

    # Load data
    print("\n[1] Loading price data...")
    prices = load_prices(store)
    if len(prices) < 5:
        print("  FATAL: fewer than 5 tickers loaded, cannot run momentum eval")
        return

    # ── Primary: trailing 7-year window ────────────────────────────────
    print("\n[2] Running trailing 7-year evaluation (2019-04-01 -> 2026-04-01)...")

    mom = CrossSectionalMomentum(
        lookback_days=252, skip_days=21, n_long=5, rebalance_freq=21,
        commission_bps=5.0, slippage_bps=5.0,
    )

    start_str = str(WINDOW_START.date())
    end_str = str(WINDOW_END.date())

    eq_momentum = mom.backtest(prices, start=start_str, end=end_str, initial_cash=INITIAL)
    eq_ew_bh = equal_weight_bh(prices, start=start_str, end=end_str, initial_cash=INITIAL)
    eq_sp = sp500_bh(store, start=start_str, end=end_str, initial_cash=INITIAL)

    curves = {"momentum_top5": eq_momentum, "ew_20_bh": eq_ew_bh}
    if len(eq_sp) > 0:
        curves["sp500_bh"] = eq_sp

    curves = trim_to_common(curves)

    print("\n  Trailing 7-Year Results:")
    print(f"  {'Strategy':<20} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Final$':>10}")
    print("  " + "-" * 58)

    rows_primary: list[dict] = []
    for name, eq in curves.items():
        row = metrics_row(name, eq)
        rows_primary.append(row)
        print(
            f"  {row['strategy']:<20} {row['sharpe']:>8.3f} "
            f"{row['cagr']:>7.1%} {row['max_drawdown']:>7.1%} "
            f"{row['final_equity']:>10,.0f}"
        )

    # ── Secondary: multi-seed windowed eval ────────────────────────────
    print("\n[3] Running 5-seed multi-seed evaluation (12 x 6-month windows)...")

    # Build a common date index for window sampling
    all_dates_set: set[pd.Timestamp] = set()
    for df in prices.values():
        all_dates_set.update(df.index)
    all_dates = sorted(all_dates_set)

    # Find eligible range: need lookback+skip history before the window
    warmup = mom._required_history() + 5  # small extra pad
    min_start = warmup
    max_start = len(all_dates) - SIX_MONTHS - 1

    seed_results: list[dict] = []

    for seed in SEEDS:
        starts = draw_non_overlapping_starts(
            seed=seed,
            min_start=min_start,
            max_start=max_start,
            window_len=SIX_MONTHS,
            n_windows=N_WINDOWS,
        )

        for s in starts:
            w_start = all_dates[s]
            w_end = all_dates[min(s + SIX_MONTHS, len(all_dates) - 1)]

            eq_m = mom.backtest(
                prices,
                start=str(w_start.date()),
                end=str(w_end.date()),
                initial_cash=INITIAL,
            )
            eq_b = equal_weight_bh(
                prices,
                start=str(w_start.date()),
                end=str(w_end.date()),
                initial_cash=INITIAL,
            )

            if len(eq_m) < 20 or len(eq_b) < 20:
                continue

            m_m = compute_metrics(eq_m, trades=[], risk_free_rate=RF, periods_per_year=PPY)
            m_b = compute_metrics(eq_b, trades=[], risk_free_rate=RF, periods_per_year=PPY)

            seed_results.append({
                "seed": seed,
                "window_start": str(w_start.date()),
                "window_end": str(w_end.date()),
                "mom_sharpe": m_m.sharpe,
                "mom_cagr": m_m.cagr,
                "mom_mdd": m_m.max_drawdown,
                "ew_sharpe": m_b.sharpe,
                "ew_cagr": m_b.cagr,
                "ew_mdd": m_b.max_drawdown,
            })

    if seed_results:
        df_seeds = pd.DataFrame(seed_results)
        n_total = len(df_seeds)
        n_beats = int((df_seeds["mom_sharpe"] > df_seeds["ew_sharpe"]).sum())

        avg_mom_sharpe = df_seeds["mom_sharpe"].mean()
        avg_ew_sharpe = df_seeds["ew_sharpe"].mean()
        avg_mom_cagr = df_seeds["mom_cagr"].mean()
        avg_ew_cagr = df_seeds["ew_cagr"].mean()

        print(f"\n  Multi-Seed Summary ({n_total} windows across {len(SEEDS)} seeds):")
        print(f"  {'Metric':<20} {'Momentum':>10} {'EW-20 B&H':>10}")
        print("  " + "-" * 42)
        print(f"  {'Avg Sharpe':<20} {avg_mom_sharpe:>10.3f} {avg_ew_sharpe:>10.3f}")
        print(f"  {'Avg CAGR':<20} {avg_mom_cagr:>9.1%} {avg_ew_cagr:>9.1%}")
        print(f"  Momentum beats EW B&H on Sharpe: {n_beats}/{n_total} windows ({n_beats/n_total:.0%})")
    else:
        df_seeds = pd.DataFrame()
        print("  [WARN] No valid windows produced results")

    # ── Save outputs ───────────────────────────────────────────────────
    print("\n[4] Saving outputs...")

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Save equity curves to parquet
    eq_df = pd.DataFrame(curves)
    eq_df.to_parquet(data_dir / "cross_sectional_momentum.parquet")

    # Write results markdown
    md_path = Path(__file__).parent / "CROSS_SECTIONAL_MOMENTUM_RESULTS.md"
    with open(md_path, "w") as f:
        f.write("# Cross-Sectional Momentum Results\n\n")
        f.write("**Strategy:** 12-1 month momentum (Jegadeesh & Titman 1993)\n")
        f.write("- Universe: 20 major US stocks\n")
        f.write("- Lookback: 252 days, skip: 21 days, top 5 equal-weight, monthly rebalance\n")
        f.write("- Costs: 5 bps commission + 5 bps slippage per rebalance trade\n")
        f.write(f"- Risk-free rate: {RF:.2%} (3M T-bill avg 2018-2024)\n")
        f.write(f"- Annualization: {PPY:.0f} days/year\n\n")

        f.write("## Trailing 7-Year (2019-04-01 to 2026-04-01)\n\n")
        f.write(f"| {'Strategy':<20} | {'Sharpe':>8} | {'CAGR':>8} | {'MaxDD':>8} | {'Final$':>10} |\n")
        f.write(f"|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*12}|\n")
        for row in rows_primary:
            f.write(
                f"| {row['strategy']:<20} | {row['sharpe']:>8.3f} | "
                f"{row['cagr']:>7.1%} | {row['max_drawdown']:>7.1%} | "
                f"{row['final_equity']:>10,.0f} |\n"
            )

        if not df_seeds.empty:
            f.write("\n## Multi-Seed Windowed Evaluation\n\n")
            f.write(f"- {len(SEEDS)} seeds x {N_WINDOWS} non-overlapping 6-month windows\n")
            f.write(f"- Total windows evaluated: {n_total}\n\n")
            f.write(f"| {'Metric':<20} | {'Momentum':>10} | {'EW-20 B&H':>10} |\n")
            f.write(f"|{'-'*22}|{'-'*12}|{'-'*12}|\n")
            f.write(f"| {'Avg Sharpe':<20} | {avg_mom_sharpe:>10.3f} | {avg_ew_sharpe:>10.3f} |\n")
            f.write(f"| {'Avg CAGR':<20} | {avg_mom_cagr:>9.1%} | {avg_ew_cagr:>9.1%} |\n")
            f.write(f"\nMomentum beats EW B&H on Sharpe: **{n_beats}/{n_total}** windows ({n_beats/n_total:.0%})\n")

        f.write("\n---\n")
        f.write("*Generated by `scripts/cross_sectional_momentum_eval.py`*\n")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Results: {md_path}")
    print(f"  Data:    {data_dir / 'cross_sectional_momentum.parquet'}")


if __name__ == "__main__":
    main()
