"""Multi-asset time-series momentum (TSMOM) evaluation.

Tests the Moskowitz-Ooi-Pedersen (2012) trend-following signal across
a diversified basket of macro asset classes: equity indices, bonds,
commodities, currencies, and crypto. The academic literature shows
TSMOM works on these asset classes (not individual stocks).

Universe:
  ^GSPC  — S&P 500 equity index
  TLT    — Long US Treasuries (20+ yr)
  IEF    — Intermediate US Treasuries (7-10 yr)
  GLD    — Gold
  USO    — Crude oil proxy
  UUP    — US Dollar index
  EFA    — International equities (EAFE)
  BTC-USD — Bitcoin

Lookbacks: 21d (1-month), 63d (3-month), 252d (12-month), Combined (avg)
Benchmark: equal-weight B&H of the same universe, SP500 B&H alone

Trailing 7-year window: 2019-04-01 -> 2026-04-01
Multi-seed eval: 5 seeds x 12 non-overlapping 6-month windows
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
from signals.model.tsmom import TimeSeriesMomentum

# ============================================================
# Constants
# ============================================================

UNIVERSE = ["^GSPC", "TLT", "IEF", "GLD", "USO", "UUP", "EFA", "BTC-USD"]
LOOKBACKS = [21, 63, 252]
WINDOW_START = pd.Timestamp("2019-04-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-04-01", tz="UTC")
INITIAL = 10_000.0
RISK_FREE = historical_usd_rate("2018-2024")
PERIODS_PER_YEAR = 252.0

# Multi-seed eval
SIX_MONTHS = 126
N_WINDOWS = 12
SEEDS = [42, 7, 100, 999, 1337]


# ============================================================
# Helpers
# ============================================================


def _load_universe(store: DataStore) -> dict[str, pd.DataFrame]:
    """Load all available assets from the universe."""
    available: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE:
        df = store.load(sym, "1d").sort_index()
        if df.empty:
            print(f"  [WARN] {sym} not available, skipping")
            continue
        available[sym] = df
    return available


def _anchor_to_equity_calendar(
    prices_dict: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find the first/last common trading day in [start, end]."""
    # Use ^GSPC as the equity calendar anchor
    sp = prices_dict.get("^GSPC")
    if sp is None:
        # Fallback: use any available asset
        sp = next(iter(prices_dict.values()))
    eligible = sp.loc[(sp.index >= start) & (sp.index <= end)]
    if eligible.empty:
        raise ValueError(f"No trading days in [{start}, {end}]")
    return eligible.index[0], eligible.index[-1]


def _equal_weight_bh(
    prices_dict: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial: float = 10_000.0,
) -> pd.Series:
    """Equal-weight buy-and-hold of all assets in prices_dict."""
    legs: dict[str, pd.Series] = {}
    for sym, df in prices_dict.items():
        sl = df.loc[(df.index >= start) & (df.index <= end), "close"].dropna()
        if not sl.empty:
            legs[sym] = sl / sl.iloc[0]
    if not legs:
        return pd.Series(dtype=float)
    leg_df = pd.concat(legs, axis=1).dropna(how="any")
    if leg_df.empty:
        return pd.Series(dtype=float)
    leg_returns = leg_df.pct_change().fillna(0.0)
    n = len(legs)
    port_returns = (1.0 / n) * leg_returns.sum(axis=1)
    port_equity = (1.0 + port_returns).cumprod() * initial
    port_equity.iloc[0] = initial
    return port_equity


def _sp_bh(
    prices_dict: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial: float = 10_000.0,
) -> pd.Series:
    """SP500 buy-and-hold equity curve."""
    sp = prices_dict.get("^GSPC")
    if sp is None:
        return pd.Series(dtype=float)
    sl = sp.loc[(sp.index >= start) & (sp.index <= end), "close"].dropna()
    if sl.empty:
        return pd.Series(dtype=float)
    return sl / sl.iloc[0] * initial


def _strategy_stats(equity: pd.Series, label: str) -> dict:
    """Compute summary stats for an equity curve."""
    if equity.empty or len(equity) < 2:
        return {
            "strategy": label, "start_value": INITIAL, "end_value": 0.0,
            "total_return": 0.0, "cagr": 0.0, "sharpe_252": 0.0,
            "max_dd": 0.0, "calmar": 0.0,
        }
    m = compute_metrics(
        equity, [], risk_free_rate=RISK_FREE, periods_per_year=PERIODS_PER_YEAR,
    )
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    return {
        "strategy": label,
        "start": equity.index[0].strftime("%Y-%m-%d"),
        "end": equity.index[-1].strftime("%Y-%m-%d"),
        "n_days": len(equity),
        "start_value": float(equity.iloc[0]),
        "end_value": float(equity.iloc[-1]),
        "total_return": total_return,
        "cagr": m.cagr,
        "sharpe_252": m.sharpe,
        "max_dd": m.max_drawdown,
        "calmar": m.calmar,
    }


# ============================================================
# Trailing 7-year evaluation
# ============================================================


def run_trailing_7y(prices_dict: dict[str, pd.DataFrame]) -> list[dict]:
    """Run TSMOM at each lookback + combined, plus benchmarks."""
    start_ts, end_ts = _anchor_to_equity_calendar(prices_dict, WINDOW_START, WINDOW_END)
    print(f"Trailing 7-year window: {start_ts.date()} -> {end_ts.date()}")

    results: list[dict] = []

    # TSMOM at each lookback
    for lb in LOOKBACKS:
        print(f"  Running TSMOM lookback={lb}d ...")
        model = TimeSeriesMomentum(
            lookback_days=lb, vol_window=63, risk_parity=True,
            rebalance_freq=21, commission_bps=5.0, slippage_bps=5.0,
        )
        eq = model.backtest(prices_dict, start_ts, end_ts, initial_cash=INITIAL)
        label = f"TSMOM-{lb}d"
        results.append(_strategy_stats(eq, label))

    # Combined: average of 3 lookback equity curves (the AQR approach)
    print("  Running TSMOM Combined (avg of 21/63/252) ...")
    equities: list[pd.Series] = []
    for lb in LOOKBACKS:
        model = TimeSeriesMomentum(
            lookback_days=lb, vol_window=63, risk_parity=True,
            rebalance_freq=21, commission_bps=5.0, slippage_bps=5.0,
        )
        eq = model.backtest(prices_dict, start_ts, end_ts, initial_cash=INITIAL)
        equities.append(eq)

    # Blend: equal-weight the 3 equity curves via returns
    eq_df = pd.concat(equities, axis=1).dropna(how="any")
    if not eq_df.empty and len(eq_df) > 1:
        eq_returns = eq_df.pct_change().fillna(0.0)
        combined_returns = eq_returns.mean(axis=1)
        combined_eq = (1.0 + combined_returns).cumprod() * INITIAL
        combined_eq.iloc[0] = INITIAL
        results.append(_strategy_stats(combined_eq, "TSMOM-Combined"))
    else:
        results.append({"strategy": "TSMOM-Combined", "sharpe_252": 0.0, "cagr": 0.0})

    # Benchmarks
    print("  Running benchmarks ...")
    eq_bh = _equal_weight_bh(prices_dict, start_ts, end_ts, INITIAL)
    results.append(_strategy_stats(eq_bh, "EqWt B&H (multi-asset)"))

    sp_eq = _sp_bh(prices_dict, start_ts, end_ts, INITIAL)
    results.append(_strategy_stats(sp_eq, "SP500 B&H"))

    return results


# ============================================================
# Multi-seed windowed evaluation
# ============================================================


def run_multi_seed_eval(prices_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """5-seed x 12-window non-overlapping evaluation."""
    # Build a common date index
    closes: dict[str, pd.Series] = {}
    for sym, df in prices_dict.items():
        closes[sym] = df["close"].dropna()
    common_df = pd.DataFrame(closes).dropna(how="any")
    n_bars = len(common_df)

    # Minimum warmup: need 252 + 63 + 5 bars before first signal
    warmup = 252 + 63 + 5
    min_start = warmup
    max_start = n_bars - SIX_MONTHS - 1

    print(f"\nMulti-seed eval: {len(SEEDS)} seeds x {N_WINDOWS} windows")
    print(f"  Common bars: {n_bars}, min_start={min_start}, max_start={max_start}")

    all_rows: list[dict] = []

    for seed in SEEDS:
        starts = draw_non_overlapping_starts(
            seed=seed, min_start=min_start, max_start=max_start,
            window_len=SIX_MONTHS, n_windows=N_WINDOWS,
        )
        for w_idx, s in enumerate(starts):
            e = min(s + SIX_MONTHS, n_bars)
            start_ts = common_df.index[s]
            end_ts = common_df.index[e - 1]

            # Build price slices for the window (include all history up to end for warmup)
            window_prices = {
                sym: df.loc[df.index <= end_ts]
                for sym, df in prices_dict.items()
            }

            for lb in LOOKBACKS + [0]:  # 0 = combined
                if lb == 0:
                    # Combined: run all 3 and average
                    eqs = []
                    for sub_lb in LOOKBACKS:
                        m = TimeSeriesMomentum(
                            lookback_days=sub_lb, vol_window=63, risk_parity=True,
                            rebalance_freq=21, commission_bps=5.0, slippage_bps=5.0,
                        )
                        eq = m.backtest(window_prices, start_ts, end_ts, initial_cash=INITIAL)
                        eqs.append(eq)
                    eq_df = pd.concat(eqs, axis=1).dropna(how="any")
                    if eq_df.empty or len(eq_df) < 2:
                        continue
                    eq_ret = eq_df.pct_change().fillna(0.0)
                    comb_ret = eq_ret.mean(axis=1)
                    eq = (1.0 + comb_ret).cumprod() * INITIAL
                    eq.iloc[0] = INITIAL
                    label = "TSMOM-Combined"
                else:
                    m = TimeSeriesMomentum(
                        lookback_days=lb, vol_window=63, risk_parity=True,
                        rebalance_freq=21, commission_bps=5.0, slippage_bps=5.0,
                    )
                    eq = m.backtest(window_prices, start_ts, end_ts, initial_cash=INITIAL)
                    label = f"TSMOM-{lb}d"

                if eq.empty or len(eq) < 10:
                    continue
                metrics = compute_metrics(
                    eq, [], risk_free_rate=RISK_FREE, periods_per_year=PERIODS_PER_YEAR,
                )
                # Also compute benchmark for this window
                bh_eq = _equal_weight_bh(window_prices, start_ts, end_ts, INITIAL)
                bh_m = compute_metrics(
                    bh_eq, [], risk_free_rate=RISK_FREE, periods_per_year=PERIODS_PER_YEAR,
                ) if not bh_eq.empty else None

                all_rows.append({
                    "seed": seed,
                    "window": w_idx,
                    "start": start_ts.strftime("%Y-%m-%d"),
                    "end": end_ts.strftime("%Y-%m-%d"),
                    "strategy": label,
                    "sharpe": metrics.sharpe,
                    "cagr": metrics.cagr,
                    "max_dd": metrics.max_drawdown,
                    "bh_sharpe": bh_m.sharpe if bh_m else 0.0,
                    "bh_cagr": bh_m.cagr if bh_m else 0.0,
                })

    return pd.DataFrame(all_rows)


# ============================================================
# Main
# ============================================================


def main() -> None:
    t0 = time.time()
    store = DataStore(SETTINGS.data.dir)
    prices_dict = _load_universe(store)
    print(f"Loaded {len(prices_dict)} assets: {sorted(prices_dict.keys())}")
    for sym, df in sorted(prices_dict.items()):
        print(f"  {sym}: {len(df)} bars ({df.index[0].date()} -> {df.index[-1].date()})")

    # --- Trailing 7-year ---
    print("\n" + "=" * 90)
    print("TRAILING 7-YEAR EVALUATION")
    print("=" * 90)

    results_7y = run_trailing_7y(prices_dict)

    # Print summary table
    summary_df = pd.DataFrame(results_7y).sort_values("sharpe_252", ascending=False)
    print("\n" + "=" * 90)
    print(
        f"  {'strategy':<30}"
        f"{'end $':>12}{'total ret':>12}"
        f"{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'Calmar':>9}"
    )
    print("  " + "-" * 88)
    for _, r in summary_df.iterrows():
        ev = r.get("end_value", 0.0)
        tr = r.get("total_return", 0.0)
        print(
            f"  {r['strategy']:<30}"
            f"${ev:>10,.0f}"
            f"{tr:>+11.1%}"
            f"{r.get('cagr', 0.0):>+8.1%}"
            f"{r.get('sharpe_252', 0.0):>+9.3f}"
            f"{r.get('max_dd', 0.0):>+8.1%}"
            f"{r.get('calmar', 0.0):>+8.2f}"
        )

    # --- Multi-seed windowed ---
    print("\n" + "=" * 90)
    print("MULTI-SEED WINDOWED EVALUATION (5 seeds x 12 non-overlapping 6-month windows)")
    print("=" * 90)

    ms_df = run_multi_seed_eval(prices_dict)
    if not ms_df.empty:
        # Per-strategy summary across all seeds and windows
        agg = ms_df.groupby("strategy").agg(
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            std_sharpe=("sharpe", "std"),
            mean_cagr=("cagr", "mean"),
            mean_mdd=("max_dd", "mean"),
            mean_bh_sharpe=("bh_sharpe", "mean"),
            n_windows=("sharpe", "count"),
            pct_positive=("sharpe", lambda x: (x > 0).mean()),
        ).sort_values("median_sharpe", ascending=False)

        print(f"\n  {'strategy':<20}"
              f"{'mean SR':>10}{'med SR':>10}{'std SR':>10}"
              f"{'mean CAGR':>11}{'mean MDD':>11}"
              f"{'BH SR':>10}{'n':>5}{'% > 0':>8}")
        print("  " + "-" * 94)
        for strat, r in agg.iterrows():
            print(
                f"  {strat:<20}"
                f"{r['mean_sharpe']:>+9.3f}"
                f"{r['median_sharpe']:>+9.3f}"
                f"{r['std_sharpe']:>9.3f}"
                f"{r['mean_cagr']:>+10.1%}"
                f"{r['mean_mdd']:>+10.1%}"
                f"{r['mean_bh_sharpe']:>+9.3f}"
                f"{int(r['n_windows']):>5d}"
                f"{r['pct_positive']:>7.0%}"
            )

        # Per-seed summary for TSMOM-Combined
        combined = ms_df[ms_df["strategy"] == "TSMOM-Combined"]
        if not combined.empty:
            seed_agg = combined.groupby("seed").agg(
                median_sharpe=("sharpe", "median"),
                mean_sharpe=("sharpe", "mean"),
            )
            print("\n  TSMOM-Combined per-seed median Sharpe:")
            for seed, r in seed_agg.iterrows():
                print(f"    seed={seed}: median={r['median_sharpe']:+.3f}, mean={r['mean_sharpe']:+.3f}")

    # --- Save outputs ---
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet
    if not ms_df.empty:
        out_pq = out_dir / "tsmom_multi_asset.parquet"
        ms_df.to_parquet(out_pq)
        print(f"\n[wrote] {out_pq}")

    # Markdown report
    out_md = Path(__file__).parent / "TSMOM_MULTI_ASSET_RESULTS.md"
    md_lines = _build_markdown(summary_df, ms_df)
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[wrote] {out_md}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")


def _build_markdown(summary_df: pd.DataFrame, ms_df: pd.DataFrame) -> list[str]:
    lines = [
        "# Multi-Asset Time-Series Momentum (TSMOM) Results",
        "",
        "Moskowitz-Ooi-Pedersen (2012) trend-following signal across a diversified",
        "macro asset-class basket. Long-only: go long if trailing return > 0,",
        "flat otherwise. Risk-parity weighting (inverse realized vol).",
        "5 bps commission + 5 bps slippage per rebalance.",
        "",
        f"Universe: {', '.join(UNIVERSE)}",
        f"Risk-free rate: {RISK_FREE:.2%} (historical 2018-2024 average)",
        f"Annualization: {PERIODS_PER_YEAR:.0f} days/year",
        "",
        "## Trailing 7-year evaluation (2019-04-01 -> 2026-04-01)",
        "",
        "| Strategy | End Value | Total Return | CAGR | Sharpe | MDD | Calmar |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary_df.iterrows():
        ev = r.get("end_value", 0.0)
        tr = r.get("total_return", 0.0)
        lines.append(
            f"| `{r['strategy']}` | "
            f"${ev:,.0f} | "
            f"{tr:+.1%} | "
            f"{r.get('cagr', 0.0):+.1%} | "
            f"{r.get('sharpe_252', 0.0):+.3f} | "
            f"{r.get('max_dd', 0.0):+.1%} | "
            f"{r.get('calmar', 0.0):+.2f} |"
        )

    if not ms_df.empty:
        lines += [
            "",
            "## Multi-seed windowed evaluation (5 seeds x 12 non-overlapping 6-month windows)",
            "",
        ]
        agg = ms_df.groupby("strategy").agg(
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            std_sharpe=("sharpe", "std"),
            mean_cagr=("cagr", "mean"),
            mean_mdd=("max_dd", "mean"),
            mean_bh_sharpe=("bh_sharpe", "mean"),
            n_windows=("sharpe", "count"),
            pct_positive=("sharpe", lambda x: (x > 0).mean()),
        ).sort_values("median_sharpe", ascending=False)

        lines.append(
            "| Strategy | Mean Sharpe | Median Sharpe | Std Sharpe | "
            "Mean CAGR | Mean MDD | BH Sharpe | N | % > 0 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for strat, r in agg.iterrows():
            lines.append(
                f"| `{strat}` | "
                f"{r['mean_sharpe']:+.3f} | "
                f"{r['median_sharpe']:+.3f} | "
                f"{r['std_sharpe']:.3f} | "
                f"{r['mean_cagr']:+.1%} | "
                f"{r['mean_mdd']:+.1%} | "
                f"{r['mean_bh_sharpe']:+.3f} | "
                f"{int(r['n_windows'])} | "
                f"{r['pct_positive']:.0%} |"
            )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/tsmom_multi_asset_eval.py",
        "```",
        "",
    ]
    return lines


if __name__ == "__main__":
    main()
