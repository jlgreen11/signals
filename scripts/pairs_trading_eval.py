"""Pairs trading evaluation — 20 major US equities, trailing 7-year window.

Tests a market-neutral statistical arbitrage strategy (pairs trading) on
the same 20-stock universe that directional models failed on. The structural
difference: pairs trading profits from relative mean-reversion between
cointegrated stocks, not from predicting absolute direction.

Universe: AAPL MSFT NVDA AMZN GOOGL META TSLA BRK-B UNH JNJ JPM V PG XOM
          LLY AVGO COST NFLX AMD ADBE

Window: 2019-04-01 -> 2026-04-01 (trailing 7 years, 252/yr annualization)
Benchmarks: SP500 B&H, equal-weight 20-stock B&H
Multi-seed: 5 seeds with _window_sampler non-overlapping 6-month windows

Output:
  scripts/data/pairs_trading.parquet    -- daily equity curves
  scripts/PAIRS_TRADING_RESULTS.md      -- summary comparison
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore
from signals.model.pairs import PairsTrading

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

SEEDS = [42, 7, 100, 999, 1337]
SIX_MONTHS = 126


def _load_prices(store: DataStore) -> dict[str, pd.Series]:
    """Load close prices for all tickers, aligned to common trading calendar."""
    prices = {}
    for ticker in TICKERS:
        df = store.load(ticker, "1d").sort_index()
        if df.empty:
            print(f"  [WARN] No data for {ticker}, skipping")
            continue
        prices[ticker] = df["close"]
    return prices


def _bh_equity(
    prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Buy-and-hold equity curve, rebased to $INITIAL."""
    sl = prices.loc[(prices.index >= start) & (prices.index <= end)]
    if sl.empty or sl.iloc[0] <= 0:
        return pd.Series(dtype=float)
    return (sl / sl.iloc[0]) * INITIAL


def _equal_weight_bh(
    prices_dict: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Equal-weight buy-and-hold of all tickers."""
    legs = {}
    for ticker, p in prices_dict.items():
        sl = p.loc[(p.index >= start) & (p.index <= end)].dropna()
        if sl.empty or sl.iloc[0] <= 0:
            continue
        legs[ticker] = sl / sl.iloc[0]

    if not legs:
        return pd.Series(dtype=float)

    leg_df = pd.DataFrame(legs).dropna(how="any")
    if leg_df.empty:
        return pd.Series(dtype=float)

    leg_returns = leg_df.pct_change().fillna(0.0)
    n = len(legs)
    port_returns = (1.0 / n) * leg_returns.sum(axis=1)
    port_equity = (1.0 + port_returns).cumprod() * INITIAL
    port_equity.iloc[0] = INITIAL
    return port_equity


def _strategy_stats(equity: pd.Series, label: str) -> dict:
    """Compute standard stats for an equity curve."""
    if equity.empty:
        return {"strategy": label}
    m = compute_metrics(
        equity, [],
        risk_free_rate=RF,
        periods_per_year=PPY,
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
        "sharpe": m.sharpe,
        "max_dd": m.max_drawdown,
        "calmar": m.calmar,
    }


def _correlation(s1: pd.Series, s2: pd.Series) -> float:
    """Correlation of daily returns between two equity curves."""
    r1 = s1.pct_change().dropna()
    r2 = s2.pct_change().dropna()
    common = r1.index.intersection(r2.index)
    if len(common) < 20:
        return float("nan")
    return float(r1.loc[common].corr(r2.loc[common]))


def run_full_backtest(prices_dict: dict[str, pd.Series]) -> None:
    """Run the main trailing 7-year pairs trading backtest."""
    print("=" * 100)
    print("PAIRS TRADING EVALUATION — trailing 7 years")
    print("=" * 100)
    print(f"Window: {WINDOW_START.date()} -> {WINDOW_END.date()}")
    print(f"Tickers: {len(prices_dict)} loaded of {len(TICKERS)} requested")
    print()

    t0 = time.time()

    # Run pairs trading strategy
    pt = PairsTrading(
        coint_pvalue=0.05,
        entry_zscore=2.0,
        exit_zscore=0.5,
        lookback=252,
        max_pairs=5,
        zscore_window=60,
    )

    result = pt.backtest(
        prices_dict,
        start=WINDOW_START,
        end=WINDOW_END,
        initial_cash=INITIAL,
    )

    elapsed = time.time() - t0
    print(f"Backtest completed in {elapsed:.1f}s")

    # Pair discovery stats
    total_pairs_found = sum(d["n_pairs"] for d in result.pair_discovery_log)
    n_discoveries = len(result.pair_discovery_log)
    avg_pairs = total_pairs_found / n_discoveries if n_discoveries > 0 else 0
    print(f"Pair re-discoveries: {n_discoveries}")
    print(f"Average pairs found per re-discovery: {avg_pairs:.1f}")
    print(f"Total pairs found across all windows: {total_pairs_found}")

    # List discovered pairs per window
    for entry in result.pair_discovery_log:
        print(f"  {entry['date'].date()}: {entry['n_pairs']} pairs", end="")
        if entry["pairs"]:
            pair_strs = [f"{a}-{b} (p={p:.3f})" for a, b, p in entry["pairs"]]
            print(f"  [{', '.join(pair_strs)}]")
        else:
            print()

    # Trade stats
    closed_trades = [t for t in result.trades if t.closed]
    n_trades = len(closed_trades)
    if n_trades > 0:
        holding_days = []
        wins = 0
        for t in closed_trades:
            if t.exit_date is not None and t.entry_date is not None:
                days = (t.exit_date - t.entry_date).days
                holding_days.append(days)
            if t.pnl > 0:
                wins += 1
        avg_holding = np.mean(holding_days) if holding_days else 0
        win_rate = wins / n_trades
        print(f"\nTrades: {n_trades}")
        print(f"Win rate: {win_rate:.1%}")
        print(f"Avg holding period: {avg_holding:.1f} days")
    else:
        avg_holding = 0
        win_rate = 0
        print("\nNo trades executed")

    # Benchmarks
    store = DataStore(SETTINGS.data.dir)
    sp = store.load("^GSPC", "1d").sort_index()
    sp_eq = _bh_equity(sp["close"], WINDOW_START, WINDOW_END)
    ew_eq = _equal_weight_bh(prices_dict, WINDOW_START, WINDOW_END)

    # Pairs equity curve
    pairs_eq = result.equity_curve
    if pairs_eq.empty:
        print("\n[ERROR] Empty equity curve, cannot compare")
        return

    # Compute stats
    strategies = []
    strategies.append(_strategy_stats(pairs_eq, "Pairs Trading"))
    strategies.append(_strategy_stats(sp_eq, "SP500 B&H"))
    strategies.append(_strategy_stats(ew_eq, "EW 20-stock B&H"))

    # Correlation to SP500
    corr_sp = _correlation(pairs_eq, sp_eq)

    # Print comparison
    print("\n" + "=" * 100)
    print("COMPARISON")
    print("=" * 100)
    print(
        f"  {'strategy':<24}{'end $':>14}{'total ret':>13}"
        f"{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'Calmar':>9}"
    )
    print("  " + "-" * 85)
    summary_df = pd.DataFrame(strategies)
    for _, r in summary_df.iterrows():
        end_val = r.get("end_value", 0)
        total_ret = r.get("total_return", 0)
        cagr_val = r.get("cagr", 0)
        sharpe_val = r.get("sharpe", 0)
        mdd_val = r.get("max_dd", 0)
        calmar_val = r.get("calmar", 0)
        print(
            f"  {r['strategy']:<24}"
            f"${end_val:>12,.0f}"
            f"{total_ret:>+12.1%}"
            f"{cagr_val:>+8.1%}"
            f"{sharpe_val:>+9.3f}"
            f"{mdd_val:>+8.1%}"
            f"{calmar_val:>+8.2f}"
        )

    print(f"\nCorrelation of pairs trading daily returns to SP500: {corr_sp:+.3f}")
    if abs(corr_sp) < 0.3:
        print("  => LOW correlation: strategy is approximately market-neutral")
    elif abs(corr_sp) < 0.6:
        print("  => MODERATE correlation: partial market-neutrality")
    else:
        print("  => HIGH correlation: strategy is NOT market-neutral")

    # Save equity curves
    out_parquet = Path(__file__).parent / "data" / "pairs_trading.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    daily_df = pd.DataFrame({
        "pairs_trading": pairs_eq,
        "sp500_bh": sp_eq,
        "ew_20stock_bh": ew_eq,
    }).dropna(how="all")
    daily_df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}")

    return strategies, result, corr_sp, n_trades, avg_holding, win_rate, avg_pairs


def run_multi_seed_eval(prices_dict: dict[str, pd.Series]) -> list[dict]:
    """5-seed multi-seed evaluation with non-overlapping windows."""
    print("\n" + "=" * 100)
    print("MULTI-SEED EVALUATION (5 seeds x non-overlapping 6-month windows)")
    print("=" * 100)

    # Build a combined price DataFrame for index alignment
    df = pd.DataFrame(prices_dict).sort_index()
    df = df.loc[(df.index >= WINDOW_START) & (df.index <= WINDOW_END)]
    df = df.dropna(axis=1, how="any")

    if len(df) < SIX_MONTHS * 2:
        print("[WARN] Not enough data for multi-seed windows")
        return []

    # Need lookback warmup before first window
    lookback = 252
    min_start = lookback + 60 + 5  # lookback + zscore_window + pad
    max_start = len(df) - SIX_MONTHS - 1

    if max_start <= min_start:
        print("[WARN] Not enough range for non-overlapping windows")
        return []

    store = DataStore(SETTINGS.data.dir)
    sp = store.load("^GSPC", "1d").sort_index()

    all_rows = []
    for seed in SEEDS:
        starts = draw_non_overlapping_starts(
            seed=seed,
            min_start=min_start,
            max_start=max_start,
            window_len=SIX_MONTHS,
            n_windows=8,
        )

        for w, start_i in enumerate(starts):
            end_i = start_i + SIX_MONTHS
            w_start = df.index[start_i]
            w_end = df.index[min(end_i, len(df) - 1)]

            # Pairs trading on this window
            pt = PairsTrading(
                lookback=lookback,
                max_pairs=5,
                zscore_window=60,
            )

            # Use prices from the full range (for lookback warmup)
            window_prices = {
                t: prices_dict[t].loc[prices_dict[t].index <= w_end]
                for t in df.columns
                if t in prices_dict
            }

            result = pt.backtest(
                window_prices, start=w_start, end=w_end, initial_cash=INITIAL,
            )

            if result.equity_curve.empty:
                pairs_sharpe, pairs_cagr, pairs_mdd = 0.0, 0.0, 0.0
            else:
                m = compute_metrics(
                    result.equity_curve, [],
                    risk_free_rate=RF, periods_per_year=PPY,
                )
                pairs_sharpe = m.sharpe
                pairs_cagr = m.cagr
                pairs_mdd = m.max_drawdown

            # SP500 B&H on same window
            sp_sl = sp.loc[(sp.index >= w_start) & (sp.index <= w_end)]
            if sp_sl.empty or sp_sl["close"].iloc[0] <= 0:
                sp_sharpe, sp_cagr = 0.0, 0.0
            else:
                sp_eq = (sp_sl["close"] / sp_sl["close"].iloc[0]) * INITIAL
                sp_m = compute_metrics(sp_eq, [], risk_free_rate=RF, periods_per_year=PPY)
                sp_sharpe = sp_m.sharpe
                sp_cagr = sp_m.cagr

            all_rows.append({
                "seed": seed,
                "window_idx": w,
                "start": w_start,
                "end": w_end,
                "pairs_sharpe": pairs_sharpe,
                "pairs_cagr": pairs_cagr,
                "pairs_mdd": pairs_mdd,
                "sp_sharpe": sp_sharpe,
                "sp_cagr": sp_cagr,
                "sharpe_delta": pairs_sharpe - sp_sharpe,
                "cagr_delta": pairs_cagr - sp_cagr,
            })

    if not all_rows:
        return []

    results_df = pd.DataFrame(all_rows)

    # Per-seed aggregation
    per_seed = (
        results_df.groupby("seed")
        .agg(
            pairs_sharpe=("pairs_sharpe", "median"),
            pairs_cagr=("pairs_cagr", "median"),
            sp_sharpe=("sp_sharpe", "median"),
            sp_cagr=("sp_cagr", "median"),
            sharpe_delta=("sharpe_delta", "median"),
            cagr_delta=("cagr_delta", "median"),
        )
        .reset_index()
    )

    print(
        f"\n  {'seed':<8}{'pairs Sh':>10}{'SP Sh':>10}{'Delta Sh':>10}"
        f"{'pairs CAGR':>12}{'SP CAGR':>12}{'Delta CAGR':>12}"
    )
    print("  " + "-" * 72)
    for _, r in per_seed.iterrows():
        print(
            f"  {int(r['seed']):<8}"
            f"{r['pairs_sharpe']:>+10.3f}{r['sp_sharpe']:>+10.3f}"
            f"{r['sharpe_delta']:>+10.3f}"
            f"{r['pairs_cagr']:>+11.1%}{r['sp_cagr']:>+11.1%}"
            f"{r['cagr_delta']:>+11.1%}"
        )

    # Cross-seed average
    avg = per_seed[["pairs_sharpe", "sp_sharpe", "sharpe_delta",
                     "pairs_cagr", "sp_cagr", "cagr_delta"]].mean()
    print("  " + "-" * 72)
    print(
        f"  {'avg':<8}"
        f"{avg['pairs_sharpe']:>+10.3f}{avg['sp_sharpe']:>+10.3f}"
        f"{avg['sharpe_delta']:>+10.3f}"
        f"{avg['pairs_cagr']:>+11.1%}{avg['sp_cagr']:>+11.1%}"
        f"{avg['cagr_delta']:>+11.1%}"
    )

    wins = int((per_seed["sharpe_delta"] > 0).sum())
    print(f"\n  Seeds where pairs beats SP on Sharpe: {wins}/{len(SEEDS)}")

    return all_rows


def write_markdown(
    strategies: list[dict],
    corr_sp: float,
    n_trades: int,
    avg_holding: float,
    win_rate: float,
    avg_pairs: float,
    multi_seed_rows: list[dict],
) -> None:
    """Write results markdown file."""
    out_md = Path(__file__).parent / "PAIRS_TRADING_RESULTS.md"

    md = [
        "# Pairs Trading Evaluation",
        "",
        f"**Window**: {WINDOW_START.date()} -> {WINDOW_END.date()} (trailing 7 years)",
        f"**Universe**: {len(TICKERS)} tickers -- {', '.join(TICKERS)}",
        "**Parameters**: coint_pvalue=0.05, entry_z=2.0, exit_z=0.5, lookback=252, max_pairs=5",
        f"**Annualization**: {int(PPY)}/yr, rf = {RF:.2%}",
        "",
        "## Strategy Summary",
        "",
        f"- Average pairs found per re-discovery: **{avg_pairs:.1f}**",
        f"- Total round-trip trades: **{n_trades}**",
        f"- Win rate: **{win_rate:.1%}**",
        f"- Average holding period: **{avg_holding:.0f} days**",
        f"- Correlation to SP500: **{corr_sp:+.3f}**",
        "",
    ]

    if abs(corr_sp) < 0.3:
        md.append("The strategy exhibits **low correlation** to SP500, confirming "
                   "approximate market-neutrality.")
    elif abs(corr_sp) < 0.6:
        md.append("The strategy exhibits **moderate correlation** to SP500 -- partial "
                   "market-neutrality.")
    else:
        md.append("The strategy exhibits **high correlation** to SP500 -- NOT "
                   "market-neutral.")

    md += [
        "",
        "## Full 7-Year Comparison",
        "",
        "| strategy | end value | total return | CAGR | Sharpe | MDD | Calmar |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in strategies:
        end_val = s.get("end_value", 0)
        total_ret = s.get("total_return", 0)
        cagr_val = s.get("cagr", 0)
        sharpe_val = s.get("sharpe", 0)
        mdd_val = s.get("max_dd", 0)
        calmar_val = s.get("calmar", 0)
        md.append(
            f"| `{s['strategy']}` | ${end_val:,.0f} | {total_ret:+.1%} | "
            f"{cagr_val:+.1%} | {sharpe_val:+.3f} | {mdd_val:+.1%} | {calmar_val:+.2f} |"
        )

    if multi_seed_rows:
        ms_df = pd.DataFrame(multi_seed_rows)
        per_seed = ms_df.groupby("seed").agg(
            pairs_sharpe=("pairs_sharpe", "median"),
            sp_sharpe=("sp_sharpe", "median"),
            sharpe_delta=("sharpe_delta", "median"),
        ).reset_index()
        avg_delta = per_seed["sharpe_delta"].mean()
        wins = int((per_seed["sharpe_delta"] > 0).sum())

        md += [
            "",
            "## Multi-Seed Evaluation (5 seeds, non-overlapping 6-month windows)",
            "",
            f"- Avg Sharpe delta (pairs - SP): **{avg_delta:+.3f}**",
            f"- Seeds where pairs beats SP on Sharpe: **{wins}/{len(SEEDS)}**",
            "",
            "| seed | pairs Sharpe | SP Sharpe | delta |",
            "|---:|---:|---:|---:|",
        ]
        for _, r in per_seed.iterrows():
            md.append(
                f"| {int(r['seed'])} | {r['pairs_sharpe']:+.3f} | "
                f"{r['sp_sharpe']:+.3f} | {r['sharpe_delta']:+.3f} |"
            )

    md += [
        "",
        "## Key Insight",
        "",
        "Pairs trading is structurally different from the directional models "
        "(Markov chains, trend filters, hybrid vol-routers) that failed on "
        "individual stocks. It is market-neutral by construction: each position "
        "consists of a long leg and a short leg, so the portfolio does not rely "
        "on the market going up. The question is whether cointegration relationships "
        "among major US stocks are strong and stable enough to generate "
        "risk-adjusted returns after transaction costs.",
        "",
        "## Raw data",
        "",
        "- `scripts/data/pairs_trading.parquet` -- daily equity curves",
        "- Reproduce: `python scripts/pairs_trading_eval.py`",
        "",
    ]

    out_md.write_text("\n".join(md) + "\n")
    print(f"[wrote] {out_md}")


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices_dict = _load_prices(store)

    if len(prices_dict) < 5:
        print(f"[ERROR] Only {len(prices_dict)} tickers loaded, need at least 5")
        return

    result = run_full_backtest(prices_dict)
    if result is None:
        return
    strategies, bt_result, corr_sp, n_trades, avg_holding, win_rate, avg_pairs = result

    multi_seed_rows = run_multi_seed_eval(prices_dict)

    write_markdown(
        strategies, corr_sp, n_trades, avg_holding, win_rate, avg_pairs,
        multi_seed_rows,
    )


if __name__ == "__main__":
    main()
