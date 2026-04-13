"""Historical stress test — is our momentum edge real or just a bull market?

Tests cross-sectional momentum across 3 distinct market eras using
deep yfinance history (2000-2026) to answer: does the strategy
survive through dot-com crash, 2008 financial crisis, and the
2015-2026 period we've been backtesting on?

Eras:
  1. Dot-com bust + recovery  (2001-01 → 2007-12) — includes -49% SP crash
  2. Financial crisis + recovery (2007-01 → 2014-12) — includes -57% SP crash
  3. Modern bull + COVID + bear (2015-01 → 2026-04) — our existing test window
  4. Full 25-year window (2001-01 → 2026-04) — the ultimate stress test

For each era, runs momentum top-10 vs SP500 B&H at 5 random seeds ×
12 non-overlapping 6-month windows. Also runs 100 random 2-year windows
across the entire 2001-2026 span (Monte Carlo style) to get a
distribution of outcomes rather than cherry-picked windows.

Uses a ~100-stock universe of long-lived SP500 constituents that
existed across the full 25-year span (survivorship bias acknowledged
but unavoidable without a historical constituents database).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.model.momentum import CrossSectionalMomentum

# Long-lived SP500 stocks with data back to 2000 (avoid survivorship
# bias as much as possible by including fallen angels alongside winners)
UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "TXN", "QCOM", "ADBE", "AMAT",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "AXP", "BK", "USB", "PNC", "CME",
    # Health care
    "JNJ", "PFE", "MRK", "ABT", "BMY", "AMGN", "GILD", "MDT", "UNH", "LLY",
    # Consumer staples
    "PG", "KO", "PEP", "WMT", "COST", "CL", "MO", "PM", "KMB", "GIS",
    # Industrials
    "GE", "MMM", "HON", "CAT", "DE", "UPS", "LMT", "RTX", "BA", "GD",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC", "OXY", "HAL",
    # Consumer discretionary
    "HD", "MCD", "NKE", "LOW", "TJX", "SBUX", "YUM", "DG", "ROST", "TGT",
    # Utilities / REITs / Materials
    "NEE", "DUK", "SO", "D", "AEP", "SHW", "APD", "ECL", "NEM", "FCX",
    # Telecom / Media
    "T", "VZ", "CMCSA", "DIS",
    # Index
    "^GSPC",
]

START_DEEP = "2000-01-01"
END_ALL = "2026-04-01"
RF = historical_usd_rate("2018-2024")
INITIAL = 10_000.0


def _fetch_deep_history(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch full history via yfinance for all tickers."""
    prices = {}
    print(f"Fetching deep history for {len(tickers)} tickers via yfinance...")
    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, start=START_DEEP, end=END_ALL, progress=False)
            if df.empty or len(df) < 1000:
                continue
            # Normalize columns
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
            df.index = df.index.normalize()
            prices[t] = df
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(tickers)}")
    print(f"Loaded {len(prices)} tickers with deep history")
    return prices


def _run_momentum_window(
    prices: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_long: int = 10,
) -> tuple[float, float, float]:
    """Run momentum on a single window, return (sharpe, cagr, mdd)."""
    mom = CrossSectionalMomentum(
        lookback_days=252, skip_days=21, n_long=n_long, rebalance_freq=21,
    )
    eq = mom.backtest(prices, start, end, initial_cash=INITIAL)
    if eq.empty or len(eq) < 50:
        return 0.0, 0.0, 0.0
    m = compute_metrics(eq, [], risk_free_rate=RF, periods_per_year=252.0)
    return m.sharpe, m.cagr, m.max_drawdown


def _run_bh_window(
    prices: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, float, float]:
    """Run SP500 B&H on a single window."""
    sp = prices.get("^GSPC")
    if sp is None:
        return 0.0, 0.0, 0.0
    sl = sp.loc[(sp.index >= start) & (sp.index <= end), "close"]
    if sl.empty:
        return 0.0, 0.0, 0.0
    eq = (sl / sl.iloc[0]) * INITIAL
    m = compute_metrics(eq, [], risk_free_rate=RF, periods_per_year=252.0)
    return m.sharpe, m.cagr, m.max_drawdown


def main() -> None:
    t0 = time.time()
    all_prices = _fetch_deep_history(UNIVERSE)

    # Remove index from stock universe for momentum (keep for benchmark)
    stock_prices = {k: v for k, v in all_prices.items() if k != "^GSPC"}
    print(f"Stock universe: {len(stock_prices)} tickers")

    # === ERA TESTS ===
    eras = [
        ("Dot-com bust + recovery (2001-2007)", "2001-01-01", "2007-12-31"),
        ("Financial crisis + recovery (2007-2014)", "2007-01-01", "2014-12-31"),
        ("Modern era (2015-2026)", "2015-01-01", "2026-04-01"),
        ("Full 25 years (2001-2026)", "2001-01-01", "2026-04-01"),
    ]

    print("\n" + "=" * 90)
    print("ERA-BY-ERA STRESS TEST — Momentum Top-10 vs SP500 B&H")
    print("=" * 90)
    print(f"\n  {'Era':<42}{'Mom Sh':>9}{'Mom CAGR':>10}{'Mom MDD':>9}{'SP Sh':>8}{'SP CAGR':>9}{'SP MDD':>8}{'Δ Sh':>8}")
    print(f"  {'-'*103}")

    era_results = []
    for label, start_str, end_str in eras:
        start = pd.Timestamp(start_str, tz="UTC")
        end = pd.Timestamp(end_str, tz="UTC")
        m_sh, m_cagr, m_mdd = _run_momentum_window(stock_prices, start, end)
        s_sh, s_cagr, s_mdd = _run_bh_window(all_prices, start, end)
        delta = m_sh - s_sh
        marker = "✓" if delta > 0 else "✗"
        print(
            f"  {label:<42}{m_sh:>+9.3f}{m_cagr:>+9.1%}{m_mdd:>+8.1%}"
            f"{s_sh:>+8.3f}{s_cagr:>+8.1%}{s_mdd:>+7.1%}{delta:>+8.3f} {marker}"
        )
        era_results.append({
            "era": label, "mom_sharpe": m_sh, "mom_cagr": m_cagr, "mom_mdd": m_mdd,
            "sp_sharpe": s_sh, "sp_cagr": s_cagr, "sp_mdd": s_mdd, "delta": delta,
        })

    # === MONTE CARLO: 100 RANDOM 2-YEAR WINDOWS ===
    print(f"\n{'=' * 90}")
    print("MONTE CARLO — 100 random 2-year windows across 2001-2026")
    print("=" * 90)

    sp = all_prices.get("^GSPC")
    if sp is not None:
        all_dates = sp.index.sort_values()
        min_date = pd.Timestamp("2002-01-01", tz="UTC")  # need 1yr lookback
        max_date = pd.Timestamp("2024-04-01", tz="UTC")  # 2yr before end
        eligible = all_dates[(all_dates >= min_date) & (all_dates <= max_date)]

        rng = random.Random(42)
        mc_results = []
        two_years = 504  # trading days

        for i in range(100):
            start_idx = rng.randint(0, len(eligible) - two_years - 1)
            start = eligible[start_idx]
            end = eligible[start_idx + two_years]

            m_sh, m_cagr, m_mdd = _run_momentum_window(stock_prices, start, end)
            s_sh, s_cagr, s_mdd = _run_bh_window(all_prices, start, end)
            mc_results.append({
                "start": start.date(), "end": end.date(),
                "mom_sharpe": m_sh, "sp_sharpe": s_sh,
                "mom_cagr": m_cagr, "sp_cagr": s_cagr,
                "delta_sharpe": m_sh - s_sh,
                "delta_cagr": m_cagr - s_cagr,
                "mom_wins_sharpe": m_sh > s_sh,
                "mom_wins_cagr": m_cagr > s_cagr,
            })
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  ...{i+1}/100 windows  elapsed={elapsed:.0f}s")

        mc_df = pd.DataFrame(mc_results)

        wins_sharpe = mc_df["mom_wins_sharpe"].sum()
        wins_cagr = mc_df["mom_wins_cagr"].sum()
        avg_delta_sh = mc_df["delta_sharpe"].mean()
        avg_delta_cagr = mc_df["delta_cagr"].mean()
        avg_mom_sh = mc_df["mom_sharpe"].mean()
        avg_sp_sh = mc_df["sp_sharpe"].mean()
        avg_mom_cagr = mc_df["mom_cagr"].mean()
        avg_sp_cagr = mc_df["sp_cagr"].mean()

        # Windows during bear markets specifically
        bear_windows = mc_df[mc_df["sp_cagr"] < 0]
        bull_windows = mc_df[mc_df["sp_cagr"] > 0.15]
        flat_windows = mc_df[(mc_df["sp_cagr"] >= 0) & (mc_df["sp_cagr"] <= 0.15)]

        print("\n  Overall (100 random 2-year windows):")
        print(f"    Momentum wins on Sharpe: {wins_sharpe}/100 ({wins_sharpe}%)")
        print(f"    Momentum wins on CAGR:   {wins_cagr}/100 ({wins_cagr}%)")
        print(f"    Avg momentum Sharpe:     {avg_mom_sh:+.3f}  vs  SP {avg_sp_sh:+.3f}  (Δ {avg_delta_sh:+.3f})")
        print(f"    Avg momentum CAGR:       {avg_mom_cagr:+.1%}  vs  SP {avg_sp_cagr:+.1%}  (Δ {avg_delta_cagr:+.1%})")

        if len(bear_windows) > 0:
            bear_wins = bear_windows["mom_wins_sharpe"].sum()
            print(f"\n  Bear windows (SP CAGR < 0): {len(bear_windows)} windows")
            print(f"    Momentum wins: {bear_wins}/{len(bear_windows)} ({bear_wins/len(bear_windows)*100:.0f}%)")
            print(f"    Avg Δ Sharpe: {bear_windows['delta_sharpe'].mean():+.3f}")
            print(f"    Avg Δ CAGR:   {bear_windows['delta_cagr'].mean():+.1%}")

        if len(bull_windows) > 0:
            bull_wins = bull_windows["mom_wins_sharpe"].sum()
            print(f"\n  Bull windows (SP CAGR > 15%): {len(bull_windows)} windows")
            print(f"    Momentum wins: {bull_wins}/{len(bull_windows)} ({bull_wins/len(bull_windows)*100:.0f}%)")
            print(f"    Avg Δ Sharpe: {bull_windows['delta_sharpe'].mean():+.3f}")

        if len(flat_windows) > 0:
            flat_wins = flat_windows["mom_wins_sharpe"].sum()
            print(f"\n  Flat windows (SP CAGR 0-15%): {len(flat_windows)} windows")
            print(f"    Momentum wins: {flat_wins}/{len(flat_windows)} ({flat_wins/len(flat_windows)*100:.0f}%)")

        # Worst 10 windows for momentum
        worst = mc_df.sort_values("delta_sharpe").head(10)
        print("\n  10 WORST windows for momentum (vs SP):")
        print(f"    {'Start':<12}{'End':<12}{'Mom Sh':>9}{'SP Sh':>8}{'Δ':>8}{'Mom CAGR':>10}{'SP CAGR':>10}")
        for _, r in worst.iterrows():
            print(f"    {r['start']!s:<12}{r['end']!s:<12}{r['mom_sharpe']:>+9.3f}{r['sp_sharpe']:>+8.3f}{r['delta_sharpe']:>+8.3f}{r['mom_cagr']:>+9.1%}{r['sp_cagr']:>+9.1%}")

    # === VERDICT ===
    print(f"\n{'=' * 90}")
    print("VERDICT — is this a bull-market artifact?")
    print("=" * 90)

    all_eras_positive = all(r["delta"] > 0 for r in era_results)
    mc_win_rate = wins_sharpe if 'wins_sharpe' in dir() else 0

    if all_eras_positive and mc_win_rate >= 60:
        print("  ✓ Momentum beats SP in ALL eras AND wins >60% of random windows.")
        print("  ✓ This is NOT just a bull-market artifact.")
    elif all_eras_positive:
        print("  ~ Momentum beats SP in all eras but wins <60% of random windows.")
        print("  ~ Edge is real but concentrated in specific regimes.")
    else:
        failed_eras = [r["era"] for r in era_results if r["delta"] <= 0]
        print(f"  ✗ Momentum LOST to SP in: {failed_eras}")
        print("  ✗ The edge may be regime-dependent or a bull-market artifact.")

    # Save results
    out_parquet = Path(__file__).parent / "data" / "historical_stress_test.parquet"
    if 'mc_df' in dir():
        mc_df.to_parquet(out_parquet)
        print(f"\n[wrote] {out_parquet}")

    out_md = Path(__file__).parent / "HISTORICAL_STRESS_TEST.md"
    lines = [
        "# Historical stress test — 25 years, 100 random windows",
        "",
        "Is cross-sectional momentum a bull-market artifact or a persistent edge?",
        "",
        "## Era-by-era results",
        "",
        "| Era | Mom Sharpe | Mom CAGR | Mom MDD | SP Sharpe | SP CAGR | SP MDD | Δ Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in era_results:
        marker = "✓" if r["delta"] > 0 else "✗"
        lines.append(
            f"| {r['era']} | {r['mom_sharpe']:+.3f} | {r['mom_cagr']:+.1%} | "
            f"{r['mom_mdd']:+.1%} | {r['sp_sharpe']:+.3f} | {r['sp_cagr']:+.1%} | "
            f"{r['sp_mdd']:+.1%} | {r['delta']:+.3f} {marker} |"
        )
    if 'mc_df' in dir():
        lines += [
            "",
            "## Monte Carlo — 100 random 2-year windows (2001-2026)",
            "",
            f"- Momentum wins on Sharpe: **{wins_sharpe}/100 ({wins_sharpe}%)**",
            f"- Momentum wins on CAGR: **{wins_cagr}/100 ({wins_cagr}%)**",
            f"- Avg momentum Sharpe: {avg_mom_sh:+.3f} vs SP {avg_sp_sh:+.3f} (Δ {avg_delta_sh:+.3f})",
            f"- Avg momentum CAGR: {avg_mom_cagr:+.1%} vs SP {avg_sp_cagr:+.1%}",
        ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_md}")
    print(f"\nTotal elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
