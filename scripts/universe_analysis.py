"""Universe analysis — which stocks should we actually be trading?

The project's momentum strategy was tested on 20 hand-picked mega-caps.
This script answers: is 20 the right number? Which 20? What happens if
we use 50, 100, or the full SP500?

Approach:
1. Fetch the full SP500 constituent list (~503 tickers)
2. Download 3 years of daily data for all of them
3. Run the momentum strategy at different universe sizes:
   - Top 20 by market cap (what we have now)
   - Top 50
   - Top 100
   - Full SP500
4. For each universe, vary n_long (number of stocks held):
   - n_long = 5, 10, 20
5. Compare Sharpe, CAGR, turnover, and drawdown
6. Identify the optimal (universe_size, n_long) pair

Also: analyze which SECTORS the momentum strategy favors — does it
concentrate in tech, or does it rotate across sectors?
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore
from signals.model.momentum import CrossSectionalMomentum

START = pd.Timestamp("2022-04-01", tz="UTC")  # 4-year window
END = pd.Timestamp("2026-04-01", tz="UTC")
INITIAL = 10_000.0
RF = historical_usd_rate("2018-2024")
PPY = 252.0


def _fetch_sp500_list() -> pd.DataFrame:
    """Fetch SP500 constituents with sector info."""
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df


def _load_prices(store: DataStore, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load available price data; skip tickers that aren't in the store."""
    prices = {}
    for t in tickers:
        try:
            df = store.load(t, "1d").sort_index()
            if len(df) > 500:
                prices[t] = df
        except Exception:
            pass
    return prices


def _fetch_missing(store: DataStore, tickers: list[str], batch_size: int = 20) -> int:
    """Fetch data for tickers not yet in the store. Returns count fetched."""

    fetched = 0
    need_fetch = []
    for t in tickers:
        try:
            df = store.load(t, "1d")
            if len(df) < 500:
                need_fetch.append(t)
        except Exception:
            need_fetch.append(t)

    if not need_fetch:
        return 0

    print(f"  Fetching {len(need_fetch)} tickers not yet in store...")
    for i in range(0, len(need_fetch), batch_size):
        batch = need_fetch[i : i + batch_size]
        for t in batch:
            try:
                from signals.data.yahoo import YahooFinanceSource

                src = YahooFinanceSource()
                df = src.fetch(t, "1d", start="2020-01-01")
                if df is not None and not df.empty:
                    store.save(t, "1d", df)
                    fetched += 1
            except Exception:
                pass
        if i % 100 == 0 and i > 0:
            print(f"    ...{i}/{len(need_fetch)} fetched")
    return fetched


def _run_momentum(
    prices: dict[str, pd.DataFrame],
    n_long: int,
    label: str,
) -> dict:
    """Run momentum strategy and return stats dict."""
    mom = CrossSectionalMomentum(
        lookback_days=252,
        skip_days=21,
        n_long=n_long,
        rebalance_freq=21,
    )
    eq = mom.backtest(prices, START, END, initial_cash=INITIAL)
    if eq.empty or len(eq) < 100:
        return {"label": label, "n_stocks": len(prices), "n_long": n_long,
                "sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0, "end_value": 0.0}
    m = compute_metrics(eq, [], risk_free_rate=RF, periods_per_year=PPY)
    return {
        "label": label,
        "n_stocks": len(prices),
        "n_long": n_long,
        "sharpe": m.sharpe,
        "cagr": m.cagr,
        "max_dd": m.max_drawdown,
        "end_value": float(eq.iloc[-1]),
    }


def _sp_bh(store: DataStore) -> dict:
    sp = store.load("^GSPC", "1d").sort_index()
    sl = sp.loc[(sp.index >= START) & (sp.index <= END), "close"]
    eq = (sl / sl.iloc[0]) * INITIAL
    m = compute_metrics(eq, [], risk_free_rate=RF, periods_per_year=PPY)
    return {"label": "SP500 B&H", "n_stocks": 1, "n_long": 1,
            "sharpe": m.sharpe, "cagr": m.cagr, "max_dd": m.max_drawdown,
            "end_value": float(eq.iloc[-1])}


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    t0 = time.time()

    # Step 1: Get SP500 constituents with sectors
    print("Step 1: Fetching SP500 constituent list...")
    sp500_df = _fetch_sp500_list()
    all_tickers = sp500_df["Symbol"].tolist()
    sector_col = "GICS Sector" if "GICS Sector" in sp500_df.columns else "Sector"
    sector_map = dict(zip(sp500_df["Symbol"], sp500_df[sector_col], strict=False))
    print(f"  {len(all_tickers)} SP500 tickers")
    print(f"  Sectors: {sp500_df[sector_col].nunique()}")

    # Step 2: Fetch data for all tickers
    print("\nStep 2: Fetching price data for all SP500 tickers...")
    n_fetched = _fetch_missing(store, all_tickers)
    print(f"  Fetched {n_fetched} new tickers")

    # Step 3: Load available data
    print("\nStep 3: Loading price data...")
    all_prices = _load_prices(store, all_tickers)
    print(f"  Loaded {len(all_prices)}/{len(all_tickers)} tickers with sufficient data")

    # Sort by market cap proxy (average volume × average close over last year)
    # This gives us a rough market cap ranking without needing a separate API
    mcap_proxy = {}
    for t, df in all_prices.items():
        recent = df.tail(252)
        if len(recent) > 100 and "volume" in recent.columns and "close" in recent.columns:
            avg_vol = recent["volume"].mean()
            avg_close = recent["close"].mean()
            mcap_proxy[t] = avg_vol * avg_close
    ranked_tickers = sorted(mcap_proxy.keys(), key=lambda t: mcap_proxy[t], reverse=True)
    print(f"  Ranked {len(ranked_tickers)} by liquidity proxy (avg vol × avg close)")

    # Step 4: Build universe tiers
    universes = {
        "Top-20 (current)": ranked_tickers[:20],
        "Top-50": ranked_tickers[:50],
        "Top-100": ranked_tickers[:100],
        "Top-200": ranked_tickers[:200],
        "Full SP500": ranked_tickers,
    }

    # Step 5: Run momentum at different universe sizes and n_long values
    print(f"\nStep 4: Running momentum strategies ({len(universes)} universes × 3 n_long values)...")
    results = []
    results.append(_sp_bh(store))  # benchmark

    for uni_name, tickers in universes.items():
        prices_subset = {t: all_prices[t] for t in tickers if t in all_prices}
        for n_long in [5, 10, 20]:
            if n_long > len(prices_subset):
                continue
            label = f"{uni_name}, top-{n_long}"
            elapsed = time.time() - t0
            print(f"  [{len(results)}/16] {label} ({len(prices_subset)} stocks)  elapsed={elapsed:.0f}s")
            result = _run_momentum(prices_subset, n_long, label)
            results.append(result)

    # Step 6: Analyze sector concentration for the best config
    print("\nStep 5: Sector analysis for top configs...")
    # Take the best-performing universe/n_long combo
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe", ascending=False)

    # Run momentum on the best universe and see which sectors it picks
    best = results_df.iloc[1]  # skip SP500 B&H benchmark
    best_label = best["label"]
    print(f"  Best config: {best_label} (Sharpe {best['sharpe']:+.3f})")

    # Get the actual sector picks over the last year
    # Run the momentum model on the full 4y window and snapshot the rankings
    # at 4 quarterly points to see sector rotation
    best_uni_name = best_label.split(",")[0].strip()
    best_n_long = int(best["n_long"])
    best_tickers = universes.get(best_uni_name, ranked_tickers[:50])
    best_prices = {t: all_prices[t] for t in best_tickers if t in all_prices}

    mom = CrossSectionalMomentum(lookback_days=252, skip_days=21, n_long=best_n_long)
    quarterly_dates = pd.date_range("2025-04-01", "2026-04-01", freq="QS", tz="UTC")
    sector_picks = []
    for date in quarterly_dates:
        weights = mom.rank(best_prices, date)
        for ticker, w in weights.items():
            if w > 0:
                sector = sector_map.get(ticker, "Unknown")
                sector_picks.append({"date": date.date(), "ticker": ticker,
                                     "weight": w, "sector": sector})

    sector_df = pd.DataFrame(sector_picks)
    if not sector_df.empty:
        sector_counts = sector_df.groupby("sector")["ticker"].count().sort_values(ascending=False)
        print(f"\n  Sector distribution across {len(quarterly_dates)} quarterly snapshots:")
        for sector, count in sector_counts.items():
            pct = count / len(sector_df) * 100
            print(f"    {sector:<30} {count:>3} picks ({pct:.0f}%)")

    # Current picks with sectors
    current_weights = mom.rank(best_prices, pd.Timestamp("2026-04-11", tz="UTC"))
    current_picks = [(t, w, sector_map.get(t, "?")) for t, w in current_weights.items() if w > 0]
    current_picks.sort(key=lambda x: x[1], reverse=True)

    # Print results
    print("\n" + "=" * 95)
    print("UNIVERSE ANALYSIS — which stocks should the momentum strategy trade?")
    print("=" * 95)
    print(f"  Window: {START.date()} → {END.date()} (4 years)")
    print("  Strategy: 12-1 month momentum, monthly rebalance, 5+5 bps costs")
    print()
    print(f"  {'Config':<30}{'N stocks':>10}{'N long':>8}{'Sharpe':>9}{'CAGR':>9}{'MDD':>8}{'End $':>12}")
    print(f"  {'-'*86}")
    for _, r in results_df.iterrows():
        print(
            f"  {r['label']:<30}{int(r['n_stocks']):>10}{int(r['n_long']):>8}"
            f"{r['sharpe']:>+9.3f}{r['cagr']:>+8.1%}{r['max_dd']:>+7.1%}"
            f"${r['end_value']:>10,.0f}"
        )

    print(f"\n  BEST CONFIG: {best_label}")
    print(f"  Sharpe: {best['sharpe']:+.3f}  CAGR: {best['cagr']:+.1%}  MDD: {best['max_dd']:+.1%}")

    print(f"\n  CURRENT PICKS ({pd.Timestamp('2026-04-11').date()}) for {best_label}:")
    for t, w, sector in current_picks:
        print(f"    {t:<8} {w:.0%}  [{sector}]")

    # Persist
    out = Path(__file__).parent / "data" / "universe_analysis.parquet"
    results_df.to_parquet(out)
    print(f"\n[wrote] {out}")

    out_md = Path(__file__).parent / "UNIVERSE_ANALYSIS.md"
    md = [
        "# Universe analysis — optimal stock universe for momentum",
        "",
        f"**Window**: {START.date()} → {END.date()} (4 years)",
        "**Strategy**: 12-1 month cross-sectional momentum, monthly rebalance",
        "",
        "## Results by universe size and number of positions held",
        "",
        "| Config | N stocks | N long | Sharpe | CAGR | MDD | End $ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in results_df.iterrows():
        md.append(
            f"| `{r['label']}` | {int(r['n_stocks'])} | {int(r['n_long'])} | "
            f"{r['sharpe']:+.3f} | {r['cagr']:+.1%} | {r['max_dd']:+.1%} | "
            f"${r['end_value']:,.0f} |"
        )
    if not sector_df.empty:
        md += [
            "",
            "## Sector concentration (quarterly snapshots, best config)",
            "",
            "| Sector | Picks | % |",
            "|---|---:|---:|",
        ]
        for sector, count in sector_counts.items():
            pct = count / len(sector_df) * 100
            md.append(f"| {sector} | {count} | {pct:.0f}% |")
    md += [
        "",
        "## Current picks with sectors",
        "",
        "| Ticker | Weight | Sector |",
        "|---|---:|---|",
    ]
    for t, w, sector in current_picks:
        md.append(f"| `{t}` | {w:.0%} | {sector} |")
    out_md.write_text("\n".join(md) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
