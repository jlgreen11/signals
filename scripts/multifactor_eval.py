"""Multi-factor model evaluation: momentum + value + quality + vol filter.

Compares MultiFactor (40/30/30 blend) against pure momentum on the same
SP500 universe. Also runs factor-only ablations and applies the NewsFilter
to the current picks.

Output:
  scripts/MULTIFACTOR_RESULTS.md — comparison table + sector distribution
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore
from signals.model.momentum import CrossSectionalMomentum
from signals.model.multifactor import MultiFactor
from signals.model.news_filter import NewsFilter

# ── Configuration ──────────────────────────────────────────────────────────

# SP500 tickers — representative sample (top ~50 by market cap across sectors)
TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "CRM", "INTC", "ORCL", "CSCO",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "CMG",
    # Communication Services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN", "BMY",
    # Financials
    "JPM", "V", "MA", "BAC", "GS", "MS", "BLK", "SCHW", "AXP",
    # Industrials
    "CAT", "UNP", "HON", "BA", "RTX", "DE", "GE", "LMT",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "CL", "PM",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Utilities
    "NEE", "DUK", "SO",
    # Materials
    "LIN", "APD", "SHW",
    # Real Estate
    "PLD", "AMT", "EQIX",
]

# Sector classification for reporting
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech", "AMD": "Tech",
    "ADBE": "Tech", "CRM": "Tech", "INTC": "Tech", "ORCL": "Tech", "CSCO": "Tech",
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "MCD": "ConsDisc",
    "NKE": "ConsDisc", "SBUX": "ConsDisc", "TJX": "ConsDisc", "LOW": "ConsDisc",
    "BKNG": "ConsDisc", "CMG": "ConsDisc",
    "GOOGL": "CommSvc", "META": "CommSvc", "NFLX": "CommSvc", "DIS": "CommSvc",
    "CMCSA": "CommSvc", "VZ": "CommSvc", "T": "CommSvc", "TMUS": "CommSvc",
    "UNH": "Health", "JNJ": "Health", "LLY": "Health", "PFE": "Health",
    "ABBV": "Health", "MRK": "Health", "TMO": "Health", "ABT": "Health",
    "AMGN": "Health", "BMY": "Health",
    "JPM": "Fin", "V": "Fin", "MA": "Fin", "BAC": "Fin", "GS": "Fin",
    "MS": "Fin", "BLK": "Fin", "SCHW": "Fin", "AXP": "Fin",
    "CAT": "Indust", "UNP": "Indust", "HON": "Indust", "BA": "Indust",
    "RTX": "Indust", "DE": "Indust", "GE": "Indust", "LMT": "Indust",
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "COST": "Staples",
    "WMT": "Staples", "CL": "Staples", "PM": "Staples",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "NEE": "Util", "DUK": "Util", "SO": "Util",
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "PLD": "RealEst", "AMT": "RealEst", "EQIX": "RealEst",
}

WINDOW_START = pd.Timestamp("2022-04-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-04-01", tz="UTC")
INITIAL = 10_000.0

RF = historical_usd_rate("2018-2024")
PPY = 252.0


# ── Helpers ────────────────────────────────────────────────────────────────

def load_prices(store: DataStore) -> dict[str, pd.DataFrame]:
    """Load daily OHLCV for all tickers; drop any that are empty."""
    prices: dict[str, pd.DataFrame] = {}
    for sym in TICKERS:
        df = store.load(sym, "1d").sort_index()
        if df.empty:
            print(f"  [WARN] no data for {sym}, skipping")
            continue
        # Normalize timestamps (Alpaca data has 5am UTC timestamps)
        df.index = df.index.normalize()
        prices[sym] = df
    print(f"  Loaded {len(prices)}/{len(TICKERS)} tickers")
    return prices


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


def sector_distribution(tickers: list[str]) -> str:
    """Return a formatted sector distribution string."""
    counts = Counter(SECTOR_MAP.get(t, "Unknown") for t in tickers)
    return ", ".join(f"{s}: {c}" for s, c in sorted(counts.items(), key=lambda x: -x[1]))


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    store = DataStore(SETTINGS.data.dir)

    print("=" * 70)
    print("Multi-Factor Model Evaluation")
    print("=" * 70)

    # Step 1: Load price data
    print("\n[1] Loading price data...")
    prices = load_prices(store)
    if len(prices) < 10:
        print("  FATAL: fewer than 10 tickers loaded, cannot run eval")
        return

    # Step 2: Load or fetch fundamentals
    print("\n[2] Loading fundamentals...")
    mf = MultiFactor(
        momentum_weight=0.40, value_weight=0.30, quality_weight=0.30,
        n_long=10, vol_filter_quantile=0.75,
        lookback_days=252, skip_days=21, rebalance_freq=21,
        commission_bps=5.0, slippage_bps=5.0,
    )
    fund = mf.fetch_fundamentals(list(prices.keys()))
    print(f"  Fundamentals loaded for {len(fund)} tickers")

    # Step 3: Run backtests
    print("\n[3] Running 4-year backtests (2022-04 -> 2026-04)...")

    start_str = str(WINDOW_START.date())
    end_str = str(WINDOW_END.date())

    # Pure momentum
    mom = CrossSectionalMomentum(
        lookback_days=252, skip_days=21, n_long=10, rebalance_freq=21,
        commission_bps=5.0, slippage_bps=5.0,
    )
    eq_momentum = mom.backtest(prices, start=start_str, end=end_str, initial_cash=INITIAL)

    # MultiFactor composite (40/30/30)
    eq_multifactor = mf.backtest(prices, fund, start=start_str, end=end_str, initial_cash=INITIAL)

    # Momentum-only (100/0/0)
    mf_mom_only = MultiFactor(
        momentum_weight=1.0, value_weight=0.0, quality_weight=0.0,
        n_long=10, vol_filter_quantile=0.75,
        lookback_days=252, skip_days=21, rebalance_freq=21,
    )
    eq_mom_only = mf_mom_only.backtest(
        prices, fund, start=start_str, end=end_str, initial_cash=INITIAL,
    )

    # Value-heavy (10/80/10)
    mf_value = MultiFactor(
        momentum_weight=0.10, value_weight=0.80, quality_weight=0.10,
        n_long=10, vol_filter_quantile=0.75,
    )
    eq_value = mf_value.backtest(
        prices, fund, start=start_str, end=end_str, initial_cash=INITIAL,
    )

    # Quality-heavy (10/10/80)
    mf_quality = MultiFactor(
        momentum_weight=0.10, value_weight=0.10, quality_weight=0.80,
        n_long=10, vol_filter_quantile=0.75,
    )
    eq_quality = mf_quality.backtest(
        prices, fund, start=start_str, end=end_str, initial_cash=INITIAL,
    )

    # Collect results
    curves = {
        "Pure Momentum (CSM)": eq_momentum,
        "MultiFactor 40/30/30": eq_multifactor,
        "Mom-Only + VolFilter": eq_mom_only,
        "Value-Heavy 10/80/10": eq_value,
        "Quality-Heavy 10/10/80": eq_quality,
    }

    print(f"\n  {'Strategy':<25} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Final$':>10}")
    print("  " + "-" * 63)

    rows: list[dict] = []
    for name, eq in curves.items():
        if len(eq) == 0:
            print(f"  {name:<25} (no data)")
            continue
        row = metrics_row(name, eq)
        rows.append(row)
        print(
            f"  {row['strategy']:<25} {row['sharpe']:>8.3f} "
            f"{row['cagr']:>7.1%} {row['max_drawdown']:>7.1%} "
            f"{row['final_equity']:>10,.0f}"
        )

    # Step 4: Compare current picks
    print("\n[4] Current picks comparison...")
    latest_dates = [df.index.max() for df in prices.values() if not df.empty]
    as_of = max(latest_dates) if latest_dates else WINDOW_END

    mom_weights = mom.rank(prices, as_of_date=as_of)
    mf_weights = mf.rank(prices, fund, as_of_date=as_of)

    mom_picks = sorted([t for t, w in mom_weights.items() if w > 0])
    mf_picks = sorted([t for t, w in mf_weights.items() if w > 0])

    print(f"\n  Pure Momentum picks:  {', '.join(mom_picks)}")
    print(f"  Sector distribution:  {sector_distribution(mom_picks)}")
    print(f"\n  MultiFactor picks:    {', '.join(mf_picks)}")
    print(f"  Sector distribution:  {sector_distribution(mf_picks)}")

    overlap = set(mom_picks) & set(mf_picks)
    print(f"\n  Overlap: {len(overlap)} stocks ({', '.join(sorted(overlap)) or 'none'})")

    # Step 5: News filter on multifactor picks
    print("\n[5] News filter on multifactor picks...")
    nf = NewsFilter(lookback_days=7, max_risk_score=3)
    news_results = []
    for ticker in mf_picks:
        result = nf.check_ticker(ticker)
        news_results.append(result)
        flag = f" *** {result['recommendation']} ***" if result["flagged"] else ""
        print(
            f"  {ticker:6s}  risk={result['risk_score']:2d}  "
            f"articles={result['n_articles']:2d}  "
            f"rec={result['recommendation']}{flag}"
        )

    # Apply filter to weights
    filtered_weights = nf.filter_signals(mf_weights)
    filtered_picks = sorted([t for t, w in filtered_weights.items() if w > 0])
    print(f"\n  After news filter:    {', '.join(filtered_picks)}")
    print(f"  Sector distribution:  {sector_distribution(filtered_picks)}")

    # Step 6: Write results
    print("\n[6] Writing results...")
    md_path = Path(__file__).parent / "MULTIFACTOR_RESULTS.md"
    with open(md_path, "w") as f:
        f.write("# Multi-Factor Model Results\n\n")
        f.write("**Model:** Momentum (40%) + Value (30%) + Quality (30%)\n")
        f.write(f"- Universe: {len(prices)} stocks (SP500 subset)\n")
        f.write("- Lookback: 252 days, skip: 21 days, top 10 equal-weight, monthly rebalance\n")
        f.write("- Volatility filter: exclude top 25% by 63-day realized vol\n")
        f.write("- Costs: 5 bps commission + 5 bps slippage per rebalance trade\n")
        f.write(f"- Risk-free rate: {RF:.2%} (3M T-bill avg 2018-2024)\n")
        f.write(f"- Annualization: {PPY:.0f} days/year\n\n")

        f.write("## Trailing 4-Year (2022-04-01 to 2026-04-01)\n\n")
        f.write(
            f"| {'Strategy':<25} | {'Sharpe':>8} | {'CAGR':>8} | "
            f"{'MaxDD':>8} | {'Final$':>10} |\n"
        )
        f.write(f"|{'-' * 27}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 12}|\n")
        for row in rows:
            f.write(
                f"| {row['strategy']:<25} | {row['sharpe']:>8.3f} | "
                f"{row['cagr']:>7.1%} | {row['max_drawdown']:>7.1%} | "
                f"{row['final_equity']:>10,.0f} |\n"
            )

        f.write("\n## Current Picks\n\n")
        f.write(f"**Pure Momentum:** {', '.join(mom_picks)}\n")
        f.write(f"- Sectors: {sector_distribution(mom_picks)}\n\n")
        f.write(f"**MultiFactor:** {', '.join(mf_picks)}\n")
        f.write(f"- Sectors: {sector_distribution(mf_picks)}\n\n")
        f.write(f"**Overlap:** {len(overlap)} ({', '.join(sorted(overlap)) or 'none'})\n\n")

        f.write("## News Filter Results\n\n")
        f.write("| Ticker | Risk Score | Articles | Recommendation |\n")
        f.write("|--------|-----------|----------|----------------|\n")
        for r in news_results:
            f.write(
                f"| {r['ticker']} | {r['risk_score']} | "
                f"{r['n_articles']} | {r['recommendation']} |\n"
            )
        f.write(f"\n**After filter:** {', '.join(filtered_picks)}\n")
        f.write(f"- Sectors: {sector_distribution(filtered_picks)}\n")

        f.write("\n---\n")
        f.write("*Generated by `scripts/multifactor_eval.py`*\n")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Results: {md_path}")


if __name__ == "__main__":
    main()
