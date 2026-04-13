"""Survivorship-bias-free momentum backtest using historical SP500 constituents.

Uses the fja05680/sp500 dataset of daily SP500 membership from 1996-2026.
At each monthly rebalance, momentum ranks ONLY the stocks that were actually
in the SP500 on that date — including companies that later went bankrupt,
got delisted, or were acquired.

Simulates $100K from 2000-01-01 to present with monthly rebalancing.

Usage:
    python scripts/survivorship_free_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.backtest.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONSTITUENT_CSV = "/tmp/sp500/S&P 500 Historical Components & Changes(01-17-2026).csv"
CACHE_DIR = Path("/tmp/sp500_price_cache")
START = "2000-01-01"
END = "2026-04-11"
INITIAL_CASH = 100_000.0
N_LONG = 10
LOOKBACK_DAYS = 252
SKIP_DAYS = 21
REBALANCE_FREQ = 21  # ~monthly in trading days
COMMISSION_BPS = 5.0
SLIPPAGE_BPS = 5.0


# ---------------------------------------------------------------------------
# Historical constituents loader
# ---------------------------------------------------------------------------
def load_constituent_map(csv_path: str = CONSTITUENT_CSV) -> dict[str, list[str]]:
    """Parse CSV into {date_str: [tickers]} map."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    result = {}
    for _, row in df.iterrows():
        d = row["date"].strftime("%Y-%m-%d")
        tickers = [t.strip() for t in row["tickers"].split(",") if t.strip()]
        result[d] = tickers
    return result


def get_constituents_for_date(
    constituent_map: dict[str, list[str]], target_date: pd.Timestamp
) -> list[str]:
    """Get SP500 constituents for a given date (nearest prior date in map)."""
    target_str = target_date.strftime("%Y-%m-%d")
    if target_str in constituent_map:
        return constituent_map[target_str]

    # Find nearest prior date
    all_dates = sorted(constituent_map.keys())
    prior = [d for d in all_dates if d <= target_str]
    if prior:
        return constituent_map[prior[-1]]
    return constituent_map[all_dates[0]]


def get_all_unique_tickers(
    constituent_map: dict[str, list[str]], start: str = START
) -> set[str]:
    """Get all unique tickers that were ever in SP500 since start date."""
    tickers = set()
    for date_str, ticker_list in constituent_map.items():
        if date_str >= start:
            tickers.update(ticker_list)
    return tickers


# ---------------------------------------------------------------------------
# Price data fetcher with caching
# ---------------------------------------------------------------------------
# Known legitimate high-priced / high-growth stocks (don't filter these)
LEGIT_HIGH_RATIO = {
    "AAPL", "AMZN", "NVDA", "NFLX", "MNST", "DECK", "AXON", "SBAC",
    "LRCX", "FIX", "BRK.B", "GOOG", "GOOGL", "MSFT", "META", "TSLA",
    "AVGO", "COST", "UNH", "LLY", "NVO", "NVR", "SEB", "BKNG", "AZO",
    "CMG", "ORLY", "MTD", "MELI",
}

# Max reasonable close price for non-whitelisted tickers
MAX_REASONABLE_PRICE = 5000.0
# Max single-day return (300%)
MAX_DAILY_RETURN = 3.0


def _filter_bad_data(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Remove tickers with clearly corrupted yfinance data."""
    filtered = {}
    removed = []
    for t, df in prices.items():
        if t in LEGIT_HIGH_RATIO:
            filtered[t] = df
            continue
        max_close = df["close"].max()
        if max_close > MAX_REASONABLE_PRICE:
            removed.append((t, max_close))
            continue
        # Check for single-day jumps > 300%
        rets = df["close"].pct_change().abs()
        if rets.max() > MAX_DAILY_RETURN:
            removed.append((t, f"max daily ret {rets.max():.0%}"))
            continue
        filtered[t] = df
    if removed:
        print(f"  Filtered {len(removed)} tickers with corrupted data: "
              f"{[r[0] for r in removed[:15]]}{'...' if len(removed) > 15 else ''}")
    return filtered


def fetch_and_cache_prices(tickers: set[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch daily prices via yfinance, cache to parquet files."""
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prices = {}
    to_fetch = []

    # Check cache first — empty parquet = "tried, no data available"
    for t in sorted(tickers):
        safe_name = t.replace(".", "_").replace("/", "_")
        cache_file = CACHE_DIR / f"{safe_name}.parquet"
        failed_file = CACHE_DIR / f"{safe_name}.failed"
        if failed_file.exists():
            continue  # known dead ticker
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 0:
                    prices[t] = df
                continue  # even if empty, don't re-fetch
            except Exception:
                pass
        to_fetch.append(t)

    if not to_fetch:
        print(f"All {len(prices)} tickers loaded from cache")
        prices = _filter_bad_data(prices)
        return prices

    print(f"Loaded {len(prices)} from cache, fetching {len(to_fetch)} from yfinance...")

    # Batch download in chunks of 50
    batch_size = 50
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i : i + batch_size]
        batch_str = " ".join(batch)
        pct = (i + len(batch)) / len(to_fetch) * 100
        print(f"  Fetching batch {i // batch_size + 1} "
              f"({i + 1}-{i + len(batch)} of {len(to_fetch)}, {pct:.0f}%)...")

        try:
            data = yf.download(
                batch_str,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            print(f"    Batch download error: {e}")
            continue

        if data.empty:
            continue

        # yf.download with multiple tickers returns MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            for t in batch:
                try:
                    df_t = data.xs(t, axis=1, level="Ticker").copy()
                    df_t.columns = [c.lower() for c in df_t.columns]
                    df_t = df_t.dropna(subset=["close"])
                    if len(df_t) > 0:
                        df_t.index = df_t.index.tz_localize("UTC") if df_t.index.tz is None else df_t.index.tz_convert("UTC")
                        df_t.index = df_t.index.normalize()
                        prices[t] = df_t
                        safe_name = t.replace(".", "_").replace("/", "_")
                        df_t.to_parquet(CACHE_DIR / f"{safe_name}.parquet")
                except Exception:
                    pass
        else:
            # Single ticker returned
            if len(batch) == 1:
                t = batch[0]
                df_t = data.copy()
                df_t.columns = [c.lower() for c in df_t.columns]
                df_t = df_t.dropna(subset=["close"])
                if len(df_t) > 0:
                    df_t.index = df_t.index.tz_localize("UTC") if df_t.index.tz is None else df_t.index.tz_convert("UTC")
                    df_t.index = df_t.index.normalize()
                    prices[t] = df_t
                    safe_name = t.replace(".", "_").replace("/", "_")
                    df_t.to_parquet(CACHE_DIR / f"{safe_name}.parquet")

        # Mark failed tickers so we don't re-fetch
        for t in batch:
            if t not in prices:
                safe_name = t.replace(".", "_").replace("/", "_")
                failed_file = CACHE_DIR / f"{safe_name}.failed"
                failed_file.touch()

        # Rate limit
        time.sleep(0.5)

    print(f"Total tickers with price data: {len(prices)}")
    prices = _filter_bad_data(prices)
    return prices


# ---------------------------------------------------------------------------
# Momentum ranking (point-in-time)
# ---------------------------------------------------------------------------
def momentum_rank(
    prices_dict: dict[str, pd.DataFrame],
    eligible_tickers: list[str],
    as_of_date: pd.Timestamp,
    lookback: int = LOOKBACK_DAYS,
    skip: int = SKIP_DAYS,
    n_long: int = N_LONG,
) -> dict[str, float]:
    """Rank eligible tickers by 12-1 month momentum, return top-N equal weights."""
    scores = {}
    for t in eligible_tickers:
        if t not in prices_dict:
            continue
        df = prices_dict[t]
        mask = df.index <= as_of_date
        available = df.loc[mask]
        if len(available) < lookback + skip:
            continue
        recent = available.iloc[-(skip):]
        past = available.iloc[-(lookback + skip) : -skip]
        if len(past) < lookback * 0.8:  # require 80% of lookback
            continue
        try:
            ret = float(recent.iloc[-1]["close"] / past.iloc[0]["close"]) - 1.0
            scores[t] = ret
        except (IndexError, KeyError, ZeroDivisionError):
            continue

    if not scores:
        return {}

    # Rank and select top N
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in ranked[:n_long]]
    w = 1.0 / len(selected)
    return {t: w for t in selected}


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------
def run_backtest(
    prices_dict: dict[str, pd.DataFrame],
    constituent_map: dict[str, list[str]],
    start: str = START,
    end: str = END,
    label: str = "Momentum",
) -> pd.Series:
    """Run survivorship-bias-free momentum backtest."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    # Build unified trading calendar
    all_dates: set[pd.Timestamp] = set()
    for df in prices_dict.values():
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        all_dates.update(df.index[mask])
    trading_dates = sorted(all_dates)

    if not trading_dates:
        return pd.Series(dtype=float)

    holdings: dict[str, float] = {}  # ticker -> shares
    cash = INITIAL_CASH
    cost_rate = (COMMISSION_BPS + SLIPPAGE_BPS) * 1e-4
    equity_points: list[tuple[pd.Timestamp, float]] = []
    bars_since_rebalance = REBALANCE_FREQ  # trigger on first bar
    rebalance_count = 0

    for i, date in enumerate(trading_dates):
        # Get current prices
        prices: dict[str, float] = {}
        for sym, df in prices_dict.items():
            if date in df.index:
                prices[sym] = float(df.loc[date, "close"])

        bars_since_rebalance += 1
        if bars_since_rebalance >= REBALANCE_FREQ:
            # Get POINT-IN-TIME SP500 constituents
            eligible = get_constituents_for_date(constituent_map, date)

            # Rank momentum only among eligible tickers
            new_weights = momentum_rank(prices_dict, eligible, date)

            if new_weights:
                rebalance_count += 1
                # Compute current equity
                equity = cash
                for sym, shares in holdings.items():
                    if sym in prices:
                        equity += shares * prices[sym]

                if equity > 0:
                    # Transaction costs
                    for sym in set(list(holdings.keys()) + list(new_weights.keys())):
                        if sym not in prices:
                            continue
                        price = prices[sym]
                        current_value = holdings.get(sym, 0.0) * price
                        target_value = new_weights.get(sym, 0.0) * equity
                        trade_value = abs(target_value - current_value)
                        if trade_value > 1e-6:
                            cash -= trade_value * cost_rate

                    # Recompute equity after costs
                    equity_after = cash
                    for sym, shares in holdings.items():
                        if sym in prices:
                            equity_after += shares * prices[sym]

                    # Liquidate all
                    for sym, shares in list(holdings.items()):
                        if sym in prices:
                            cash += shares * prices[sym]
                        holdings[sym] = 0.0
                    holdings = {}

                    # Buy new positions
                    for sym, w in new_weights.items():
                        if sym in prices and prices[sym] > 0:
                            target_value = w * equity_after
                            holdings[sym] = target_value / prices[sym]
                            cash -= target_value

                bars_since_rebalance = 0

        # Mark-to-market
        equity = cash
        for sym, shares in holdings.items():
            if sym in prices:
                equity += shares * prices[sym]
        equity_points.append((date, equity))

        # Progress
        if (i + 1) % 500 == 0 or i == len(trading_dates) - 1:
            print(f"  [{label}] Day {i + 1}/{len(trading_dates)}: "
                  f"equity=${equity:,.0f}, rebalances={rebalance_count}, "
                  f"holdings={len([s for s, sh in holdings.items() if sh > 0])}")

    if not equity_points:
        return pd.Series(dtype=float)

    ts, eq = zip(*equity_points)
    return pd.Series(eq, index=pd.DatetimeIndex(ts), name=label)


def run_spy_benchmark(
    prices_dict: dict[str, pd.DataFrame],
    start: str = START,
    end: str = END,
) -> pd.Series:
    """Buy-and-hold SPY (or ^GSPC) benchmark."""
    import yfinance as yf

    cache_file = CACHE_DIR / "SPY_benchmark.parquet"
    if cache_file.exists():
        spy = pd.read_parquet(cache_file)
    else:
        spy = yf.download("SPY", start="1999-01-01", end=end, auto_adjust=True, progress=False)
        spy.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in spy.columns]
        spy.index = spy.index.tz_localize("UTC") if spy.index.tz is None else spy.index.tz_convert("UTC")
        spy.index = spy.index.normalize()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        spy.to_parquet(cache_file)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    spy = spy.loc[(spy.index >= start_ts) & (spy.index <= end_ts)]

    if spy.empty:
        return pd.Series(dtype=float)

    # Buy and hold from day 1
    initial_price = float(spy.iloc[0]["close"])
    shares = INITIAL_CASH / initial_price
    equity = spy["close"].astype(float) * shares
    equity.name = "SPY B&H"
    return equity


# ---------------------------------------------------------------------------
# Survivorship-biased comparison (uses today's SP500 at all times)
# ---------------------------------------------------------------------------
def run_biased_backtest(
    prices_dict: dict[str, pd.DataFrame],
    current_tickers: list[str],
    start: str = START,
    end: str = END,
) -> pd.Series:
    """Run momentum using ONLY today's SP500 members (survivorship-biased)."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    all_dates: set[pd.Timestamp] = set()
    for df in prices_dict.values():
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        all_dates.update(df.index[mask])
    trading_dates = sorted(all_dates)

    if not trading_dates:
        return pd.Series(dtype=float)

    holdings: dict[str, float] = {}
    cash = INITIAL_CASH
    cost_rate = (COMMISSION_BPS + SLIPPAGE_BPS) * 1e-4
    equity_points: list[tuple[pd.Timestamp, float]] = []
    bars_since_rebalance = REBALANCE_FREQ
    rebalance_count = 0

    for i, date in enumerate(trading_dates):
        prices: dict[str, float] = {}
        for sym, df in prices_dict.items():
            if date in df.index:
                prices[sym] = float(df.loc[date, "close"])

        bars_since_rebalance += 1
        if bars_since_rebalance >= REBALANCE_FREQ:
            # BIASED: always use current SP500 members
            new_weights = momentum_rank(prices_dict, current_tickers, date)

            if new_weights:
                rebalance_count += 1
                equity = cash
                for sym, shares in holdings.items():
                    if sym in prices:
                        equity += shares * prices[sym]

                if equity > 0:
                    for sym in set(list(holdings.keys()) + list(new_weights.keys())):
                        if sym not in prices:
                            continue
                        price = prices[sym]
                        current_value = holdings.get(sym, 0.0) * price
                        target_value = new_weights.get(sym, 0.0) * equity
                        trade_value = abs(target_value - current_value)
                        if trade_value > 1e-6:
                            cash -= trade_value * cost_rate

                    equity_after = cash
                    for sym, shares in holdings.items():
                        if sym in prices:
                            equity_after += shares * prices[sym]

                    for sym, shares in list(holdings.items()):
                        if sym in prices:
                            cash += shares * prices[sym]
                        holdings[sym] = 0.0
                    holdings = {}

                    for sym, w in new_weights.items():
                        if sym in prices and prices[sym] > 0:
                            target_value = w * equity_after
                            holdings[sym] = target_value / prices[sym]
                            cash -= target_value

                bars_since_rebalance = 0

        equity = cash
        for sym, shares in holdings.items():
            if sym in prices:
                equity += shares * prices[sym]
        equity_points.append((date, equity))

        if (i + 1) % 500 == 0 or i == len(trading_dates) - 1:
            print(f"  [Biased Mom] Day {i + 1}/{len(trading_dates)}: "
                  f"equity=${equity:,.0f}")

    if not equity_points:
        return pd.Series(dtype=float)

    ts, eq = zip(*equity_points)
    return pd.Series(eq, index=pd.DatetimeIndex(ts), name="Biased Momentum")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("SURVIVORSHIP-BIAS-FREE MOMENTUM BACKTEST")
    print(f"$100K | {START} to {END} | Monthly rebalance | Top-{N_LONG}")
    print("=" * 70)

    # Step 1: Load constituent map
    print("\n[1/4] Loading historical SP500 constituent data...")
    constituent_map = load_constituent_map()
    all_tickers = get_all_unique_tickers(constituent_map, START)
    print(f"  {len(constituent_map)} daily snapshots, {len(all_tickers)} unique tickers since {START}")

    # Step 2: Fetch price data
    print(f"\n[2/4] Fetching price data for {len(all_tickers)} tickers...")
    prices_dict = fetch_and_cache_prices(all_tickers, "1998-01-01", END)
    print(f"  Got data for {len(prices_dict)} tickers")

    # Current SP500 for biased comparison
    current_sp500 = pd.read_csv("/tmp/sp500_with_sectors.csv")["Symbol"].tolist()

    # Step 3: Run backtests
    print("\n[3/4] Running backtests...")
    print("\n--- Survivorship-Bias-Free Momentum ---")
    equity_free = run_backtest(prices_dict, constituent_map, label="Bias-Free Mom")

    print("\n--- Survivorship-Biased Momentum (today's SP500 only) ---")
    equity_biased = run_biased_backtest(prices_dict, current_sp500)

    print("\n--- SPY Buy & Hold ---")
    equity_spy = run_spy_benchmark(prices_dict)

    # Step 4: Compare
    print("\n" + "=" * 70)
    print("[4/4] RESULTS")
    print("=" * 70)

    results = []
    for name, equity in [
        ("Bias-Free Momentum", equity_free),
        ("Biased Momentum", equity_biased),
        ("SPY Buy & Hold", equity_spy),
    ]:
        if equity.empty:
            print(f"  {name}: NO DATA")
            continue

        m = compute_metrics(equity, trades=[], periods_per_year=252)
        results.append({
            "Strategy": name,
            "Final Equity": f"${m.final_equity:,.0f}",
            "CAGR": f"{m.cagr:.2%}",
            "Sharpe": f"{m.sharpe:.3f}",
            "Max Drawdown": f"{m.max_drawdown:.2%}",
            "Start": equity.index[0].strftime("%Y-%m-%d"),
            "End": equity.index[-1].strftime("%Y-%m-%d"),
        })
        print(f"\n  {name}:")
        print(f"    ${INITIAL_CASH:,.0f} -> ${m.final_equity:,.0f}")
        print(f"    CAGR: {m.cagr:.2%}  Sharpe: {m.sharpe:.3f}  Max DD: {m.max_drawdown:.2%}")

    # Era breakdown
    eras = {
        "Dot-com crash (2000-2002)": ("2000-01-01", "2002-12-31"),
        "Recovery (2003-2006)": ("2003-01-01", "2006-12-31"),
        "Financial crisis (2007-2009)": ("2007-01-01", "2009-12-31"),
        "Bull run (2010-2019)": ("2010-01-01", "2019-12-31"),
        "COVID + aftermath (2020-2022)": ("2020-01-01", "2022-12-31"),
        "Recent (2023-2026)": ("2023-01-01", "2026-12-31"),
    }

    print("\n\nERA-BY-ERA SHARPE COMPARISON:")
    print("-" * 70)
    era_results = []
    for era_name, (era_start, era_end) in eras.items():
        s = pd.Timestamp(era_start, tz="UTC")
        e = pd.Timestamp(era_end, tz="UTC")
        row = {"Era": era_name}
        for strat_name, equity in [
            ("Bias-Free", equity_free),
            ("Biased", equity_biased),
            ("SPY", equity_spy),
        ]:
            sub = equity[(equity.index >= s) & (equity.index <= e)]
            if len(sub) > 50:
                m = compute_metrics(sub, trades=[], periods_per_year=252)
                row[f"{strat_name} Sharpe"] = f"{m.sharpe:.3f}"
                row[f"{strat_name} CAGR"] = f"{m.cagr:.2%}"
            else:
                row[f"{strat_name} Sharpe"] = "N/A"
                row[f"{strat_name} CAGR"] = "N/A"
        era_results.append(row)

    era_df = pd.DataFrame(era_results)
    print(era_df.to_string(index=False))

    # Survivorship bias impact
    if not equity_free.empty and not equity_biased.empty:
        print("\n\nSURVIVORSHIP BIAS IMPACT:")
        print("-" * 70)
        m_free = compute_metrics(equity_free, trades=[], periods_per_year=252)
        m_biased = compute_metrics(equity_biased, trades=[], periods_per_year=252)
        print(f"  Bias-free Sharpe:  {m_free.sharpe:.3f}")
        print(f"  Biased Sharpe:     {m_biased.sharpe:.3f}")
        print(f"  Difference:        {m_free.sharpe - m_biased.sharpe:.3f}")
        print(f"  Bias-free CAGR:    {m_free.cagr:.2%}")
        print(f"  Biased CAGR:       {m_biased.cagr:.2%}")
        print(f"  CAGR difference:   {m_free.cagr - m_biased.cagr:.2%}")
        if m_biased.sharpe > 0:
            pct = (m_biased.sharpe - m_free.sharpe) / m_free.sharpe * 100
            print(f"\n  Survivorship bias inflates Sharpe by {pct:.1f}%" if pct > 0
                  else f"\n  Survivorship bias effect on Sharpe: {pct:.1f}%")

    # Save results
    if results:
        out_df = pd.DataFrame(results)
        out_path = Path(__file__).parent / "SURVIVORSHIP_FREE_RESULTS.md"
        with open(out_path, "w") as f:
            f.write("# Survivorship-Bias-Free Momentum Test\n\n")
            f.write(f"$100K initial | {START} to {END} | Monthly rebalance | Top-{N_LONG}\n\n")
            f.write("## Overall Results\n\n")
            f.write(out_df.to_markdown(index=False))
            f.write("\n\n## Era Breakdown\n\n")
            f.write(era_df.to_markdown(index=False))
            f.write("\n\n## Methodology\n\n")
            f.write("- **Bias-Free**: At each rebalance, uses the ACTUAL SP500 constituent list "
                    "from that date (fja05680/sp500 dataset). Includes companies that later went "
                    "bankrupt (Enron, Lehman), were acquired, or were delisted.\n")
            f.write("- **Biased**: Uses today's SP500 members for all historical dates. "
                    "This is the standard (flawed) approach most backtests use.\n")
            f.write("- **SPY B&H**: Buy $100K of SPY on day 1, hold forever.\n")
            f.write(f"\nData source: {len(prices_dict)} tickers with price data out of "
                    f"{len(all_tickers)} unique historical constituents.\n")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
