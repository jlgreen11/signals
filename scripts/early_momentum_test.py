"""Early-breakout momentum with sector diversification and fixed 4-month hold.

Instead of buying the top trailing-12-month winners (already extended),
this strategy finds stocks with ACCELERATING momentum — where recent
3-month return is strong but 12-month return is still moderate. This
catches breakouts early before they're crowded.

Rules:
  1. Momentum acceleration = 3-month return - (12-month return / 4)
     High accel = stock is inflecting upward recently
  2. Require 3-month return > 0 (actually trending up)
  3. Require 12-month return < 100% (not already a moonshot)
  4. Max 2 stocks per GICS sector
  5. Hold for exactly 4 months (84 trading days), no early exit
  6. Pick top 10 at each entry window

Tested bias-free using point-in-time SP500 constituents (2000-2026).

Usage:
    python3 -u scripts/early_momentum_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.backtest.metrics import compute_metrics

# -------------------------------------------------------
# Config
# -------------------------------------------------------
CACHE_DIR = Path("/tmp/sp500_price_cache")
CONSTITUENT_CSV = "/tmp/sp500/S&P 500 Historical Components & Changes(01-17-2026).csv"
SECTOR_CSV = "/tmp/sp500_with_sectors.csv"
START = "2000-01-01"
END = "2026-04-11"
INITIAL_CASH = 100_000.0
COST_RATE = 10e-4  # 5+5 bps
N_LONG = 10
HOLD_DAYS = 84  # ~4 months in trading days
REBALANCE_FREQ = 21  # check monthly for new entries
MAX_PER_SECTOR = 2

LEGIT_HIGH = {
    "AAPL", "AMZN", "NVDA", "NFLX", "MNST", "DECK", "AXON", "SBAC",
    "LRCX", "FIX", "BRK.B", "GOOG", "GOOGL", "MSFT", "META", "TSLA",
    "AVGO", "COST", "UNH", "LLY", "NVO", "NVR", "SEB", "BKNG", "AZO",
    "CMG", "ORLY", "MTD", "MELI",
}


# -------------------------------------------------------
# Data loading
# -------------------------------------------------------
def load_prices() -> dict[str, pd.DataFrame]:
    prices = {}
    for f in CACHE_DIR.glob("*.parquet"):
        try:
            d = pd.read_parquet(f)
            if len(d) < 100:
                continue
            ticker = f.stem.replace("_", ".")
            if ticker not in LEGIT_HIGH:
                if d["close"].max() > 5000:
                    continue
                if d["close"].pct_change().abs().max() > 3.0:
                    continue
            prices[ticker] = d
        except Exception:
            pass
    return prices


def load_constituents() -> dict[str, list[str]]:
    df = pd.read_csv(CONSTITUENT_CSV)
    df["date"] = pd.to_datetime(df["date"])
    result = {}
    for _, row in df.iterrows():
        d = row["date"].strftime("%Y-%m-%d")
        result[d] = [t.strip() for t in row["tickers"].split(",") if t.strip()]
    return result


def load_sectors() -> dict[str, str]:
    """Load ticker -> GICS sector mapping."""
    df = pd.read_csv(SECTOR_CSV)
    sectors = {}
    for _, row in df.iterrows():
        sectors[row["Symbol"]] = row["GICS Sector"]

    # Try to get sectors for historical tickers from yfinance cache
    # For tickers not in current SP500, assign "Unknown"
    return sectors


def get_constituents(cmap: dict, cmap_dates: list[str], date: pd.Timestamp) -> list[str]:
    d_str = date.strftime("%Y-%m-%d")
    lo, hi = 0, len(cmap_dates) - 1
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if cmap_dates[mid] <= d_str:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return cmap[cmap_dates[best]]


def build_close_matrix(
    prices_dict: dict[str, pd.DataFrame], trading_dates: list[pd.Timestamp]
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    tickers = sorted(prices_dict.keys())
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    mat = np.full((len(trading_dates), len(tickers)), np.nan)
    date_to_row = {d: i for i, d in enumerate(trading_dates)}
    for t, df in prices_dict.items():
        col = ticker_to_idx[t]
        for dt in df.index:
            if dt in date_to_row:
                mat[date_to_row[dt], col] = float(df.loc[dt, "close"])
    return mat, tickers, ticker_to_idx


# -------------------------------------------------------
# Early momentum scoring
# -------------------------------------------------------
def score_early_momentum(
    close_mat: np.ndarray,
    row_idx: int,
    eligible_cols: list[int],
    tickers: list[str],
    sectors: dict[str, str],
) -> list[tuple[int, float, float, float, str]]:
    """Score stocks by momentum acceleration with sector info.

    Returns: [(col_idx, accel_score, ret_3m, ret_12m, sector), ...] sorted by accel desc.
    """
    if row_idx < 252:
        return []

    candidates = []
    for col in eligible_cols:
        # 3-month return (~63 trading days)
        p_now = close_mat[row_idx, col]
        p_3m = close_mat[row_idx - 63, col] if row_idx >= 63 else np.nan
        p_12m = close_mat[row_idx - 252, col] if row_idx >= 252 else np.nan

        if np.isnan(p_now) or np.isnan(p_3m) or np.isnan(p_12m):
            continue
        if p_3m <= 0 or p_12m <= 0:
            continue

        ret_3m = p_now / p_3m - 1.0
        ret_12m = p_now / p_12m - 1.0

        # Filter: must be trending up recently
        if ret_3m <= 0:
            continue

        # Filter: not already a moonshot (avoid buying at the top)
        if ret_12m > 1.00:  # cap at 100% trailing 12m
            continue

        # Acceleration: how much is recent momentum exceeding long-term pace?
        # Normalized 12m pace per quarter vs actual 3m
        quarterly_pace = ret_12m / 4.0
        accel = ret_3m - quarterly_pace

        ticker = tickers[col]
        sector = sectors.get(ticker, "Unknown")
        candidates.append((col, accel, ret_3m, ret_12m, sector))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def select_with_sector_cap(
    candidates: list[tuple[int, float, float, float, str]],
    existing_sectors: dict[str, int],  # sector -> count in current portfolio
    n_slots: int,
) -> list[tuple[int, float, float, float, str]]:
    """Pick top candidates respecting sector cap."""
    selected = []
    sector_count = dict(existing_sectors)

    for cand in candidates:
        if len(selected) >= n_slots:
            break
        sector = cand[4]
        current = sector_count.get(sector, 0)
        if current >= MAX_PER_SECTOR:
            continue
        selected.append(cand)
        sector_count[sector] = current + 1

    return selected


# -------------------------------------------------------
# Backtest: early momentum with fixed hold + sector cap
# -------------------------------------------------------
def run_early_momentum(
    close_mat: np.ndarray,
    tickers: list[str],
    ticker_to_idx: dict[str, int],
    trading_dates: list[pd.Timestamp],
    cmap: dict[str, list[str]],
    cmap_dates: list[str],
    sectors: dict[str, str],
    label: str = "Early Momentum",
) -> tuple[pd.Series, list[dict]]:
    # holdings: col_idx -> {entry_price, entry_row, shares, sector}
    holdings: dict[int, dict] = {}
    cash = INITIAL_CASH
    equity_points = []
    bars_since = REBALANCE_FREQ
    trade_log = []

    n_rows = len(trading_dates)

    for row in range(n_rows):
        # --- Check fixed-hold exits ---
        for col in list(holdings.keys()):
            h = holdings[col]
            if (row - h["entry_row"]) >= HOLD_DAYS:
                price = close_mat[row, col]
                if not np.isnan(price):
                    pnl = price / h["entry_price"] - 1.0
                    sell_val = h["shares"] * price
                    cash += sell_val * (1 - COST_RATE)
                    trade_log.append({
                        "ticker": tickers[col],
                        "sector": h["sector"],
                        "entry_date": trading_dates[h["entry_row"]].strftime("%Y-%m-%d"),
                        "exit_date": trading_dates[row].strftime("%Y-%m-%d"),
                        "entry_price": h["entry_price"],
                        "exit_price": price,
                        "return": pnl,
                        "accel": h.get("accel", 0),
                        "ret_3m_at_entry": h.get("ret_3m", 0),
                        "ret_12m_at_entry": h.get("ret_12m", 0),
                    })
                del holdings[col]

        # --- Monthly check for new entries ---
        bars_since += 1
        if bars_since >= REBALANCE_FREQ:
            eligible_tickers = get_constituents(cmap, cmap_dates, trading_dates[row])
            eligible_cols = [
                ticker_to_idx[t] for t in eligible_tickers if t in ticker_to_idx
            ]

            candidates = score_early_momentum(
                close_mat, row, eligible_cols, tickers, sectors
            )

            if candidates:
                # Count current sector exposure
                sector_count: dict[str, int] = {}
                for col, h in holdings.items():
                    s = h["sector"]
                    sector_count[s] = sector_count.get(s, 0) + 1

                # How many slots open?
                n_slots = N_LONG - len(holdings)

                if n_slots > 0:
                    # Filter out stocks already held
                    candidates = [c for c in candidates if c[0] not in holdings]
                    selected = select_with_sector_cap(candidates, sector_count, n_slots)

                    # Compute equity for sizing
                    equity = cash
                    for col, h in holdings.items():
                        p = close_mat[row, col]
                        if not np.isnan(p):
                            equity += h["shares"] * p

                    if equity > 0 and selected:
                        per_pos = equity / N_LONG
                        for col, accel, ret_3m, ret_12m, sector in selected:
                            p = close_mat[row, col]
                            if np.isnan(p) or p <= 0:
                                continue
                            shares = per_pos / p
                            cost = per_pos * (1 + COST_RATE)
                            if cost <= cash:
                                holdings[col] = {
                                    "entry_price": p,
                                    "entry_row": row,
                                    "shares": shares,
                                    "sector": sector,
                                    "accel": accel,
                                    "ret_3m": ret_3m,
                                    "ret_12m": ret_12m,
                                }
                                cash -= cost

            bars_since = 0

        # Mark equity
        equity = cash
        for col, h in holdings.items():
            p = close_mat[row, col]
            if not np.isnan(p):
                equity += h["shares"] * p
        equity_points.append(equity)

        if (row + 1) % 1000 == 0 or row == n_rows - 1:
            n_held = len(holdings)
            unique_sectors = len(set(h["sector"] for h in holdings.values()))
            print(
                f"  [{label}] Day {row+1}/{n_rows}: equity=${equity:,.0f} "
                f"holdings={n_held} sectors={unique_sectors}",
                flush=True,
            )

    # Final exit
    for col in list(holdings.keys()):
        p = close_mat[n_rows - 1, col]
        h = holdings[col]
        if not np.isnan(p):
            pnl = p / h["entry_price"] - 1.0
            sell_val = h["shares"] * p
            cash += sell_val * (1 - COST_RATE)
            trade_log.append({
                "ticker": tickers[col],
                "sector": h["sector"],
                "entry_date": trading_dates[h["entry_row"]].strftime("%Y-%m-%d"),
                "exit_date": trading_dates[n_rows - 1].strftime("%Y-%m-%d"),
                "entry_price": h["entry_price"],
                "exit_price": p,
                "return": pnl,
                "accel": h.get("accel", 0),
                "ret_3m_at_entry": h.get("ret_3m", 0),
                "ret_12m_at_entry": h.get("ret_12m", 0),
            })

    equity_s = pd.Series(equity_points, index=pd.DatetimeIndex(trading_dates))
    return equity_s, trade_log


# -------------------------------------------------------
# Classic momentum baseline (same framework)
# -------------------------------------------------------
def run_classic_momentum(
    close_mat: np.ndarray,
    tickers: list[str],
    ticker_to_idx: dict[str, int],
    trading_dates: list[pd.Timestamp],
    cmap: dict[str, list[str]],
    cmap_dates: list[str],
) -> pd.Series:
    """Standard 12-month momentum, monthly rebalance, no sector cap."""
    holdings: dict[int, dict] = {}
    cash = INITIAL_CASH
    equity_points = []
    bars_since = REBALANCE_FREQ
    n_rows = len(trading_dates)

    for row in range(n_rows):
        bars_since += 1
        if bars_since >= REBALANCE_FREQ and row >= 252:
            eligible_tickers = get_constituents(cmap, cmap_dates, trading_dates[row])
            eligible_cols = [
                ticker_to_idx[t] for t in eligible_tickers if t in ticker_to_idx
            ]

            scores = []
            for col in eligible_cols:
                p_now = close_mat[row, col]
                p_12m = close_mat[row - 252, col]
                if np.isnan(p_now) or np.isnan(p_12m) or p_12m <= 0:
                    continue
                ret = p_now / p_12m - 1.0
                scores.append((col, ret))
            scores.sort(key=lambda x: x[1], reverse=True)

            if scores:
                new_top = set(c for c, _ in scores[:N_LONG])

                for col in list(holdings.keys()):
                    if col not in new_top:
                        p = close_mat[row, col]
                        if not np.isnan(p):
                            cash += holdings[col]["shares"] * p * (1 - COST_RATE)
                        del holdings[col]

                equity = cash
                for col, h in holdings.items():
                    p = close_mat[row, col]
                    if not np.isnan(p):
                        equity += h["shares"] * p

                n_slots = N_LONG - len(holdings)
                if n_slots > 0 and equity > 0:
                    per_pos = equity / N_LONG
                    for col, _ in scores[:N_LONG]:
                        if col not in holdings:
                            p = close_mat[row, col]
                            if not np.isnan(p) and p > 0:
                                shares = per_pos / p
                                cost = per_pos * (1 + COST_RATE)
                                if cost <= cash:
                                    holdings[col] = {"entry_price": p, "shares": shares}
                                    cash -= cost

                bars_since = 0

        equity = cash
        for col, h in holdings.items():
            p = close_mat[row, col]
            if not np.isnan(p):
                equity += h["shares"] * p
        equity_points.append(equity)

    return pd.Series(equity_points, index=pd.DatetimeIndex(trading_dates))


# -------------------------------------------------------
# SPY benchmark
# -------------------------------------------------------
def run_spy(trading_dates: list[pd.Timestamp]) -> pd.Series:
    cache_file = CACHE_DIR / "SPY_benchmark.parquet"
    if cache_file.exists():
        spy = pd.read_parquet(cache_file)
    else:
        import yfinance as yf
        spy = yf.download("SPY", start="1999-01-01", end=END, auto_adjust=True, progress=False)
        spy.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in spy.columns]
        spy.index = spy.index.tz_localize("UTC") if spy.index.tz is None else spy.index.tz_convert("UTC")
        spy.index = spy.index.normalize()
        spy.to_parquet(cache_file)

    start_ts = pd.Timestamp(START, tz="UTC")
    end_ts = pd.Timestamp(END, tz="UTC")
    spy = spy.loc[(spy.index >= start_ts) & (spy.index <= end_ts)]
    if spy.empty:
        return pd.Series(dtype=float)
    initial_price = float(spy.iloc[0]["close"])
    shares = INITIAL_CASH / initial_price
    equity = spy["close"].astype(float) * shares
    equity = equity.reindex(pd.DatetimeIndex(trading_dates)).ffill()
    return equity


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main() -> None:
    print("=" * 80, flush=True)
    print("EARLY-BREAKOUT MOMENTUM vs CLASSIC MOMENTUM vs SPY", flush=True)
    print(f"$100K | {START} to {END} | Top-{N_LONG} | Max {MAX_PER_SECTOR}/sector | "
          f"{HOLD_DAYS}-day hold", flush=True)
    print("=" * 80, flush=True)

    print("\nLoading data...", flush=True)
    prices_dict = load_prices()
    print(f"  {len(prices_dict)} tickers", flush=True)

    sectors = load_sectors()
    print(f"  {len(sectors)} tickers with sector data", flush=True)

    cmap = load_constituents()
    cmap_dates = sorted(cmap.keys())

    start_ts = pd.Timestamp(START, tz="UTC")
    end_ts = pd.Timestamp(END, tz="UTC")
    all_dates: set[pd.Timestamp] = set()
    for df in prices_dict.values():
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        all_dates.update(df.index[mask])
    trading_dates = sorted(all_dates)
    print(f"  {len(trading_dates)} trading days", flush=True)

    print("Building price matrix...", flush=True)
    close_mat, tickers, ticker_to_idx = build_close_matrix(prices_dict, trading_dates)
    print(f"  Matrix: {close_mat.shape}", flush=True)
    del prices_dict

    # --- Run strategies ---
    print("\n--- Early Breakout Momentum ---", flush=True)
    eq_early, trade_log = run_early_momentum(
        close_mat, tickers, ticker_to_idx, trading_dates,
        cmap, cmap_dates, sectors, "Early Mom",
    )

    print("\n--- Classic 12-Month Momentum ---", flush=True)
    eq_classic = run_classic_momentum(
        close_mat, tickers, ticker_to_idx, trading_dates,
        cmap, cmap_dates,
    )

    print("\n--- SPY Buy & Hold ---", flush=True)
    eq_spy = run_spy(trading_dates)

    # --- Results ---
    print("\n" + "=" * 80, flush=True)
    print("RESULTS", flush=True)
    print("=" * 80, flush=True)

    for name, eq in [
        ("Early Breakout Mom", eq_early),
        ("Classic 12m Mom", eq_classic),
        ("SPY B&H", eq_spy),
    ]:
        if eq.empty:
            continue
        m = compute_metrics(eq, trades=[], periods_per_year=252)
        print(f"\n  {name}:", flush=True)
        print(f"    ${INITIAL_CASH:,.0f} -> ${m.final_equity:,.0f}", flush=True)
        print(f"    CAGR: {m.cagr:.2%}  Sharpe: {m.sharpe:.3f}  Max DD: {m.max_drawdown:.2%}", flush=True)

    # --- Trade analysis for early momentum ---
    if trade_log:
        tdf = pd.DataFrame(trade_log)
        print(f"\n\nEARLY MOMENTUM TRADE ANALYSIS ({len(tdf)} trades):", flush=True)
        print("-" * 60, flush=True)

        wins = tdf[tdf["return"] > 0]
        losses = tdf[tdf["return"] <= 0]
        print(f"  Win rate: {len(wins)}/{len(tdf)} = {len(wins)/len(tdf):.1%}", flush=True)
        print(f"  Avg winner: {wins['return'].mean():+.1%}", flush=True)
        print(f"  Avg loser: {losses['return'].mean():+.1%}", flush=True)
        print(f"  Avg return: {tdf['return'].mean():+.1%}", flush=True)
        print(f"  Median return: {tdf['return'].median():+.1%}", flush=True)

        # Sector distribution
        print(f"\n  Sector distribution:", flush=True)
        for sector, grp in tdf.groupby("sector"):
            wr = (grp["return"] > 0).mean()
            avg = grp["return"].mean()
            print(f"    {sector:30s} n={len(grp):4d}  win={wr:.0%}  avg={avg:+.1%}", flush=True)

        # Year-by-year
        print(f"\n  Annual win rate:", flush=True)
        tdf["year"] = tdf["exit_date"].str[:4]
        for year, grp in tdf.groupby("year"):
            wr = (grp["return"] > 0).mean()
            avg = grp["return"].mean()
            print(f"    {year}: win={wr:.0%}  avg={avg:+.1%}  trades={len(grp)}", flush=True)

        # Entry momentum profile
        print(f"\n  Entry profile (avg):", flush=True)
        print(f"    3m momentum at entry: {tdf['ret_3m_at_entry'].mean():+.1%}", flush=True)
        print(f"    12m momentum at entry: {tdf['ret_12m_at_entry'].mean():+.1%}", flush=True)
        print(f"    Acceleration at entry: {tdf['accel'].mean():+.1%}", flush=True)

        # Best/worst
        print(f"\n  Best 10 trades:", flush=True)
        for _, r in tdf.nlargest(10, "return").iterrows():
            print(f"    {r['ticker']:8s} {r['sector']:25s} {r['entry_date']}->{r['exit_date']} "
                  f"ret={r['return']:+.1%}  3m@entry={r['ret_3m_at_entry']:+.1%}", flush=True)

        print(f"\n  Worst 10 trades:", flush=True)
        for _, r in tdf.nsmallest(10, "return").iterrows():
            print(f"    {r['ticker']:8s} {r['sector']:25s} {r['entry_date']}->{r['exit_date']} "
                  f"ret={r['return']:+.1%}  3m@entry={r['ret_3m_at_entry']:+.1%}", flush=True)

        tdf.to_csv("/tmp/early_momentum_trades.csv", index=False)
        print(f"\n  Full trade log: /tmp/early_momentum_trades.csv", flush=True)

    # Era breakdown
    print(f"\n\nERA COMPARISON:", flush=True)
    print("-" * 80, flush=True)
    eras = {
        "Dot-com (2000-02)": ("2000-01-01", "2002-12-31"),
        "Recovery (2003-06)": ("2003-01-01", "2006-12-31"),
        "Crisis (2007-09)": ("2007-01-01", "2009-12-31"),
        "Bull (2010-19)": ("2010-01-01", "2019-12-31"),
        "COVID+ (2020-22)": ("2020-01-01", "2022-12-31"),
        "Recent (2023-26)": ("2023-01-01", "2026-12-31"),
    }
    for era_name, (es, ee) in eras.items():
        s = pd.Timestamp(es, tz="UTC")
        e = pd.Timestamp(ee, tz="UTC")
        row = f"  {era_name:22s}"
        for sname, eq in [("Early", eq_early), ("Classic", eq_classic), ("SPY", eq_spy)]:
            sub = eq[(eq.index >= s) & (eq.index <= e)]
            if len(sub) > 50:
                m = compute_metrics(sub, trades=[], periods_per_year=252)
                row += f"  {sname} Sh={m.sharpe:.2f} CAGR={m.cagr:+.1%}"
            else:
                row += f"  {sname} N/A"
        print(row, flush=True)


if __name__ == "__main__":
    main()
