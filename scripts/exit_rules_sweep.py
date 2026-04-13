"""Sweep exit rules (profit targets, stop losses, trailing stops) on bias-free momentum.

Optimized: pre-builds a unified price matrix to avoid per-ticker dict lookups
in the inner loop. Flushes output after each rule.

Usage:
    python3 -u scripts/exit_rules_sweep.py
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
START = "2000-01-01"
END = "2026-04-11"
INITIAL_CASH = 100_000.0
COST_RATE = 10e-4  # 5+5 bps
LOOKBACK = 252
SKIP = 21
N_LONG = 10
REBALANCE_FREQ = 21

LEGIT_HIGH = {
    "AAPL", "AMZN", "NVDA", "NFLX", "MNST", "DECK", "AXON", "SBAC",
    "LRCX", "FIX", "BRK.B", "GOOG", "GOOGL", "MSFT", "META", "TSLA",
    "AVGO", "COST", "UNH", "LLY", "NVO", "NVR", "SEB", "BKNG", "AZO",
    "CMG", "ORLY", "MTD", "MELI",
}

# -------------------------------------------------------
# Load data once
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


def get_constituents(cmap: dict, cmap_dates: list[str], date: pd.Timestamp) -> list[str]:
    d_str = date.strftime("%Y-%m-%d")
    # Binary search for nearest prior date
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


# -------------------------------------------------------
# Pre-build price matrix for fast lookups
# -------------------------------------------------------
def build_close_matrix(
    prices_dict: dict[str, pd.DataFrame], trading_dates: list[pd.Timestamp]
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Build (n_dates x n_tickers) close-price matrix. NaN where no data."""
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
# Momentum scorer using matrix
# -------------------------------------------------------
def score_momentum_matrix(
    close_mat: np.ndarray,
    row_idx: int,
    eligible_cols: list[int],
    lookback: int = LOOKBACK,
    skip: int = SKIP,
) -> list[tuple[int, float]]:
    """Return [(col_idx, momentum_return)] for eligible tickers, sorted desc."""
    needed = lookback + skip
    if row_idx < needed:
        return []

    scores = []
    for col in eligible_cols:
        end_price = close_mat[row_idx, col]
        start_row = row_idx - lookback
        if start_row < 0:
            continue
        start_price = close_mat[start_row, col]
        if np.isnan(end_price) or np.isnan(start_price) or start_price <= 0:
            continue
        # Check we have enough data in between (80% fill)
        segment = close_mat[start_row:row_idx + 1, col]
        if np.count_nonzero(~np.isnan(segment)) < lookback * 0.8:
            continue
        ret = end_price / start_price - 1.0
        scores.append((col, ret))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# -------------------------------------------------------
# Backtest with exit rules
# -------------------------------------------------------
def run_with_rules(
    close_mat: np.ndarray,
    tickers: list[str],
    ticker_to_idx: dict[str, int],
    trading_dates: list[pd.Timestamp],
    cmap: dict[str, list[str]],
    cmap_dates: list[str],
    profit_target: float | None = None,
    stop_loss: float | None = None,
    trailing_stop: float | None = None,
) -> dict:
    # holdings: col_idx -> {entry_price, shares, peak_price}
    holdings: dict[int, dict] = {}
    cash = INITIAL_CASH
    equity_points = []
    bars_since = REBALANCE_FREQ
    n_profit = n_stop = n_trail = 0
    trade_returns = []

    n_rows = len(trading_dates)

    for row in range(n_rows):
        date = trading_dates[row]

        # --- Daily exit checks ---
        for col in list(holdings.keys()):
            price = close_mat[row, col]
            if np.isnan(price):
                continue
            h = holdings[col]
            pnl = price / h["entry_price"] - 1.0

            if price > h["peak_price"]:
                h["peak_price"] = price

            exit_it = False
            if profit_target is not None and pnl >= profit_target:
                exit_it = True
                n_profit += 1
            elif stop_loss is not None and pnl <= stop_loss:
                exit_it = True
                n_stop += 1
            elif trailing_stop is not None:
                dd = 1.0 - price / h["peak_price"]
                if dd >= trailing_stop:
                    exit_it = True
                    n_trail += 1

            if exit_it:
                sell_val = h["shares"] * price
                cash += sell_val * (1 - COST_RATE)
                trade_returns.append(pnl)
                del holdings[col]

        # --- Monthly rebalance ---
        bars_since += 1
        if bars_since >= REBALANCE_FREQ:
            eligible_tickers = get_constituents(cmap, cmap_dates, date)
            eligible_cols = [
                ticker_to_idx[t]
                for t in eligible_tickers
                if t in ticker_to_idx
            ]

            scores = score_momentum_matrix(close_mat, row, eligible_cols)
            if scores:
                new_top = set(c for c, _ in scores[:N_LONG])

                # Exit positions not in new top
                for col in list(holdings.keys()):
                    if col not in new_top:
                        price = close_mat[row, col]
                        if not np.isnan(price):
                            h = holdings[col]
                            pnl = price / h["entry_price"] - 1.0
                            sell_val = h["shares"] * price
                            cash += sell_val * (1 - COST_RATE)
                            trade_returns.append(pnl)
                        del holdings[col]

                # Equity
                equity = cash
                for col, h in holdings.items():
                    p = close_mat[row, col]
                    if not np.isnan(p):
                        equity += h["shares"] * p

                # Enter new
                n_slots = N_LONG - len(holdings)
                if n_slots > 0 and equity > 0:
                    per_pos = equity / N_LONG
                    for col, _ in scores[:N_LONG]:
                        if col not in holdings:
                            p = close_mat[row, col]
                            if np.isnan(p) or p <= 0:
                                continue
                            shares = per_pos / p
                            cost = per_pos * (1 + COST_RATE)
                            if cost <= cash:
                                holdings[col] = {
                                    "entry_price": p,
                                    "shares": shares,
                                    "peak_price": p,
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

    # Metrics
    equity_s = pd.Series(equity_points, index=pd.DatetimeIndex(trading_dates))
    m = compute_metrics(equity_s, trades=[], periods_per_year=252)
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])

    return {
        "sharpe": m.sharpe,
        "cagr": m.cagr,
        "max_dd": m.max_drawdown,
        "final": m.final_equity,
        "win_rate": float((tr > 0).mean()),
        "avg_ret": float(tr.mean()),
        "median_ret": float(np.median(tr)),
        "n_trades": len(tr),
        "n_profit_exits": n_profit,
        "n_stop_exits": n_stop,
        "n_trail_exits": n_trail,
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main() -> None:
    print("Loading data...", flush=True)
    prices_dict = load_prices()
    print(f"  {len(prices_dict)} tickers", flush=True)

    cmap = load_constituents()
    cmap_dates = sorted(cmap.keys())

    # Trading calendar
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

    # Free memory
    del prices_dict

    rules = [
        ("No rules (baseline)", None, None, None),
        # Profit targets
        ("Take profit @ 20%", 0.20, None, None),
        ("Take profit @ 30%", 0.30, None, None),
        ("Take profit @ 50%", 0.50, None, None),
        ("Take profit @ 75%", 0.75, None, None),
        ("Take profit @ 100%", 1.00, None, None),
        # Stop losses
        ("Stop loss @ -10%", None, -0.10, None),
        ("Stop loss @ -15%", None, -0.15, None),
        ("Stop loss @ -20%", None, -0.20, None),
        ("Stop loss @ -30%", None, -0.30, None),
        # Trailing stops
        ("Trail stop 10%", None, None, 0.10),
        ("Trail stop 15%", None, None, 0.15),
        ("Trail stop 20%", None, None, 0.20),
        ("Trail stop 25%", None, None, 0.25),
        # Combos
        ("TP30% + SL-10%", 0.30, -0.10, None),
        ("TP50% + SL-15%", 0.50, -0.15, None),
        ("TP50% + SL-20%", 0.50, -0.20, None),
        ("TP75% + SL-15%", 0.75, -0.15, None),
        ("TP50% + Trail15%", 0.50, None, 0.15),
        ("TP50% + Trail20%", 0.50, None, 0.20),
        ("TP75% + Trail20%", 0.75, None, 0.20),
        ("SL-15% + Trail15%", None, -0.15, 0.15),
        ("SL-15% + Trail20%", None, -0.15, 0.20),
        ("TP50% + SL-15% + Trail15%", 0.50, -0.15, 0.15),
        ("TP50% + SL-15% + Trail20%", 0.50, -0.15, 0.20),
        ("TP75% + SL-20% + Trail25%", 0.75, -0.20, 0.25),
    ]

    print(f"\nTesting {len(rules)} exit rules...\n", flush=True)
    results = []

    for i, (label, pt, sl, ts) in enumerate(rules):
        r = run_with_rules(
            close_mat, tickers, ticker_to_idx,
            trading_dates, cmap, cmap_dates,
            profit_target=pt, stop_loss=sl, trailing_stop=ts,
        )
        results.append({"label": label, **r})
        star = " ***" if r["sharpe"] > 0.387 else ""
        print(
            f"[{i+1:2d}/{len(rules)}] {label:35s} "
            f"Sharpe={r['sharpe']:.3f}  CAGR={r['cagr']:+.1%}  "
            f"MaxDD={r['max_dd']:+.1%}  Win={r['win_rate']:.0%}  "
            f"Final=${r['final']:>12,.0f}  "
            f"Trades={r['n_trades']}  "
            f"PX={r['n_profit_exits']} SX={r['n_stop_exits']} TX={r['n_trail_exits']}"
            f"{star}",
            flush=True,
        )

    # Ranked
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print("\n" + "=" * 100, flush=True)
    print("RANKED BY SHARPE", flush=True)
    print("=" * 100, flush=True)
    for j, r in enumerate(results):
        flag = " <-- BEST" if j == 0 else ""
        print(
            f"  {r['label']:35s} Sharpe={r['sharpe']:.3f}  "
            f"CAGR={r['cagr']:+.1%}  MaxDD={r['max_dd']:+.1%}  "
            f"${r['final']:>12,.0f}{flag}",
            flush=True,
        )

    # Save
    df = pd.DataFrame(results)
    df.to_csv("/tmp/exit_rules_sweep.csv", index=False)
    print("\nSaved to /tmp/exit_rules_sweep.csv", flush=True)


if __name__ == "__main__":
    main()
