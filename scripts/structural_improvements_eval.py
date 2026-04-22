"""Structural improvements evaluation — full-universe & staggered entry.

Experiments:
  A: Baseline (canonical S&P 500 constituent filter)
  B: Full-universe (all 782 tickers, no constituent filter)
  C: Full-universe + sector cap relaxed to 3/sector
  D: Staggered entry (5/week over 3 weeks) with S&P constituent filter
  E: Full-universe + staggered entry

For each: full-period, train (2000-2018), holdout (2019-2026) Sharpe/CAGR/MDD.
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.backtest.bias_free import (
    BacktestResult,
    BiasFreData,
    _get_constituents,
    default_acceleration_score,
    load_bias_free_data,
    run_bias_free_backtest,
)
from signals.backtest.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Modified backtest: full-universe (skip constituent filter)
# ---------------------------------------------------------------------------
def run_full_universe_backtest(
    data: BiasFreData,
    score_fn=None,
    short: int = 63,
    long: int = 252,
    hold_days: int = 105,
    n_long: int = 15,
    max_per_sector: int = 2,
    min_short_return: float = 0.10,
    max_long_return: float = 1.50,
    rebalance_freq: int = 21,
    initial_cash: float = 100_000.0,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """Like run_bias_free_backtest but uses ALL tickers with valid prices."""

    if score_fn is None:
        def score_fn(cm, r, c, s=short, lg=long):
            return default_acceleration_score(cm, r, c, s, lg, min_short_return, max_long_return)

    mat = data.close_mat
    cost_rate = cost_bps * 1e-4
    n_dates = len(data.trading_dates)

    holdings: dict[int, dict] = {}
    cash = initial_cash
    equity_points: list[float] = []
    trade_returns: list[float] = []
    bars_since_rebal = rebalance_freq

    for row in range(n_dates):
        # === STEP 1: Fixed-hold exits ===
        for col in list(holdings):
            if (row - holdings[col]["entry_row"]) >= hold_days:
                p = mat[row, col]
                if not np.isnan(p):
                    pnl = p / holdings[col]["ep"] - 1.0
                    cash += holdings[col]["sh"] * p * (1 - cost_rate)
                    trade_returns.append(pnl)
                del holdings[col]

        # === STEP 2: Deploy idle cash ===
        if holdings and cash > 100:
            n_held = len(holdings)
            per = cash / n_held
            for col in holdings:
                p = mat[row, col]
                if not np.isnan(p) and p > 0:
                    holdings[col]["sh"] += per / p
                    cash -= per

        # === STEP 3: New entries (monthly) ===
        bars_since_rebal += 1
        if bars_since_rebal >= rebalance_freq and row >= long:
            # KEY DIFFERENCE: use ALL tickers with valid price at this row
            eligible_cols = []
            for col_idx in range(mat.shape[1]):
                if not np.isnan(mat[row, col_idx]):
                    eligible_cols.append(col_idx)

            candidates: list[tuple[int, float, str]] = []
            for col in eligible_cols:
                if col in holdings:
                    continue
                score = score_fn(mat, row, col, short, long)
                if score is None:
                    continue
                ticker = data.tickers[col]
                sector = data.sectors.get(ticker, "Unknown")
                candidates.append((col, score, sector))

            candidates.sort(key=lambda x: x[1], reverse=True)

            n_slots = n_long - len(holdings)
            if n_slots > 0 and candidates:
                sector_count: dict[str, int] = {}
                for h in holdings.values():
                    s = h["sec"]
                    sector_count[s] = sector_count.get(s, 0) + 1

                selected: list[tuple[int, str]] = []
                for col, _score, sector in candidates:
                    if len(selected) >= n_slots:
                        break
                    if sector_count.get(sector, 0) >= max_per_sector:
                        continue
                    selected.append((col, sector))
                    sector_count[sector] = sector_count.get(sector, 0) + 1

                equity = cash
                for col, h in holdings.items():
                    p = mat[row, col]
                    if not np.isnan(p):
                        equity += h["sh"] * p

                if selected and equity > 0:
                    target_n = max(len(holdings) + len(selected), n_long)
                    per_pos = equity / target_n

                    for col, sector in selected:
                        p = mat[row, col]
                        if np.isnan(p) or p <= 0:
                            continue
                        cost = per_pos * (1 + cost_rate)
                        if cost <= cash:
                            holdings[col] = {
                                "ep": p,
                                "sh": per_pos / p,
                                "entry_row": row,
                                "sec": sector,
                            }
                            cash -= cost

                    if holdings and cash > 100:
                        n_held = len(holdings)
                        per = cash / n_held
                        for col in holdings:
                            p = mat[row, col]
                            if not np.isnan(p) and p > 0:
                                holdings[col]["sh"] += per / p
                                cash -= per

            bars_since_rebal = 0

        # === STEP 4: Mark equity ===
        equity = cash
        for col, h in holdings.items():
            p = mat[row, col]
            if not np.isnan(p):
                equity += h["sh"] * p
        equity_points.append(equity)

    # Final liquidation
    for col in list(holdings):
        p = mat[n_dates - 1, col]
        if not np.isnan(p):
            pnl = p / holdings[col]["ep"] - 1.0
            trade_returns.append(pnl)

    equity_s = pd.Series(
        equity_points,
        index=pd.DatetimeIndex(data.trading_dates[:len(equity_points)]),
    )
    m = compute_metrics(equity_s, trades=[], periods_per_year=252)
    tr = np.array(trade_returns)
    win_rate = float((tr > 0).mean()) if len(tr) > 0 else 0.0

    return BacktestResult(
        sharpe=m.sharpe,
        cagr=m.cagr,
        max_drawdown=m.max_drawdown,
        final_equity=m.final_equity,
        win_rate=win_rate,
        n_trades=len(tr),
        equity_series=equity_s,
        trade_returns=trade_returns,
    )


# ---------------------------------------------------------------------------
# Modified backtest: staggered entry (5 per week over 3 weeks)
# ---------------------------------------------------------------------------
def run_staggered_entry_backtest(
    data: BiasFreData,
    score_fn=None,
    short: int = 63,
    long: int = 252,
    hold_days: int = 105,
    n_long: int = 15,
    max_per_sector: int = 2,
    min_short_return: float = 0.10,
    max_long_return: float = 1.50,
    rebalance_freq: int = 21,
    initial_cash: float = 100_000.0,
    cost_bps: float = 10.0,
    entries_per_week: int = 5,
    use_full_universe: bool = False,
) -> BacktestResult:
    """Backtest with staggered entry: spread new entries over ~3 weeks.

    On rebalance day, score and rank candidates, then enter up to
    `entries_per_week` per week (1/day for 5 days), repeating until all
    slots are filled or candidates exhausted (~15 trading days for 15 slots).
    """

    if score_fn is None:
        def score_fn(cm, r, c, s=short, lg=long):
            return default_acceleration_score(cm, r, c, s, lg, min_short_return, max_long_return)

    mat = data.close_mat
    cost_rate = cost_bps * 1e-4
    n_dates = len(data.trading_dates)

    holdings: dict[int, dict] = {}
    cash = initial_cash
    equity_points: list[float] = []
    trade_returns: list[float] = []
    bars_since_rebal = rebalance_freq

    # Pending entries: list of (col, sector) to enter, 1 per day
    pending: list[tuple[int, str]] = []

    for row in range(n_dates):
        # === STEP 1: Fixed-hold exits ===
        for col in list(holdings):
            if (row - holdings[col]["entry_row"]) >= hold_days:
                p = mat[row, col]
                if not np.isnan(p):
                    pnl = p / holdings[col]["ep"] - 1.0
                    cash += holdings[col]["sh"] * p * (1 - cost_rate)
                    trade_returns.append(pnl)
                del holdings[col]

        # === STEP 2: Deploy idle cash (only if no pending entries) ===
        if holdings and cash > 100 and not pending:
            n_held = len(holdings)
            per = cash / n_held
            for col in holdings:
                p = mat[row, col]
                if not np.isnan(p) and p > 0:
                    holdings[col]["sh"] += per / p
                    cash -= per

        # === STEP 3: Process pending entries (1 per day) ===
        if pending and len(holdings) < n_long:
            # Enter 1 position per trading day (5 per week)
            col, sector = pending.pop(0)
            # Skip if already held or price is invalid
            if col not in holdings:
                p = mat[row, col]
                if not (np.isnan(p) or p <= 0):
                    equity = cash
                    for c, h in holdings.items():
                        pp = mat[row, c]
                        if not np.isnan(pp):
                            equity += h["sh"] * pp

                    if equity > 0:
                        target_n = max(len(holdings) + 1, n_long)
                        per_pos = equity / target_n
                        cost = per_pos * (1 + cost_rate)
                        if cost <= cash:
                            holdings[col] = {
                                "ep": p,
                                "sh": per_pos / p,
                                "entry_row": row,
                                "sec": sector,
                            }
                            cash -= cost

        # === STEP 4: Check for new candidates (monthly) ===
        bars_since_rebal += 1
        if bars_since_rebal >= rebalance_freq and row >= long:
            if use_full_universe:
                eligible_cols = [
                    col_idx for col_idx in range(mat.shape[1])
                    if not np.isnan(mat[row, col_idx])
                ]
            else:
                eligible_tickers = _get_constituents(data, data.trading_dates[row])
                eligible_cols = [
                    data.ticker_to_idx[t]
                    for t in eligible_tickers
                    if t in data.ticker_to_idx
                ]

            candidates: list[tuple[int, float, str]] = []
            for col in eligible_cols:
                if col in holdings:
                    continue
                score = score_fn(mat, row, col, short, long)
                if score is None:
                    continue
                ticker = data.tickers[col]
                sector = data.sectors.get(ticker, "Unknown")
                candidates.append((col, score, sector))

            candidates.sort(key=lambda x: x[1], reverse=True)

            n_slots = n_long - len(holdings)
            if n_slots > 0 and candidates:
                sector_count: dict[str, int] = {}
                for h in holdings.values():
                    s = h["sec"]
                    sector_count[s] = sector_count.get(s, 0) + 1

                selected: list[tuple[int, str]] = []
                for col, _score, sector in candidates:
                    if len(selected) >= n_slots:
                        break
                    if sector_count.get(sector, 0) >= max_per_sector:
                        continue
                    selected.append((col, sector))
                    sector_count[sector] = sector_count.get(sector, 0) + 1

                # Queue selected for staggered entry
                pending = selected  # replace any leftover pending

            bars_since_rebal = 0

        # === STEP 5: Mark equity ===
        equity = cash
        for col, h in holdings.items():
            p = mat[row, col]
            if not np.isnan(p):
                equity += h["sh"] * p
        equity_points.append(equity)

    # Final liquidation
    for col in list(holdings):
        p = mat[n_dates - 1, col]
        if not np.isnan(p):
            pnl = p / holdings[col]["ep"] - 1.0
            trade_returns.append(pnl)

    equity_s = pd.Series(
        equity_points,
        index=pd.DatetimeIndex(data.trading_dates[:len(equity_points)]),
    )
    m = compute_metrics(equity_s, trades=[], periods_per_year=252)
    tr = np.array(trade_returns)
    win_rate = float((tr > 0).mean()) if len(tr) > 0 else 0.0

    return BacktestResult(
        sharpe=m.sharpe,
        cagr=m.cagr,
        max_drawdown=m.max_drawdown,
        final_equity=m.final_equity,
        win_rate=win_rate,
        n_trades=len(tr),
        equity_series=equity_s,
        trade_returns=trade_returns,
    )


# ---------------------------------------------------------------------------
# Period splitting helper
# ---------------------------------------------------------------------------
def split_result(result: BacktestResult, split_date: str = "2019-01-01"):
    """Split equity series into train/holdout and compute metrics for each."""
    eq = result.equity_series
    split_ts = pd.Timestamp(split_date, tz="UTC")

    train_eq = eq[eq.index < split_ts]
    holdout_eq = eq[eq.index >= split_ts]

    def metrics_from_eq(es: pd.Series):
        if len(es) < 10:
            return {"sharpe": float("nan"), "cagr": float("nan"), "mdd": float("nan")}
        m = compute_metrics(es, trades=[], periods_per_year=252)
        return {"sharpe": m.sharpe, "cagr": m.cagr, "mdd": m.max_drawdown}

    return {
        "full": {"sharpe": result.sharpe, "cagr": result.cagr, "mdd": result.max_drawdown},
        "train": metrics_from_eq(train_eq),
        "holdout": metrics_from_eq(holdout_eq),
        "n_trades": result.n_trades,
        "win_rate": result.win_rate,
        "final_equity": result.final_equity,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("STRUCTURAL IMPROVEMENTS EVALUATION")
    print("=" * 75)

    t0 = time.time()
    print("\nLoading data...")
    data = load_bias_free_data()
    print(f"  {len(data.tickers)} tickers, {len(data.trading_dates)} dates")
    print(f"  {data.trading_dates[0].date()} -> {data.trading_dates[-1].date()}")

    results: dict[str, dict] = {}

    # --- A: Baseline (canonical) ---
    print("\n[A] Baseline (canonical S&P constituent filter)...")
    ta = time.time()
    r_a = run_bias_free_backtest(data)
    results["A: Baseline"] = split_result(r_a)
    print(f"    done in {time.time()-ta:.0f}s — Sharpe={r_a.sharpe:.3f}, CAGR={r_a.cagr:.1%}")

    # --- B: Full-universe (all 782 tickers) ---
    print("\n[B] Full-universe (all tickers, no constituent filter)...")
    tb = time.time()
    r_b = run_full_universe_backtest(data)
    results["B: Full-universe"] = split_result(r_b)
    print(f"    done in {time.time()-tb:.0f}s — Sharpe={r_b.sharpe:.3f}, CAGR={r_b.cagr:.1%}")

    # --- C: Full-universe + relaxed sector cap (3/sector) ---
    print("\n[C] Full-universe + sector cap 3...")
    tc = time.time()
    r_c = run_full_universe_backtest(data, max_per_sector=3)
    results["C: Full+sec3"] = split_result(r_c)
    print(f"    done in {time.time()-tc:.0f}s — Sharpe={r_c.sharpe:.3f}, CAGR={r_c.cagr:.1%}")

    # --- D: Staggered entry (S&P constituent filter) ---
    print("\n[D] Staggered entry (5/week, S&P filter)...")
    td = time.time()
    r_d = run_staggered_entry_backtest(data, use_full_universe=False)
    results["D: Stagger+SP"] = split_result(r_d)
    print(f"    done in {time.time()-td:.0f}s — Sharpe={r_d.sharpe:.3f}, CAGR={r_d.cagr:.1%}")

    # --- E: Full-universe + staggered entry ---
    print("\n[E] Full-universe + staggered entry...")
    te = time.time()
    r_e = run_staggered_entry_backtest(data, use_full_universe=True)
    results["E: Full+stagger"] = split_result(r_e)
    print(f"    done in {time.time()-te:.0f}s — Sharpe={r_e.sharpe:.3f}, CAGR={r_e.cagr:.1%}")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    print("\n" + "=" * 95)
    print("RESULTS SUMMARY")
    print("=" * 95)
    print(f"\n  {'Experiment':<22} {'Period':<9} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Trades':>7} {'WinRate':>8}")
    print("  " + "-" * 75)

    for label, r in results.items():
        for period in ["full", "train", "holdout"]:
            m = r[period]
            trades_str = f"{r['n_trades']:>7d}" if period == "full" else f"{'':>7}"
            wr_str = f"{r['win_rate']:>7.1%}" if period == "full" else f"{'':>7}"
            print(
                f"  {label:<22} {period:<9} {m['sharpe']:>7.3f} "
                f"{m['cagr']:>6.1%} {m['mdd']:>6.1%} {trades_str} {wr_str}"
            )
        print()

    # Delta table vs baseline
    base = results["A: Baseline"]
    print(f"\n  {'Experiment':<22} {'Period':<9} {'dSharpe':>8} {'dCAGR':>8} {'dMDD':>8}")
    print("  " + "-" * 55)
    for label, r in results.items():
        if label == "A: Baseline":
            continue
        for period in ["full", "train", "holdout"]:
            m = r[period]
            b = base[period]
            ds = m["sharpe"] - b["sharpe"]
            dc = m["cagr"] - b["cagr"]
            dm = m["mdd"] - b["mdd"]
            print(
                f"  {label:<22} {period:<9} {ds:>+7.3f} "
                f"{dc:>+7.1%} {dm:>+7.1%}"
            )
        print()

    print("=" * 95)
    print("INTERPRETATION GUIDE")
    print("=" * 95)
    print("  - dSharpe > 0 is better; dMDD closer to 0 (less negative) is better")
    print("  - Holdout is the true test: train results may overfit the 2000-2018 period")
    print("  - Full-universe removes survivorship-bias protection — if it helps,")
    print("    the gain is real alpha from near-S&P stocks, not survivorship bias")
    print("  - Staggered entry should reduce MDD with minimal Sharpe impact")


if __name__ == "__main__":
    main()
