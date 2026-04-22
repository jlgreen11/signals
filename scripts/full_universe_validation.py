"""Validate the full-universe momentum improvement.

The structural_improvements_eval.py found that removing the S&P 500 constituent
constraint lifts Sharpe from 0.659 to 0.913. This script validates whether
that improvement is genuine or survivorship bias.

Tests:
  A: Baseline (S&P constituent filter)
  B: Full universe (all 782 tickers)
  C: Full universe EXCLUDING dead tickers (remove the 36 delisted stocks)
  D: Full universe with PENALTY for dead stocks (if held when delisted, take -100% loss)
  E: Full universe on holdout only (2019-2026) — most relevant for forward-looking

If B and C have similar performance, the improvement isn't from dead stocks.
If D still beats A, the improvement survives even worst-case delisting losses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.backtest.bias_free import (
    BacktestResult,
    BiasFreData,
    default_acceleration_score,
    load_bias_free_data,
    run_bias_free_backtest,
    clear_cache,
    _get_constituents,
)
from signals.backtest.metrics import compute_metrics


def run_full_universe_backtest(
    data: BiasFreData,
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
    exclude_tickers: set | None = None,
    delisting_penalty: bool = False,
) -> BacktestResult:
    """Full-universe backtest: all tickers with valid prices, not just S&P."""
    mat = data.close_mat
    cost_rate = cost_bps * 1e-4
    n_dates = len(data.trading_dates)

    exclude_cols = set()
    if exclude_tickers:
        for t in exclude_tickers:
            if t in data.ticker_to_idx:
                exclude_cols.add(data.ticker_to_idx[t])

    holdings: dict[int, dict] = {}
    cash = initial_cash
    equity_points: list[float] = []
    trade_returns: list[float] = []
    bars_since_rebal = rebalance_freq

    def score_fn(cm, r, c, s=short, lg=long):
        return default_acceleration_score(cm, r, c, s, lg, min_short_return, max_long_return)

    for row in range(n_dates):
        # Check for delisted stocks in holdings
        if delisting_penalty:
            for col in list(holdings):
                if row > 0 and np.isnan(mat[row, col]) and not np.isnan(mat[row - 1, col]):
                    # Stock just went NaN — delisted. Take full loss.
                    trade_returns.append(-1.0)
                    del holdings[col]

        # Fixed-hold exits
        for col in list(holdings):
            if (row - holdings[col]["entry_row"]) >= hold_days:
                p = mat[row, col]
                if not np.isnan(p):
                    pnl = p / holdings[col]["ep"] - 1.0
                    cash += holdings[col]["sh"] * p * (1 - cost_rate)
                    trade_returns.append(pnl)
                del holdings[col]

        # Deploy idle cash
        if holdings and cash > 100:
            per = cash / len(holdings)
            for col in holdings:
                p = mat[row, col]
                if not np.isnan(p) and p > 0:
                    holdings[col]["sh"] += per / p
                    cash -= per

        # Rebalance
        bars_since_rebal += 1
        if bars_since_rebal >= rebalance_freq and row >= long:
            # ALL tickers with valid prices (not just S&P constituents)
            eligible_cols = [
                i for i in range(len(data.tickers))
                if not np.isnan(mat[row, i]) and i not in exclude_cols
            ]

            candidates = []
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
                    sector_count[h["sec"]] = sector_count.get(h["sec"], 0) + 1

                selected = []
                for col, _, sector in candidates:
                    if len(selected) >= n_slots:
                        break
                    if sector_count.get(sector, 0) >= max_per_sector:
                        continue
                    selected.append((col, sector))
                    sector_count[sector] = sector_count.get(sector, 0) + 1

                equity = cash + sum(
                    h["sh"] * mat[row, c] for c, h in holdings.items()
                    if not np.isnan(mat[row, c])
                )
                if selected and equity > 0:
                    target_n = max(len(holdings) + len(selected), n_long)
                    per_pos = equity / target_n
                    for col, sector in selected:
                        p = mat[row, col]
                        if np.isnan(p) or p <= 0:
                            continue
                        cost = per_pos * (1 + cost_rate)
                        if cost <= cash:
                            holdings[col] = {"ep": p, "sh": per_pos / p, "entry_row": row, "sec": sector}
                            cash -= cost

                    if holdings and cash > 100:
                        per = cash / len(holdings)
                        for col in holdings:
                            p = mat[row, col]
                            if not np.isnan(p) and p > 0:
                                holdings[col]["sh"] += per / p
                                cash -= per

            bars_since_rebal = 0

        # Mark equity
        equity = cash + sum(
            h["sh"] * mat[row, c] for c, h in holdings.items()
            if not np.isnan(mat[row, c])
        )
        equity_points.append(equity)

    for col in list(holdings):
        p = mat[n_dates - 1, col]
        if not np.isnan(p):
            trade_returns.append(p / holdings[col]["ep"] - 1.0)

    equity_s = pd.Series(equity_points, index=pd.DatetimeIndex(data.trading_dates[:len(equity_points)]))
    m = compute_metrics(equity_s, trades=[], periods_per_year=252)
    tr = np.array(trade_returns)

    return BacktestResult(
        sharpe=m.sharpe, cagr=m.cagr, max_drawdown=m.max_drawdown,
        final_equity=float(equity_points[-1]) if equity_points else 0.0,
        win_rate=float((tr > 0).mean()) if len(tr) > 0 else 0.0,
        n_trades=len(trade_returns), equity_series=equity_s,
        trade_returns=list(trade_returns),
    )


def main():
    print("FULL-UNIVERSE VALIDATION")
    print("=" * 70)

    # Identify dead tickers
    data = load_bias_free_data()
    mat = data.close_mat
    n_dates = len(data.trading_dates)

    dead_tickers = set()
    for i, t in enumerate(data.tickers):
        valid = np.where(~np.isnan(mat[:, i]))[0]
        if len(valid) > 0 and valid[-1] < n_dates - 10:
            dead_tickers.add(t)

    print(f"Data: {n_dates} bars, {len(data.tickers)} tickers, {len(dead_tickers)} dead")

    configs = [
        ("A: Baseline (S&P only)", "baseline", {}),
        ("B: Full universe", "full", {}),
        ("C: Full universe ex-dead", "full", {"exclude_tickers": dead_tickers}),
        ("D: Full universe + delist penalty", "full", {"delisting_penalty": True}),
    ]

    for period, start, end in [
        ("Full (2000-2026)", "2000-01-01", "2026-04-13"),
        ("Train (2000-2018)", "2000-01-01", "2018-12-31"),
        ("Holdout (2019-2026)", "2019-01-01", "2026-04-13"),
    ]:
        print(f"\n  === {period} ===")
        clear_cache()
        pdata = load_bias_free_data(start=start, end=end)

        for name, mode, kwargs in configs:
            if mode == "baseline":
                r = run_bias_free_backtest(pdata)
            else:
                r = run_full_universe_backtest(pdata, **kwargs)
            calmar = abs(r.cagr / r.max_drawdown) if r.max_drawdown != 0 else 0
            print(
                f"  {name:<40s} Sharpe={r.sharpe:.3f}  CAGR={r.cagr:.1%}  "
                f"MDD={r.max_drawdown:.1%}  Calmar={calmar:.2f}  Trades={r.n_trades}"
            )

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("  If B ≈ C: improvement is NOT from dead stocks (survivorship bias minimal)")
    print("  If D ≈ B: delisting losses don't matter (dead stocks weren't selected)")
    print("  If D >> A: improvement is genuine even under worst-case assumptions")


if __name__ == "__main__":
    main()
