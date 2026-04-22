"""Algorithm V2 evaluation: multi-factor scoring + risk management.

Tests three improvements over the canonical early-breakout momentum:

1. INTEGRATED MULTI-FACTOR SCORING
   Blend acceleration with price-derived quality (low trailing vol) via z-scores.
   Academic basis: Asness, Frazzini & Pedersen (2019) show quality and momentum
   are near-zero correlated, making the combination powerful. Using trailing vol
   as quality proxy because we only have price data for 26 years (no historical
   fundamentals).

2. RISK-MANAGED SIZING (Barroso & Santa-Clara, JFE 2015)
   Scale total portfolio exposure by inverse trailing 6-month realized vol of
   the strategy. When the portfolio is volatile, hold fewer positions or more cash.
   Approximately doubles Sharpe ratio of naive momentum in the literature.

3. CRASH PROTECTION (Daniel & Moskowitz, JFE 2016)
   When trailing 2-year market return is negative AND portfolio vol is spiking,
   reduce momentum exposure.

Usage:
    cd /Users/jlg/claude/signals
    .venv/bin/python scripts/algo_v2_eval.py
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
    _get_constituents,
    clear_cache,
    default_acceleration_score,
    load_bias_free_data,
    run_bias_free_backtest,
)
from signals.backtest.metrics import compute_metrics


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score. NaN-safe."""
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return np.full_like(arr, 0.0)
    mu = float(np.nanmean(arr))
    sigma = float(np.nanstd(arr))
    if sigma < 1e-12:
        return np.full_like(arr, 0.0)
    out = (arr - mu) / sigma
    out[~valid] = np.nan
    return out


def make_multifactor_scorer(
    w_accel: float = 0.50,
    w_quality: float = 0.30,
    w_value: float = 0.20,
    vol_lookback: int = 63,
    min_short_return: float = 0.10,
    max_long_return: float = 1.50,
):
    """Return a score_fn that blends acceleration + quality + value z-scores."""
    _cache: dict[str, tuple] = {}

    def scorer(
        close_mat: np.ndarray,
        row: int,
        col: int,
        short: int = 63,
        long: int = 252,
    ) -> float | None:
        # Hard filters from canonical config
        accel = default_acceleration_score(
            close_mat, row, col, short, long,
            min_short_return, max_long_return,
        )
        if accel is None:
            return None

        # Check cache for cross-sectional z-scores at this row
        if "row" not in _cache or _cache["row"] != row:
            n_cols = close_mat.shape[1]
            accels = np.full(n_cols, np.nan)
            vols = np.full(n_cols, np.nan)
            values = np.full(n_cols, np.nan)

            for c in range(n_cols):
                # Acceleration (unfiltered for z-scoring)
                a = default_acceleration_score(
                    close_mat, row, c, short, long,
                    min_short_return=-999.0, max_long_return=999.0,
                )
                if a is not None:
                    accels[c] = a

                # Trailing vol (quality proxy)
                if row >= vol_lookback:
                    prices = close_mat[row - vol_lookback:row + 1, c]
                    vp = prices[~np.isnan(prices)]
                    if len(vp) >= vol_lookback // 2:
                        rets = np.diff(np.log(vp))
                        if len(rets) > 1:
                            vols[c] = np.std(rets)

                # Value proxy: negative of trailing return (mean reversion)
                if row >= 252:
                    p_now = close_mat[row, c]
                    p_ago = close_mat[row - 252, c]
                    if not (np.isnan(p_now) or np.isnan(p_ago)) and p_ago > 0:
                        values[c] = -(p_now / p_ago - 1.0)

            _cache["row"] = row
            _cache["z_accel"] = _zscore(accels)
            _cache["z_quality"] = _zscore(-vols)  # negate: low vol = high quality
            _cache["z_value"] = _zscore(values)

        za = _cache["z_accel"][col]
        zq = _cache["z_quality"][col]
        zv = _cache["z_value"][col]

        if np.isnan(za):
            return accel  # fallback

        composite = (
            w_accel * za
            + w_quality * (zq if not np.isnan(zq) else 0.0)
            + w_value * (zv if not np.isnan(zv) else 0.0)
        )
        return composite

    return scorer


def run_risk_managed_backtest(
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
    vol_target: float = 0.15,
    vol_lookback: int = 126,
    crash_lookback: int = 504,
) -> BacktestResult:
    """Bias-free backtest with Barroso/Santa-Clara risk scaling."""
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

        # Rebalance with risk scaling
        bars_since_rebal += 1
        if bars_since_rebal >= rebalance_freq and row >= long:
            # Risk scaling factor
            scale = 1.0
            if len(equity_points) >= vol_lookback:
                eq_arr = np.array(equity_points[-vol_lookback:])
                eq_rets = np.diff(eq_arr) / eq_arr[:-1]
                eq_rets = eq_rets[np.isfinite(eq_rets)]
                if len(eq_rets) > 10:
                    realized_vol = float(np.std(eq_rets)) * np.sqrt(252)
                    if realized_vol > 0.01:
                        scale = vol_target / realized_vol
                        scale = max(0.3, min(scale, 2.0))

            # Crash protection
            if row >= crash_lookback:
                mkt_now = float(np.nanmean(mat[row, :]))
                mkt_ago = float(np.nanmean(mat[row - crash_lookback, :]))
                if mkt_now > 0 and mkt_ago > 0 and mkt_now / mkt_ago - 1.0 < 0 and scale < 0.7:
                    scale *= 0.5

            adj_n_long = max(3, int(round(n_long * scale)))

            eligible = _get_constituents(data, data.trading_dates[row])
            eligible_cols = [data.ticker_to_idx[t] for t in eligible if t in data.ticker_to_idx]

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

            n_slots = adj_n_long - len(holdings)
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
                    target_n = max(len(holdings) + len(selected), adj_n_long)
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
    print("ALGORITHM V2 EVALUATION")
    print("=" * 70)

    data = load_bias_free_data()
    print(f"Data: {len(data.trading_dates)} bars, {len(data.tickers)} tickers\n")

    configs = [
        ("A: Baseline (accel only)", None, False, {}),
        ("B: Accel+Quality+Value (50/30/20)", make_multifactor_scorer(0.50, 0.30, 0.20), False, {}),
        ("C: Accel + Risk-managed", None, True, {}),
        ("D: Full (MF + Risk-mgmt)", make_multifactor_scorer(0.50, 0.30, 0.20), True, {}),
        ("E: Quality-heavy (35/45/20)", make_multifactor_scorer(0.35, 0.45, 0.20), False, {}),
        ("F: Accel + Tight risk (vol=0.12)", None, True, {"vol_target": 0.12}),
        ("G: Accel+Quality (60/40/0)", make_multifactor_scorer(0.60, 0.40, 0.0), False, {}),
        ("H: Full + Tight risk", make_multifactor_scorer(0.50, 0.30, 0.20), True, {"vol_target": 0.12}),
    ]

    results = []
    for name, scorer, risk_mgd, kwargs in configs:
        print(f"  Running: {name} ...", end="", flush=True)
        if risk_mgd:
            r = run_risk_managed_backtest(data, score_fn=scorer, **kwargs)
        else:
            r = run_bias_free_backtest(data, score_fn=scorer)
        results.append((name, r))
        calmar = abs(r.cagr / r.max_drawdown) if r.max_drawdown != 0 else 0
        print(f"  Sharpe={r.sharpe:.3f}  CAGR={r.cagr:.1%}  MDD={r.max_drawdown:.1%}  Calmar={calmar:.2f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Config':<40s} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'Final$':>12}")
    print("  " + "-" * 85)
    for name, r in results:
        calmar = abs(r.cagr / r.max_drawdown) if r.max_drawdown != 0 else 0
        print(f"  {name:<40s} {r.sharpe:>7.3f} {r.cagr:>6.1%} {r.max_drawdown:>6.1%} {calmar:>7.2f} ${r.final_equity:>11,.0f}")
    print(f"  {'SPY B&H (total return)':<40s} {'0.497':>7} {'8.0%':>7} {'-55.2%':>7} {'0.14':>7} ${'760,073':>11}")

    base = results[0][1]
    print("\n  Deltas vs baseline:")
    for name, r in results[1:]:
        print(f"    {name:<38s}  dSharpe={r.sharpe - base.sharpe:+.3f}  dMDD={r.max_drawdown - base.max_drawdown:+.1%}  dCAGR={r.cagr - base.cagr:+.1%}")

    # Train/holdout for top 3
    top3 = sorted(results, key=lambda x: x[1].sharpe, reverse=True)[:3]
    print("\n  Train/Holdout for top 3:")
    for name, _ in top3:
        cfg_idx = [i for i, (n, *_) in enumerate(configs) if n == name][0]
        _, scorer, risk_mgd, kwargs = configs[cfg_idx]
        for period, start, end in [("Train 2000-18", "2000-01-01", "2018-12-31"), ("Holdout 2019-26", "2019-01-01", "2026-04-13")]:
            clear_cache()
            pdata = load_bias_free_data(start=start, end=end)
            if risk_mgd:
                r = run_risk_managed_backtest(pdata, score_fn=scorer, **kwargs)
            else:
                r = run_bias_free_backtest(pdata, score_fn=scorer)
            print(f"    {name:<38s} {period:<15s} Sharpe={r.sharpe:.3f}  CAGR={r.cagr:.1%}  MDD={r.max_drawdown:.1%}")


if __name__ == "__main__":
    main()
