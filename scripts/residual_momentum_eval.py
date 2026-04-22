"""Residual momentum evaluation: market-adjusted acceleration signal.

Academic basis: Blitz, Huij & Martens (2011) "Residual Momentum" shows that
stock-specific momentum (after removing market/sector returns) is more
persistent and less crash-prone than raw momentum.

Instead of raw returns, we compute:
  residual_return = stock_return - market_return (equal-weighted cross-section)
then compute acceleration on residual returns.

Also tests a 50/50 blend of raw + residual acceleration (z-scored before
blending).

Usage:
    cd /Users/jlg/claude/signals
    .venv/bin/python scripts/residual_momentum_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.backtest.bias_free import (
    clear_cache,
    default_acceleration_score,
    load_bias_free_data,
    run_bias_free_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Residual momentum scorer
# ---------------------------------------------------------------------------

def _compute_betas(
    close_mat: np.ndarray,
    row: int,
    beta_lookback: int = 252,
) -> np.ndarray:
    """Estimate trailing beta for each stock vs equal-weighted market.

    Uses daily log returns over the lookback window. Returns array of
    betas (n_cols,) with NaN where insufficient data.
    """
    n_cols = close_mat.shape[1]
    betas = np.full(n_cols, np.nan)

    if row < beta_lookback:
        return betas

    # Daily log returns for the lookback window
    prices = close_mat[row - beta_lookback:row + 1, :]  # (lookback+1, n_cols)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rets = np.diff(np.log(prices), axis=0)  # (lookback, n_cols)

    # Market return = equal-weighted mean of all valid stocks each day
    mkt_rets = np.nanmean(log_rets, axis=1)  # (lookback,)

    mkt_var = np.nanvar(mkt_rets)
    if mkt_var < 1e-16:
        return betas

    for c in range(n_cols):
        stock_rets = log_rets[:, c]
        valid = ~np.isnan(stock_rets) & ~np.isnan(mkt_rets)
        if valid.sum() < beta_lookback // 2:
            continue
        sr = stock_rets[valid]
        mr = mkt_rets[valid]
        cov = np.mean(sr * mr) - np.mean(sr) * np.mean(mr)
        var = np.mean(mr ** 2) - np.mean(mr) ** 2
        if var > 1e-16:
            betas[c] = cov / var

    return betas


def make_residual_momentum_scorer(
    min_short_return: float = 0.10,
    max_long_return: float = 1.50,
    beta_lookback: int = 252,
):
    """Return a score_fn that computes acceleration on beta-adjusted residual returns.

    For each rebalance row, we compute:
      - beta_i from trailing daily returns (stock vs EW market)
      - market_ret_short/long = EW mean return across all stocks
      - residual_ret = stock_ret - beta_i * market_ret  (stock-specific!)
      - acceleration on residual returns

    Unlike simple market subtraction (which is a constant shift that doesn't
    change cross-sectional rankings), beta-adjustment makes the residual
    stock-specific because each stock has a different beta.
    """
    _cache: dict[str, object] = {}

    def scorer(
        close_mat: np.ndarray,
        row: int,
        col: int,
        short: int = 63,
        long: int = 252,
    ) -> float | None:
        if row < long:
            return None

        # Compute and cache market returns + betas for this row
        cache_key = f"{row}_{short}_{long}"
        if _cache.get("key") != cache_key:
            n_cols = close_mat.shape[1]
            p_now_all = close_mat[row, :]
            p_short_all = close_mat[row - short, :]
            p_long_all = close_mat[row - long, :]

            # Vectorized short returns
            valid_short = (
                ~np.isnan(p_now_all) & ~np.isnan(p_short_all) & (p_short_all > 0)
            )
            ret_short_all = np.full(n_cols, np.nan)
            ret_short_all[valid_short] = (
                p_now_all[valid_short] / p_short_all[valid_short] - 1.0
            )

            # Vectorized long returns
            valid_long = (
                ~np.isnan(p_now_all) & ~np.isnan(p_long_all) & (p_long_all > 0)
            )
            ret_long_all = np.full(n_cols, np.nan)
            ret_long_all[valid_long] = (
                p_now_all[valid_long] / p_long_all[valid_long] - 1.0
            )

            mkt_ret_short = float(np.nanmean(ret_short_all))
            mkt_ret_long = float(np.nanmean(ret_long_all))

            # Trailing betas
            betas = _compute_betas(close_mat, row, beta_lookback)

            _cache["key"] = cache_key
            _cache["ret_short"] = ret_short_all
            _cache["ret_long"] = ret_long_all
            _cache["mkt_short"] = mkt_ret_short
            _cache["mkt_long"] = mkt_ret_long
            _cache["betas"] = betas

        ret_short_all = _cache["ret_short"]
        ret_long_all = _cache["ret_long"]
        mkt_ret_short = _cache["mkt_short"]
        mkt_ret_long = _cache["mkt_long"]
        betas = _cache["betas"]

        # This stock's raw returns
        rs = ret_short_all[col]
        rl = ret_long_all[col]
        if np.isnan(rs) or np.isnan(rl):
            return None

        # Beta-adjusted residual returns (stock-specific!)
        beta = betas[col]
        if np.isnan(beta):
            beta = 1.0  # fallback to market-neutral if no beta estimate
        resid_short = rs - beta * mkt_ret_short
        resid_long = rl - beta * mkt_ret_long

        # Apply minimum short return filter on RAW return
        if rs <= min_short_return:
            return None

        # Moonshot filter on raw long return
        if rl > max_long_return:
            return None

        # Acceleration on residual returns
        accel = resid_short - resid_long / (long / short)
        return accel

    return scorer


# ---------------------------------------------------------------------------
# Blended scorer: 50% raw acceleration + 50% residual acceleration (z-scored)
# ---------------------------------------------------------------------------

def make_blended_scorer(
    w_raw: float = 0.50,
    w_resid: float = 0.50,
    min_short_return: float = 0.10,
    max_long_return: float = 1.50,
    beta_lookback: int = 252,
):
    """Return a score_fn that blends raw and beta-adjusted residual acceleration via z-scores."""
    _cache: dict[str, object] = {}

    def scorer(
        close_mat: np.ndarray,
        row: int,
        col: int,
        short: int = 63,
        long: int = 252,
    ) -> float | None:
        if row < long:
            return None

        # Build cross-sectional z-scores for this row (cached)
        cache_key = f"{row}_{short}_{long}"
        if _cache.get("key") != cache_key:
            n_cols = close_mat.shape[1]
            p_now_all = close_mat[row, :]
            p_short_all = close_mat[row - short, :]
            p_long_all = close_mat[row - long, :]

            valid_short = (
                ~np.isnan(p_now_all) & ~np.isnan(p_short_all) & (p_short_all > 0)
            )
            valid_long = (
                ~np.isnan(p_now_all) & ~np.isnan(p_long_all) & (p_long_all > 0)
            )

            ret_short_all = np.full(n_cols, np.nan)
            ret_short_all[valid_short] = p_now_all[valid_short] / p_short_all[valid_short] - 1.0
            ret_long_all = np.full(n_cols, np.nan)
            ret_long_all[valid_long] = p_now_all[valid_long] / p_long_all[valid_long] - 1.0

            mkt_short = float(np.nanmean(ret_short_all))
            mkt_long = float(np.nanmean(ret_long_all))

            # Trailing betas for beta-adjusted residuals
            betas = _compute_betas(close_mat, row, beta_lookback)

            # Raw acceleration for all stocks
            raw_accel = np.full(n_cols, np.nan)
            # Beta-adjusted residual acceleration for all stocks
            resid_accel = np.full(n_cols, np.nan)

            both_valid = valid_short & valid_long
            ratio = long / short

            for c in range(n_cols):
                if not both_valid[c]:
                    continue
                rs = ret_short_all[c]
                rl = ret_long_all[c]
                raw_accel[c] = rs - rl / ratio

                beta = betas[c] if not np.isnan(betas[c]) else 1.0
                resid_s = rs - beta * mkt_short
                resid_l = rl - beta * mkt_long
                resid_accel[c] = resid_s - resid_l / ratio

            _cache["key"] = cache_key
            _cache["z_raw"] = _zscore(raw_accel)
            _cache["z_resid"] = _zscore(resid_accel)
            _cache["ret_short"] = ret_short_all
            _cache["ret_long"] = ret_long_all

        z_raw = _cache["z_raw"]
        z_resid = _cache["z_resid"]
        ret_short_all = _cache["ret_short"]
        ret_long_all = _cache["ret_long"]

        rs = ret_short_all[col]
        rl = ret_long_all[col]
        if np.isnan(rs) or np.isnan(rl):
            return None

        # Filters on raw returns
        if rs <= min_short_return:
            return None
        if rl > max_long_return:
            return None

        zr = z_raw[col]
        zres = z_resid[col]
        if np.isnan(zr) or np.isnan(zres):
            return None

        return w_raw * zr + w_resid * zres

    return scorer


# ---------------------------------------------------------------------------
# Period evaluation helper
# ---------------------------------------------------------------------------

def eval_period(name: str, start: str, end: str, scorers: dict[str, object]):
    """Run all scorers on a date range and return results."""
    clear_cache()
    data = load_bias_free_data(start=start, end=end)
    n_dates = len(data.trading_dates)
    n_tickers = len(data.tickers)
    print(f"\n  {name}: {n_dates} bars, {n_tickers} tickers")

    results = {}
    for label, scorer in scorers.items():
        r = run_bias_free_backtest(data, score_fn=scorer)
        calmar = abs(r.cagr / r.max_drawdown) if r.max_drawdown != 0 else 0
        results[label] = r
        print(
            f"    {label:<30s}  Sharpe={r.sharpe:.3f}  "
            f"CAGR={r.cagr:.1%}  MDD={r.max_drawdown:.1%}  "
            f"Calmar={calmar:.2f}  Trades={r.n_trades}  "
            f"WinRate={r.win_rate:.1%}"
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("RESIDUAL MOMENTUM EVALUATION")
    print("=" * 70)
    print("Academic basis: Blitz, Huij & Martens (2011)")
    print("Removes equal-weighted market return before computing acceleration")
    print("=" * 70)

    scorers = {
        "Baseline (raw accel)": None,  # uses default
        "Residual momentum": make_residual_momentum_scorer(),
        "Blend 50/50 (z-scored)": make_blended_scorer(0.50, 0.50),
    }

    # Full period
    print("\n" + "-" * 70)
    print("FULL PERIOD (2000-2026)")
    print("-" * 70)
    full = eval_period("Full", "2000-01-01", "2026-04-13", scorers)

    # Train
    print("\n" + "-" * 70)
    print("TRAIN (2000-2018)")
    print("-" * 70)
    train = eval_period("Train", "2000-01-01", "2018-12-31", scorers)

    # Holdout
    print("\n" + "-" * 70)
    print("HOLDOUT (2019-2026)")
    print("-" * 70)
    holdout = eval_period("Holdout", "2019-01-01", "2026-04-13", scorers)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n  {'Period':<12} {'Strategy':<30} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'WinRate':>8}")
    print("  " + "-" * 85)

    for period_name, period_results in [("Full", full), ("Train", train), ("Holdout", holdout)]:
        for label, r in period_results.items():
            calmar = abs(r.cagr / r.max_drawdown) if r.max_drawdown != 0 else 0
            print(
                f"  {period_name:<12} {label:<30} {r.sharpe:>7.3f} "
                f"{r.cagr:>6.1%} {r.max_drawdown:>6.1%} "
                f"{calmar:>7.2f} {r.win_rate:>7.1%}"
            )
        print()

    # Delta vs baseline
    print("  Deltas vs Baseline (raw accel):")
    for period_name, period_results in [("Full", full), ("Train", train), ("Holdout", holdout)]:
        base = period_results["Baseline (raw accel)"]
        for label, r in period_results.items():
            if label == "Baseline (raw accel)":
                continue
            print(
                f"    {period_name:<10} {label:<28} "
                f"dSharpe={r.sharpe - base.sharpe:+.3f}  "
                f"dCAGR={r.cagr - base.cagr:+.1%}  "
                f"dMDD={r.max_drawdown - base.max_drawdown:+.1%}"
            )


if __name__ == "__main__":
    main()
