"""Factor analysis: IC/IR computation and layered backtest.

Ported from Vibe-Trading's ``factor_analysis_tool.py`` and adapted to the
signals project's native data format (``prices_dict`` + callable factor
functions).

Core functions:
  - ``compute_ic_series``: daily Spearman rank IC between factor and returns
  - ``compute_factor_summary``: IC mean, std, IR, positive ratio
  - ``layered_backtest``: sort stocks into quintiles, track cumulative NAV
  - ``analyze_factor``: high-level wrapper accepting a factor callable
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic_series(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """Daily Spearman rank IC between factor cross-section and next-day returns.

    Args:
        factor_values: Factor values; index=date, columns=tickers.
        forward_returns: Forward returns; index=date, columns=tickers.

    Returns:
        IC series indexed by date.
    """
    common_dates = factor_values.index.intersection(forward_returns.index)
    common_tickers = factor_values.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_tickers) == 0:
        return pd.Series(dtype=float)

    factor_values = factor_values.loc[common_dates, common_tickers]
    forward_returns = forward_returns.loc[common_dates, common_tickers]

    ic_values: dict = {}
    for date in common_dates:
        f = factor_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        shared = f.index.intersection(r.index)
        if len(shared) < 5:
            continue
        corr, _ = spearmanr(f[shared], r[shared])
        if not np.isnan(corr):
            ic_values[date] = corr

    return pd.Series(ic_values, dtype=float)


def compute_factor_summary(ic_series: pd.Series) -> dict:
    """Summarize IC series: mean, std, IR, positive ratio.

    Args:
        ic_series: Output of ``compute_ic_series``.

    Returns:
        Dict with ic_mean, ic_std, ir, ic_positive_ratio, ic_count.
    """
    if ic_series.empty:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ir": 0.0,
            "ic_positive_ratio": 0.0,
            "ic_count": 0,
        }

    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std())
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_positive_ratio = float((ic_series > 0).mean())

    return {
        "ic_mean": round(ic_mean, 6),
        "ic_std": round(ic_std, 6),
        "ir": round(ir, 4),
        "ic_positive_ratio": round(ic_positive_ratio, 4),
        "ic_count": len(ic_series),
    }


def layered_backtest(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """Sort stocks into quintiles by factor score, track cumulative NAV per quintile.

    Args:
        factor_values: Factor values; index=date, columns=tickers.
        forward_returns: Forward returns; index=date, columns=tickers.
        n_groups: Number of quantile groups (default 5 = quintiles).

    Returns:
        DataFrame with index=date and columns Group_1..Group_N of cumulative NAV.
    """
    common_dates = sorted(factor_values.index.intersection(forward_returns.index))
    common_tickers = factor_values.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_tickers) == 0:
        return pd.DataFrame()

    factor_values = factor_values.loc[common_dates, common_tickers]
    forward_returns = forward_returns.loc[common_dates, common_tickers]

    group_returns: dict[str, list[float]] = {
        f"Group_{i + 1}": [] for i in range(n_groups)
    }
    valid_dates: list = []

    for date in common_dates:
        f = factor_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        shared = f.index.intersection(r.index)
        if len(shared) < n_groups:
            continue
        valid_dates.append(date)
        ranked = f[shared].rank(method="first")
        bins = pd.qcut(ranked, n_groups, labels=False, duplicates="drop")
        if bins.nunique() < n_groups:
            bins = pd.cut(ranked, n_groups, labels=False)
        for g in range(n_groups):
            members = bins[bins == g].index
            if len(members) > 0:
                group_returns[f"Group_{g + 1}"].append(r[members].mean())
            else:
                group_returns[f"Group_{g + 1}"].append(0.0)

    if not valid_dates:
        return pd.DataFrame()

    ret_df = pd.DataFrame(group_returns, index=valid_dates)
    equity_df = (1 + ret_df).cumprod()
    return equity_df


def analyze_factor(
    factor_name: str,
    prices_dict: dict[str, pd.DataFrame],
    factor_fn: Callable[[dict[str, pd.DataFrame], pd.Timestamp], dict[str, float]],
    n_groups: int = 5,
) -> dict:
    """High-level: compute a factor across a universe, run IC + layered backtest.

    Args:
        factor_name: Descriptive name for the factor.
        prices_dict: {ticker: DataFrame with 'close' column and DatetimeIndex}.
        factor_fn: Callable(prices_dict, date) -> {ticker: factor_value}.
            Must return factor values for a given cross-section date.
        n_groups: Number of quantile groups for the layered backtest.

    Returns:
        Dict with factor_name, ic_summary, layered_equity (DataFrame),
        long_short_spread, and group_final_equity.
    """
    # Build a common date index from all tickers
    all_dates: set[pd.Timestamp] = set()
    for df in prices_dict.values():
        all_dates.update(df.index)
    trading_dates = sorted(all_dates)

    if len(trading_dates) < 2:
        return {"factor_name": factor_name, "error": "insufficient dates"}

    # Compute factor values and forward returns for each date
    tickers = sorted(prices_dict.keys())
    factor_rows: dict[pd.Timestamp, dict[str, float]] = {}
    return_rows: dict[pd.Timestamp, dict[str, float]] = {}

    # Build close price DataFrame for forward returns
    close_frames = {t: prices_dict[t]["close"] for t in tickers}
    close_df = pd.DataFrame(close_frames)
    fwd_ret = close_df.pct_change().shift(-1)  # next-day return

    for date in trading_dates[:-1]:  # skip last (no forward return)
        try:
            fv = factor_fn(prices_dict, date)
        except Exception:
            continue
        if fv:
            factor_rows[date] = fv
            ret_row = {}
            for t in tickers:
                if date in fwd_ret.index and t in fwd_ret.columns:
                    val = fwd_ret.loc[date, t]
                    if not np.isnan(val):
                        ret_row[t] = val
            if ret_row:
                return_rows[date] = ret_row

    if not factor_rows or not return_rows:
        return {"factor_name": factor_name, "error": "no valid factor/return data"}

    factor_df = pd.DataFrame(factor_rows).T
    forward_returns_df = pd.DataFrame(return_rows).T

    ic_series = compute_ic_series(factor_df, forward_returns_df)
    ic_summary = compute_factor_summary(ic_series)
    equity_df = layered_backtest(factor_df, forward_returns_df, n_groups=n_groups)

    result: dict = {
        "factor_name": factor_name,
        "ic_summary": ic_summary,
        "layered_equity": equity_df,
    }

    if not equity_df.empty:
        result["long_short_spread"] = round(
            float(equity_df.iloc[-1, -1] - equity_df.iloc[-1, 0]), 4
        )
        result["group_final_equity"] = {
            col: round(float(equity_df[col].iloc[-1]), 4)
            for col in equity_df.columns
        }

    return result
