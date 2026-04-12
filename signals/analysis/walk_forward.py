"""Walk-forward consistency analysis.

Ported from Vibe-Trading's ``validation.py`` and simplified for the signals
project: works on an equity curve only (no trades parameter needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_analysis(
    equity_curve: pd.Series,
    n_windows: int = 5,
    periods_per_year: int = 252,
) -> dict:
    """Split equity curve into N sequential windows, compute per-window metrics.

    Args:
        equity_curve: Equity time series (monotonic DatetimeIndex).
        n_windows: Number of non-overlapping windows.
        periods_per_year: Annualization factor (252 for equities, 365 for crypto).

    Returns:
        Dict with:
          - windows: list of per-window {window, start, end, return, sharpe, max_dd}
          - consistency_rate: fraction of windows with positive return
          - sharpe_mean / sharpe_std: across windows
    """
    if len(equity_curve) < n_windows * 2:
        return {
            "error": f"need at least {n_windows * 2} bars for {n_windows} windows"
        }

    indices = equity_curve.index
    window_size = len(indices) // n_windows
    windows: list[dict] = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size if i < n_windows - 1 else len(indices)
        win_eq = equity_curve.iloc[start_idx:end_idx]
        win_start = indices[start_idx]
        win_end = indices[end_idx - 1]

        # Per-window return
        ret = (
            float(win_eq.iloc[-1] / win_eq.iloc[0] - 1)
            if win_eq.iloc[0] > 0
            else 0.0
        )

        # Per-window Sharpe
        win_returns = win_eq.pct_change().dropna().values
        if len(win_returns) > 1:
            std = win_returns.std()
            sharpe = float(
                win_returns.mean() / (std + 1e-10) * np.sqrt(periods_per_year)
            )
        else:
            sharpe = 0.0

        # Per-window max drawdown
        peak = win_eq.cummax()
        dd = (win_eq - peak) / peak.replace(0, 1)
        max_dd = float(dd.min())

        windows.append(
            {
                "window": i + 1,
                "start": (
                    str(win_start.date())
                    if hasattr(win_start, "date")
                    else str(win_start)
                ),
                "end": (
                    str(win_end.date())
                    if hasattr(win_end, "date")
                    else str(win_end)
                ),
                "return": round(ret, 6),
                "sharpe": round(sharpe, 4),
                "max_dd": round(max_dd, 6),
            }
        )

    returns_list = [w["return"] for w in windows]
    sharpes_list = [w["sharpe"] for w in windows]
    profitable_windows = sum(1 for r in returns_list if r > 0)

    return {
        "n_windows": n_windows,
        "windows": windows,
        "profitable_windows": profitable_windows,
        "consistency_rate": round(profitable_windows / n_windows, 4),
        "return_mean": round(float(np.mean(returns_list)), 6),
        "return_std": round(float(np.std(returns_list)), 6),
        "sharpe_mean": round(float(np.mean(sharpes_list)), 4),
        "sharpe_std": round(float(np.std(sharpes_list)), 4),
    }
