"""Volatility features."""

from __future__ import annotations

import pandas as pd


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of returns."""
    return returns.rolling(window=window, min_periods=window).std().rename(
        f"volatility_{window}d"
    )
