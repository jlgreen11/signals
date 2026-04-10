"""Misc technical indicators (placeholder for future expansion)."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean().rename(f"sma_{window}")


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean().rename(f"ema_{window}")
