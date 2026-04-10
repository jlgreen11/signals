"""Return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """Single-period log returns."""
    return np.log(close).diff().rename("return_1d")


def simple_returns(close: pd.Series) -> pd.Series:
    """Single-period simple returns."""
    return close.pct_change().rename("return_simple_1d")
