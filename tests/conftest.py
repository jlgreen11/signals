"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Deterministic price series with mean reversion → useful for tests."""
    rng = np.random.default_rng(42)
    n = 600
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    rets = rng.normal(0.0005, 0.02, size=n)
    # Inject mild mean reversion: tomorrow's return tilted opposite to today's
    for i in range(1, n):
        rets[i] -= 0.2 * rets[i - 1]
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
