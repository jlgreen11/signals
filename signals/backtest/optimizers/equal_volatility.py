"""Equal-volatility (inverse-volatility) weighting.

Higher weight on lower-volatility names so each asset contributes similar vol.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from signals.backtest.optimizers.base import BaseOptimizer


class EqualVolatilityOptimizer(BaseOptimizer):
    """Inverse-volatility weights without a full covariance model."""

    def _build_context(
        self, window: pd.DataFrame, active: list[str]
    ) -> dict[str, Any] | None:
        """Rolling per-asset volatilities."""
        vols = window.std()
        if vols.isna().any() or (vols < 1e-12).any():
            return None
        return {"vols": vols}

    def _calc_weights(self, ctx: dict[str, Any]) -> np.ndarray:
        """Inverse-volatility weights."""
        inv_vol = 1.0 / ctx["vols"]
        return (inv_vol / inv_vol.sum()).values


def optimize_weights(
    prices_dict: dict[str, pd.DataFrame],
    selected_tickers: list[str],
    lookback: int = 60,
) -> dict[str, float]:
    """Convenience: inverse-vol weights from the signals project's data format.

    Args:
        prices_dict: {ticker: DataFrame with 'close' column}.
        selected_tickers: Tickers to allocate across.
        lookback: Rolling window days.

    Returns:
        {ticker: weight} dict, weights sum to 1.0.
    """
    return EqualVolatilityOptimizer(lookback=lookback).optimize_from_prices(
        prices_dict, selected_tickers
    )
