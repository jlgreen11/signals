"""Maximum diversification ratio: maximize (w' sigma) / sqrt(w' Sigma w).

``sigma`` is the vector of asset volatilities; ``Sigma`` is the covariance matrix.
Higher DR means more diversification per unit of risk.
Choueifaty & Coignard (2008).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from signals.backtest.optimizers.base import BaseOptimizer


class MaxDiversificationOptimizer(BaseOptimizer):
    """Maximize diversification ratio (Choueifaty & Coignard)."""

    def _calc_weights(self, ctx: dict[str, Any]) -> np.ndarray:
        """SLSQP max-DR weights."""
        from scipy.optimize import minimize

        cov = ctx["cov"]
        n = cov.shape[0]
        if n == 0:
            return self._equal_weight(0)

        vols = np.sqrt(np.diag(cov))
        if np.any(vols < 1e-12):
            return self._equal_weight(n)

        def neg_dr(w: np.ndarray) -> float:
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                return 0.0
            return -(w @ vols) / port_vol

        result = minimize(
            neg_dr,
            self._equal_weight(n),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"maxiter": 200, "ftol": 1e-10},
        )

        if result.success:
            return self._normalize(result.x)
        return self._equal_weight(n)


def optimize_weights(
    prices_dict: dict[str, pd.DataFrame],
    selected_tickers: list[str],
    lookback: int = 60,
) -> dict[str, float]:
    """Convenience: max-diversification weights from the signals project's data format.

    Args:
        prices_dict: {ticker: DataFrame with 'close' column}.
        selected_tickers: Tickers to allocate across.
        lookback: Rolling window days.

    Returns:
        {ticker: weight} dict, weights sum to 1.0.
    """
    return MaxDiversificationOptimizer(lookback=lookback).optimize_from_prices(
        prices_dict, selected_tickers
    )
