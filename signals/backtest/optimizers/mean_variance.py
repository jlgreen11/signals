"""Mean-variance (max Sharpe) optimizer.

Maximize (w'mu - r_f) / sqrt(w'Sigma w), subject to w >= 0, sum(w) = 1.
Long-only constraint via scipy SLSQP.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from signals.backtest.optimizers.base import BaseOptimizer


class MeanVarianceOptimizer(BaseOptimizer):
    """Maximize Sharpe ratio subject to long-only simplex."""

    def __init__(
        self, lookback: int = 60, risk_free: float = 0.0, **kwargs: Any
    ) -> None:
        super().__init__(lookback=lookback, **kwargs)
        self.risk_free = risk_free

    def _build_context(
        self, window: pd.DataFrame, active: list[str]
    ) -> dict[str, Any] | None:
        """Mean vector and covariance."""
        mu = window.mean().values
        cov = window.cov().values
        if np.isnan(cov).any() or np.isnan(mu).any():
            return None
        return {"cov": cov, "mu": mu}

    def _calc_weights(self, ctx: dict[str, Any]) -> np.ndarray:
        """SLSQP max-Sharpe weights."""
        from scipy.optimize import minimize

        mu, cov = ctx["mu"], ctx["cov"]
        n = len(mu)
        if n == 0:
            return self._equal_weight(0)

        rf = self.risk_free

        def neg_sharpe(w: np.ndarray) -> float:
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                return 0.0
            return -(w @ mu - rf) / port_vol

        result = minimize(
            neg_sharpe,
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
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Convenience: max-Sharpe weights from the signals project's data format.

    Args:
        prices_dict: {ticker: DataFrame with 'close' column}.
        selected_tickers: Tickers to allocate across.
        lookback: Rolling window days.
        risk_free: Annualized risk-free rate.

    Returns:
        {ticker: weight} dict, weights sum to 1.0.
    """
    return MeanVarianceOptimizer(
        lookback=lookback, risk_free=risk_free
    ).optimize_from_prices(prices_dict, selected_tickers)
