"""Tests for portfolio optimizers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.backtest.optimizers.equal_volatility import (
    optimize_weights as equal_vol_weights,
)
from signals.backtest.optimizers.max_diversification import (
    optimize_weights as max_div_weights,
)
from signals.backtest.optimizers.mean_variance import (
    optimize_weights as mv_weights,
)
from signals.backtest.optimizers.risk_parity import (
    optimize_weights as rp_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(
    n_days: int = 200,
    tickers: list[str] | None = None,
    vols: list[float] | None = None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic price DataFrames for testing.

    Args:
        n_days: Number of trading days.
        tickers: Ticker names.
        vols: Per-ticker annualized volatility (used to scale daily returns).
        seed: Random seed.

    Returns:
        {ticker: DataFrame with 'close' column and DatetimeIndex}.
    """
    rng = np.random.default_rng(seed)
    if tickers is None:
        tickers = ["A", "B", "C", "D"]
    if vols is None:
        vols = [0.15, 0.25, 0.35, 0.45]

    dates = pd.bdate_range("2023-01-01", periods=n_days, freq="B")
    prices_dict: dict[str, pd.DataFrame] = {}

    for ticker, vol in zip(tickers, vols):  # noqa: B905
        daily_vol = vol / np.sqrt(252)
        returns = rng.normal(0.0003, daily_vol, n_days)
        close = 100.0 * np.exp(np.cumsum(returns))
        prices_dict[ticker] = pd.DataFrame({"close": close}, index=dates)

    return prices_dict


# ---------------------------------------------------------------------------
# Risk Parity
# ---------------------------------------------------------------------------


class TestRiskParity:
    def test_weights_sum_to_one(self) -> None:
        prices = _make_prices()
        w = rp_weights(prices, list(prices.keys()))
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_all_weights_positive(self) -> None:
        prices = _make_prices()
        w = rp_weights(prices, list(prices.keys()))
        for v in w.values():
            assert v >= 0.0

    def test_lower_vol_gets_more_weight(self) -> None:
        """Asset with lower vol should get higher weight than higher-vol asset."""
        prices = _make_prices(tickers=["LOW", "HIGH"], vols=[0.10, 0.50])
        w = rp_weights(prices, ["LOW", "HIGH"])
        assert w["LOW"] > w["HIGH"]


# ---------------------------------------------------------------------------
# Mean-Variance
# ---------------------------------------------------------------------------


class TestMeanVariance:
    def test_weights_sum_to_one(self) -> None:
        prices = _make_prices()
        w = mv_weights(prices, list(prices.keys()))
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_long_only(self) -> None:
        prices = _make_prices()
        w = mv_weights(prices, list(prices.keys()))
        for v in w.values():
            assert v >= -1e-9  # long-only constraint


# ---------------------------------------------------------------------------
# Max Diversification
# ---------------------------------------------------------------------------


class TestMaxDiversification:
    def test_weights_sum_to_one(self) -> None:
        prices = _make_prices()
        w = max_div_weights(prices, list(prices.keys()))
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_diversification_ratio_above_one(self) -> None:
        """Diversification ratio should be >= 1 for the optimized portfolio."""
        prices = _make_prices()
        tickers = list(prices.keys())
        w = max_div_weights(prices, tickers)

        # Build covariance from returns
        close_df = pd.DataFrame({t: prices[t]["close"] for t in tickers})
        ret_df = close_df.pct_change().dropna().tail(60)
        cov = ret_df.cov().values
        vols = np.sqrt(np.diag(cov))
        wv = np.array([w[t] for t in tickers])

        port_vol = np.sqrt(wv @ cov @ wv)
        weighted_vol_sum = wv @ vols
        dr = weighted_vol_sum / port_vol if port_vol > 1e-12 else 1.0
        assert dr >= 1.0 - 1e-6


# ---------------------------------------------------------------------------
# Equal Volatility
# ---------------------------------------------------------------------------


class TestEqualVolatility:
    def test_weights_sum_to_one(self) -> None:
        prices = _make_prices()
        w = equal_vol_weights(prices, list(prices.keys()))
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_inverse_vol_relationship(self) -> None:
        """Lower-vol asset should get higher weight."""
        prices = _make_prices(tickers=["LOW", "HIGH"], vols=[0.10, 0.40])
        w = equal_vol_weights(prices, ["LOW", "HIGH"])
        assert w["LOW"] > w["HIGH"]


# ---------------------------------------------------------------------------
# Base class edge cases
# ---------------------------------------------------------------------------


class TestBaseEdgeCases:
    def test_single_asset_returns_full_weight(self) -> None:
        prices = _make_prices(tickers=["ONLY"], vols=[0.20])
        w = rp_weights(prices, ["ONLY"])
        assert abs(w["ONLY"] - 1.0) < 1e-6

    def test_empty_tickers_returns_empty(self) -> None:
        prices = _make_prices()
        w = rp_weights(prices, [])
        assert w == {}

    def test_insufficient_history_falls_back(self) -> None:
        """With very few data points, should fall back to equal weight."""
        tickers = ["A", "B"]
        dates = pd.bdate_range("2023-01-01", periods=3, freq="B")
        prices = {
            t: pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=dates)
            for t in tickers
        }
        w = rp_weights(prices, tickers, lookback=60)
        # Should fall back to equal weight
        assert abs(w["A"] - 0.5) < 1e-6
        assert abs(w["B"] - 0.5) < 1e-6
