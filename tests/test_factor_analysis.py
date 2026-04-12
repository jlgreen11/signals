"""Tests for factor analysis and walk-forward modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.analysis.factor_analysis import (
    compute_factor_summary,
    compute_ic_series,
    layered_backtest,
)
from signals.analysis.walk_forward import walk_forward_analysis
from signals.model.multifactor import zscore_cross_section

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factor_and_returns(
    n_dates: int = 100,
    n_stocks: int = 20,
    perfect: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic factor values and forward returns.

    If ``perfect=True``, factor perfectly predicts returns (IC ~1.0).
    Otherwise, factor is random (IC ~0.0).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates, freq="B")
    tickers = [f"S{i:02d}" for i in range(n_stocks)]

    factor_data = rng.standard_normal((n_dates, n_stocks))
    factor_df = pd.DataFrame(factor_data, index=dates, columns=tickers)

    if perfect:
        # Returns are a monotone function of factor + small noise
        return_data = factor_data * 0.01 + rng.standard_normal((n_dates, n_stocks)) * 0.001
    else:
        # Returns are independent of factor
        return_data = rng.standard_normal((n_dates, n_stocks)) * 0.01

    return_df = pd.DataFrame(return_data, index=dates, columns=tickers)
    return factor_df, return_df


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------


class TestICComputation:
    def test_perfect_factor_high_ic(self) -> None:
        """A factor that perfectly predicts returns should have IC near 1.0."""
        factor_df, return_df = _make_factor_and_returns(perfect=True)
        ic = compute_ic_series(factor_df, return_df)
        assert len(ic) > 0
        assert ic.mean() > 0.8

    def test_random_factor_ic_near_zero(self) -> None:
        """A random factor should have IC near 0.0."""
        factor_df, return_df = _make_factor_and_returns(perfect=False)
        ic = compute_ic_series(factor_df, return_df)
        assert len(ic) > 0
        assert abs(ic.mean()) < 0.2

    def test_empty_input_returns_empty(self) -> None:
        empty = pd.DataFrame()
        ic = compute_ic_series(empty, empty)
        assert ic.empty


class TestFactorSummary:
    def test_summary_keys(self) -> None:
        factor_df, return_df = _make_factor_and_returns(perfect=True)
        ic = compute_ic_series(factor_df, return_df)
        summary = compute_factor_summary(ic)
        assert "ic_mean" in summary
        assert "ic_std" in summary
        assert "ir" in summary
        assert "ic_positive_ratio" in summary
        assert "ic_count" in summary

    def test_empty_ic_returns_zeros(self) -> None:
        summary = compute_factor_summary(pd.Series(dtype=float))
        assert summary["ic_mean"] == 0.0
        assert summary["ic_count"] == 0


# ---------------------------------------------------------------------------
# Layered backtest
# ---------------------------------------------------------------------------


class TestLayeredBacktest:
    def test_top_quintile_outperforms_with_strong_factor(self) -> None:
        """When factor is strong, top quintile should beat bottom quintile."""
        factor_df, return_df = _make_factor_and_returns(perfect=True, n_dates=200)
        equity = layered_backtest(factor_df, return_df, n_groups=5)
        assert not equity.empty
        # Group_5 (top) should outperform Group_1 (bottom)
        assert equity["Group_5"].iloc[-1] > equity["Group_1"].iloc[-1]

    def test_correct_number_of_groups(self) -> None:
        factor_df, return_df = _make_factor_and_returns()
        equity = layered_backtest(factor_df, return_df, n_groups=3)
        assert len(equity.columns) == 3


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


class TestWalkForward:
    def test_window_count_matches(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=500, freq="B")
        equity = pd.Series(
            np.exp(np.cumsum(np.random.default_rng(42).normal(0.0003, 0.01, 500))),
            index=dates,
        )
        result = walk_forward_analysis(equity, n_windows=5)
        assert result["n_windows"] == 5
        assert len(result["windows"]) == 5

    def test_consistency_rate_in_range(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=500, freq="B")
        equity = pd.Series(
            np.exp(np.cumsum(np.random.default_rng(42).normal(0.0003, 0.01, 500))),
            index=dates,
        )
        result = walk_forward_analysis(equity, n_windows=5)
        assert 0.0 <= result["consistency_rate"] <= 1.0

    def test_too_few_bars_returns_error(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=5, freq="B")
        equity = pd.Series([100, 101, 102, 103, 104], index=dates, dtype=float)
        result = walk_forward_analysis(equity, n_windows=5)
        assert "error" in result


# ---------------------------------------------------------------------------
# Z-score
# ---------------------------------------------------------------------------


class TestZScore:
    def test_mean_approx_zero(self) -> None:
        values = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 50.0}
        z = zscore_cross_section(values)
        mean_z = np.mean(list(z.values()))
        assert abs(mean_z) < 1e-6

    def test_std_approx_one(self) -> None:
        values = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 50.0}
        z = zscore_cross_section(values)
        std_z = np.std(list(z.values()), ddof=1)
        assert abs(std_z - 1.0) < 1e-6

    def test_handles_nan_values(self) -> None:
        values = {"A": 1.0, "B": float("nan"), "C": 3.0}
        z = zscore_cross_section(values)
        assert z["B"] == 0.0
        assert not np.isnan(z["A"])

    def test_single_value_returns_zeros(self) -> None:
        values = {"A": 5.0}
        z = zscore_cross_section(values)
        assert z["A"] == 0.0
