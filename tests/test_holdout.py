"""Tests for holdout split + deflated Sharpe."""

from __future__ import annotations

import pandas as pd

from signals.backtest.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    sharpe_ratio,
)
from signals.cli import _split_holdout


def test_split_holdout_basic():
    idx = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    prices = pd.DataFrame({"close": range(100)}, index=idx)
    train, holdout = _split_holdout(prices, 0.2)
    assert len(train) == 80
    assert len(holdout) == 20
    assert train.index[-1] < holdout.index[0]


def test_split_holdout_zero_returns_full():
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame({"close": range(10)}, index=idx)
    train, holdout = _split_holdout(prices, 0.0)
    assert len(train) == 10
    assert holdout.empty


def test_split_holdout_one_returns_full():
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame({"close": range(10)}, index=idx)
    train, holdout = _split_holdout(prices, 1.0)
    assert len(train) == 10
    assert holdout.empty


def test_expected_max_sharpe_grows_with_n_trials():
    """E[max SR] under H0 should be monotonically non-decreasing in N."""
    e1 = expected_max_sharpe(1)
    e10 = expected_max_sharpe(10)
    e100 = expected_max_sharpe(100)
    e1000 = expected_max_sharpe(1000)
    assert e1 == 0.0
    assert 0 < e10 < e100 < e1000


def test_deflated_sharpe_decreases_with_more_trials():
    """For a fixed observed Sharpe, DSR should drop as the number of trials
    grows — that's the whole point of deflation."""
    sr = 1.0
    n_obs = 1000
    dsr_few = deflated_sharpe_ratio(sr, n_trials=2, n_observations=n_obs)
    dsr_many = deflated_sharpe_ratio(sr, n_trials=200, n_observations=n_obs)
    assert dsr_few > dsr_many
    assert 0.0 <= dsr_many <= dsr_few <= 1.0


def test_deflated_sharpe_high_for_strong_signal():
    """A genuinely strong Sharpe on a large sample should clear DSR ≥ 0.95
    even after deflating for a modest sweep."""
    dsr = deflated_sharpe_ratio(sharpe=2.5, n_trials=20, n_observations=2000)
    assert dsr >= 0.95


def test_sharpe_with_risk_free_rate_lower_than_zero_rf():
    """Subtracting a positive risk-free rate must lower the Sharpe."""
    rng = pd.date_range("2020-01-01", periods=252, freq="D", tz="UTC")
    # Noisy positive-drift series so std > 0 and mean > 0.
    rets = pd.Series([0.001, -0.0005] * 126, index=rng)
    sr0 = sharpe_ratio(rets, periods_per_year=252, risk_free_rate=0.0)
    sr2 = sharpe_ratio(rets, periods_per_year=252, risk_free_rate=0.02)
    assert sr0 > 0
    assert sr2 < sr0
