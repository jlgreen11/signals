"""Tests for NaiveVolFilter — the null hypothesis baseline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signals.model.vol_filter import NaiveVolFilter


def _make_obs(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic price data with known vol regimes."""
    rng = np.random.RandomState(seed)
    # First half: low vol. Second half: high vol.
    returns_low = rng.normal(0.001, 0.01, n // 2)
    returns_high = rng.normal(-0.001, 0.05, n - n // 2)
    returns = np.concatenate([returns_low, returns_high])
    close = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({"close": close, "open": close * 0.999}, index=idx)
    df["return_1d"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["return_1d"].rolling(10, min_periods=10).std()
    return df


def test_fit_sets_threshold():
    obs = _make_obs()
    m = NaiveVolFilter(vol_window=10, quantile=0.50)
    m.fit(obs)
    assert m.fitted_
    assert m.vol_threshold > 0


def test_predict_state_low_vol_is_long():
    obs = _make_obs()
    m = NaiveVolFilter(vol_window=10, quantile=0.50)
    m.fit(obs)
    # Low-vol window (first half of data)
    state = m.predict_state(obs.iloc[:100])
    assert state == 1  # low vol → long


def test_predict_state_high_vol_is_flat():
    obs = _make_obs()
    m = NaiveVolFilter(vol_window=10, quantile=0.50)
    m.fit(obs)
    # High-vol window (second half of data)
    state = m.predict_state(obs.iloc[200:])
    assert state == 0  # high vol → flat


def test_predict_next_one_hot():
    m = NaiveVolFilter()
    m.fitted_ = True
    assert np.array_equal(m.predict_next(0), [1.0, 0.0])
    assert np.array_equal(m.predict_next(1), [0.0, 1.0])


def test_state_returns_saturated():
    m = NaiveVolFilter()
    assert m.state_returns_[0] == -1.0  # high vol → sell signal
    assert m.state_returns_[1] == +1.0  # low vol → buy signal


def test_labels():
    m = NaiveVolFilter()
    assert m.label(0) == "high_vol_flat"
    assert m.label(1) == "low_vol_long"


def test_save_load_roundtrip():
    obs = _make_obs()
    m = NaiveVolFilter(vol_window=10, quantile=0.70)
    m.fit(obs)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.json"
        m.save(path)
        m2 = NaiveVolFilter.load(path)
        assert m2.vol_window == m.vol_window
        assert m2.quantile == m.quantile
        assert m2.vol_threshold == pytest.approx(m.vol_threshold)
        assert m2.fitted_


def test_rejects_bad_quantile():
    with pytest.raises(ValueError, match="quantile"):
        NaiveVolFilter(quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        NaiveVolFilter(quantile=1.0)


def test_rejects_tiny_vol_window():
    with pytest.raises(ValueError, match="vol_window"):
        NaiveVolFilter(vol_window=1)


def test_not_fitted_raises():
    m = NaiveVolFilter()
    obs = _make_obs()
    with pytest.raises(RuntimeError, match="not fit"):
        m.predict_state(obs)


def test_backtest_engine_integration(synthetic_prices):
    """Vol filter runs through the full backtest engine."""
    from signals.backtest.engine import BacktestConfig, BacktestEngine

    cfg = BacktestConfig(
        model_type="vol_filter",
        train_window=200,
        retrain_freq=40,
        vol_window=10,
        vol_filter_quantile=0.50,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(synthetic_prices, symbol="TEST")
    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0


def test_fits_from_return_col_without_volatility():
    """Can fit using return_1d alone (no pre-computed volatility column)."""
    obs = _make_obs()
    obs_no_vol = obs[["close", "open", "return_1d"]].copy()
    m = NaiveVolFilter(vol_window=10, quantile=0.50)
    m.fit(obs_no_vol)
    assert m.fitted_
    assert m.vol_threshold > 0
