"""Tests for TrendFilter and DualMovingAverage models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.model.trend import DualMovingAverage, TrendFilter


def _features(prices: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=prices.index)
    f["close"] = prices["close"]
    f["open"] = prices["open"]
    return f


# ============================================================
# TrendFilter
# ============================================================


def test_trend_filter_fits_and_predicts(synthetic_prices):
    feats = _features(synthetic_prices)
    m = TrendFilter(window=50).fit(feats)
    assert m.fitted_
    assert m.state_counts_.sum() > 0
    state = m.predict_state(feats)
    assert state in (0, 1)


def test_trend_filter_predict_next_is_one_hot():
    m = TrendFilter(window=10)
    m.fitted_ = True
    np.testing.assert_array_equal(m.predict_next(0), [1.0, 0.0])
    np.testing.assert_array_equal(m.predict_next(1), [0.0, 1.0])


def test_trend_filter_state_returns_saturate_signal():
    """The synthetic [-1, +1] state_returns_ must cross both the buy
    threshold (25 bps) and sell threshold (-35 bps) decisively."""
    m = TrendFilter(window=10)
    assert m.state_returns_[0] < -0.0035  # below sell threshold
    assert m.state_returns_[1] > 0.0025   # above buy threshold


def test_trend_filter_rejects_short_window():
    with pytest.raises(ValueError):
        TrendFilter(window=1)


def test_trend_filter_needs_enough_data_to_fit(synthetic_prices):
    # window=500 with a 600-bar fixture — should work
    TrendFilter(window=500).fit(_features(synthetic_prices))
    # window=800 — too few bars
    with pytest.raises(ValueError, match="Need >="):
        TrendFilter(window=800).fit(_features(synthetic_prices))


def test_trend_filter_save_load_roundtrip(tmp_path, synthetic_prices):
    feats = _features(synthetic_prices)
    m = TrendFilter(window=50).fit(feats)
    path = tmp_path / "trend.json"
    m.save(path)
    m2 = TrendFilter.load(path)
    assert m2.window == m.window
    assert m2.fitted_
    assert m2.predict_state(feats) == m.predict_state(feats)


def test_trend_filter_detects_above_ma():
    """Feed a rising series and check state=1. Last bar at 130, last
    9 bars before it at 100 → MA(10) = 103 → last close (130) > 103."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    close = np.concatenate([np.full(90, 100.0), np.full(9, 100.0), [130.0]])
    df = pd.DataFrame({"close": close, "open": close}, index=idx)
    m = TrendFilter(window=10).fit(df)
    assert m.predict_state(df) == 1  # last close 130 > MA(10)=103


def test_trend_filter_detects_below_ma():
    """Last bar at 70, last 9 bars before at 100 → MA(10) = 97 →
    70 < 97 → state 0."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    close = np.concatenate([np.full(90, 100.0), np.full(9, 100.0), [70.0]])
    df = pd.DataFrame({"close": close, "open": close}, index=idx)
    m = TrendFilter(window=10).fit(df)
    assert m.predict_state(df) == 0  # last close 70 < MA(10)=97


# ============================================================
# DualMovingAverage
# ============================================================


def test_dual_ma_fits_and_predicts(synthetic_prices):
    feats = _features(synthetic_prices)
    m = DualMovingAverage(fast_window=10, slow_window=30).fit(feats)
    assert m.fitted_
    state = m.predict_state(feats)
    assert state in (0, 1)


def test_dual_ma_rejects_bad_windows():
    with pytest.raises(ValueError, match="fast_window must be <"):
        DualMovingAverage(fast_window=100, slow_window=50)
    with pytest.raises(ValueError, match=">= 2"):
        DualMovingAverage(fast_window=1, slow_window=10)


def test_dual_ma_labels():
    m = DualMovingAverage(fast_window=10, slow_window=30)
    m.fitted_ = True
    assert m.label(0) == "death_cross"
    assert m.label(1) == "golden_cross"


def test_dual_ma_save_load_roundtrip(tmp_path, synthetic_prices):
    feats = _features(synthetic_prices)
    m = DualMovingAverage(fast_window=10, slow_window=30).fit(feats)
    path = tmp_path / "dual_ma.json"
    m.save(path)
    m2 = DualMovingAverage.load(path)
    assert m2.fast_window == m.fast_window
    assert m2.slow_window == m.slow_window
    assert m2.fitted_
    assert m2.predict_state(feats) == m.predict_state(feats)


# ============================================================
# Engine integration
# ============================================================


def test_engine_runs_with_trend_backend(synthetic_prices):
    cfg = BacktestConfig(
        model_type="trend",
        trend_window=50,
        train_window=100,  # must be >= trend_window
        retrain_freq=40,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(synthetic_prices, symbol="TEST")
    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0


def test_engine_runs_with_golden_cross_backend(synthetic_prices):
    cfg = BacktestConfig(
        model_type="golden_cross",
        trend_fast_window=20,
        trend_slow_window=50,
        train_window=100,
        retrain_freq=40,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(synthetic_prices, symbol="TEST")
    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0


# ============================================================
# Lookahead regression — critical for MA-based models
# ============================================================


def test_trend_filter_engine_no_lookahead(synthetic_prices):
    """The trend filter's rolling MA must never reach into the future.
    Classic trap: pandas default rolling is right-aligned (no leakage),
    but a bug could flip to center=True or shift the wrong direction."""
    cfg = BacktestConfig(
        model_type="trend",
        trend_window=50,
        train_window=100,
        retrain_freq=40,
    )

    cutoff = 500
    short_result = BacktestEngine(cfg).run(
        synthetic_prices.iloc[:cutoff], symbol="TEST"
    )
    long_result = BacktestEngine(cfg).run(synthetic_prices, symbol="TEST")

    short_eq = short_result.equity_curve.iloc[:-2]
    common = short_eq.index.intersection(long_result.equity_curve.index)
    assert len(common) > 100
    pd.testing.assert_series_equal(
        short_eq.loc[common],
        long_result.equity_curve.loc[common],
        check_names=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_dual_ma_engine_no_lookahead(synthetic_prices):
    cfg = BacktestConfig(
        model_type="golden_cross",
        trend_fast_window=20,
        trend_slow_window=50,
        train_window=100,
        retrain_freq=40,
    )

    cutoff = 500
    short_result = BacktestEngine(cfg).run(
        synthetic_prices.iloc[:cutoff], symbol="TEST"
    )
    long_result = BacktestEngine(cfg).run(synthetic_prices, symbol="TEST")

    short_eq = short_result.equity_curve.iloc[:-2]
    common = short_eq.index.intersection(long_result.equity_curve.index)
    assert len(common) > 100
    pd.testing.assert_series_equal(
        short_eq.loc[common],
        long_result.equity_curve.loc[common],
        check_names=False,
        rtol=1e-9,
        atol=1e-9,
    )
