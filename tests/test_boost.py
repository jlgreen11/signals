"""Tests for GradientBoostingModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.boost import GradientBoostingModel


def _features(prices: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=prices.index)
    f["close"] = prices["close"]
    f["open"] = prices["open"]
    f["return_1d"] = log_returns(prices["close"])
    f["volatility_20d"] = rolling_volatility(f["return_1d"], window=10)
    return f.dropna()


def test_boost_fits_and_predicts(synthetic_prices):
    feats = _features(synthetic_prices)
    m = GradientBoostingModel(n_estimators=20, max_depth=2).fit(feats)
    assert m.fitted_
    state = m.predict_state(feats)
    assert state in (0, 1)


def test_boost_predict_next_is_one_hot():
    m = GradientBoostingModel()
    m.fitted_ = True
    np.testing.assert_array_equal(m.predict_next(0), [1.0, 0.0])
    np.testing.assert_array_equal(m.predict_next(1), [0.0, 1.0])


def test_boost_state_returns_saturate_signal():
    """The synthetic ±50 bps state_returns_ must cross both the buy
    threshold (25 bps) and sell threshold (-35 bps)."""
    m = GradientBoostingModel()
    assert m.state_returns_[0] < -0.0035
    assert m.state_returns_[1] > 0.0025


def test_boost_labels():
    m = GradientBoostingModel()
    m.fitted_ = True
    assert m.label(0) == "down"
    assert m.label(1) == "up"


def test_boost_rejects_insufficient_data(synthetic_prices):
    """Not enough training data → error."""
    m = GradientBoostingModel(min_training_samples=10_000)
    with pytest.raises(ValueError, match="Need >="):
        m.fit(_features(synthetic_prices))


def test_boost_save_load_roundtrip(tmp_path, synthetic_prices):
    feats = _features(synthetic_prices)
    m = GradientBoostingModel(n_estimators=20, max_depth=2).fit(feats)
    path = tmp_path / "boost.json"
    m.save(path)
    m2 = GradientBoostingModel.load(path)
    assert m2.fitted_
    assert m2.n_estimators == m.n_estimators
    assert m2.predict_state(feats) == m.predict_state(feats)


def test_engine_runs_with_boost_backend(synthetic_prices):
    cfg = BacktestConfig(
        model_type="boost",
        train_window=200,
        retrain_freq=60,
        vol_window=10,
        boost_n_estimators=20,
        boost_max_depth=2,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(synthetic_prices, symbol="TEST")
    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0


def test_boost_engine_no_lookahead(synthetic_prices):
    """Feature engineering must not reach into the future."""
    cfg = BacktestConfig(
        model_type="boost",
        train_window=200,
        retrain_freq=60,
        vol_window=10,
        boost_n_estimators=20,
        boost_max_depth=2,
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
