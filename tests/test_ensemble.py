"""Tests for EnsembleModel."""

from __future__ import annotations

import pandas as pd
import pytest

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.composite import CompositeMarkovChain
from signals.model.ensemble import EnsembleModel


def _features(prices: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=prices.index)
    f["close"] = prices["close"]
    f["open"] = prices["open"]
    f["return_1d"] = log_returns(prices["close"])
    f["volatility_20d"] = rolling_volatility(f["return_1d"], window=10)
    return f.dropna()


def test_ensemble_rejects_bad_weights():
    with pytest.raises(ValueError, match="sum to 1.0"):
        EnsembleModel(components=[
            ("a", CompositeMarkovChain(), 0.5),
            ("b", CompositeMarkovChain(), 0.3),
        ])


def test_weights_from_sharpes_equal():
    w = EnsembleModel.weights_from_sharpes({"a": 1.0, "b": 1.0, "c": 1.0})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(abs(v - 1 / 3) < 1e-9 for v in w.values())


def test_weights_from_sharpes_skewed_ranks_preserved():
    w = EnsembleModel.weights_from_sharpes({"a": 2.0, "b": 1.0, "c": 0.5})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["a"] > w["b"] > w["c"]


def test_weights_from_sharpes_negative_floor_policy_clips_to_zero():
    w = EnsembleModel.weights_from_sharpes(
        {"a": 2.0, "b": 1.0, "c": -0.5}, negative_policy="floor"
    )
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["c"] == 0.0


def test_weights_from_sharpes_negative_shift_policy_keeps_all_alive():
    w = EnsembleModel.weights_from_sharpes(
        {"a": 2.0, "b": 1.0, "c": -0.5}, negative_policy="shift"
    )
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v > 0 for v in w.values())


def test_weights_from_sharpes_all_negative_falls_back_to_equal():
    w = EnsembleModel.weights_from_sharpes({"a": -0.1, "b": -0.2, "c": -0.3})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(abs(v - 1 / 3) < 1e-9 for v in w.values())


def test_weights_from_sharpes_floor_parameter_enforced():
    w = EnsembleModel.weights_from_sharpes(
        {"a": 2.0, "b": 1.0, "c": -0.5}, floor=0.1
    )
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v >= 0.1 - 1e-9 for v in w.values())


def test_weights_from_sharpes_integrates_with_ensemble_constructor():
    """End-to-end: compute weights, use them to build an ensemble, verify construction."""
    sharpes = {"comp": 1.5, "homc": 2.0, "boost": 0.2}
    weights = EnsembleModel.weights_from_sharpes(sharpes, floor=0.05)
    from signals.model.boost import GradientBoostingModel
    from signals.model.homc import HigherOrderMarkovChain
    components = [
        ("comp", CompositeMarkovChain(return_bins=3, volatility_bins=3), weights["comp"]),
        ("homc", HigherOrderMarkovChain(n_states=3, order=2), weights["homc"]),
        ("boost", GradientBoostingModel(n_estimators=50, max_depth=2), weights["boost"]),
    ]
    ens = EnsembleModel(components=components)
    # Weight on HOMC (highest Sharpe) should be the largest.
    weights_by_name = {name: w for name, _, w in ens.components}
    assert weights_by_name["homc"] > weights_by_name["comp"]
    assert weights_by_name["homc"] > weights_by_name["boost"]


def test_ensemble_default_components_are_3way():
    m = EnsembleModel()
    names = [n for n, _, _ in m.components]
    assert set(names) == {"composite", "homc", "boost"}
    weights = [w for _, _, w in m.components]
    assert sum(weights) == pytest.approx(1.0)


def test_ensemble_fits_and_predicts(synthetic_prices):
    feats = _features(synthetic_prices)
    m = EnsembleModel()
    m.fit(feats)
    assert m.fitted_
    state = m.predict_state(feats)
    assert state == 0  # synthetic single-state
    probs = m.predict_next(state)
    assert probs.shape == (1,)
    assert probs[0] == 1.0
    # expected = probs @ state_returns_ = blended
    expected = float(probs @ m.state_returns_)
    assert expected == m._blended_expected


def test_ensemble_engine_runs(synthetic_prices):
    cfg = BacktestConfig(
        model_type="ensemble",
        train_window=200,
        retrain_freq=50,
        return_bins=3,
        volatility_bins=3,
        n_states=5,
        order=3,
        vol_window=10,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(synthetic_prices, symbol="TEST")
    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0


def test_ensemble_no_lookahead(synthetic_prices):
    """Ensemble inherits the lookahead contract from its components."""
    cfg = BacktestConfig(
        model_type="ensemble",
        train_window=200,
        retrain_freq=50,
        return_bins=3,
        volatility_bins=3,
        n_states=5,
        order=3,
        vol_window=10,
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
