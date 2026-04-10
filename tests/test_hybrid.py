"""Tests for HybridRegimeModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.hybrid import DEFAULT_ROUTING, HybridRegimeModel


def _features(prices: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=prices.index)
    f["return_1d"] = log_returns(prices["close"])
    f["volatility_20d"] = rolling_volatility(f["return_1d"], window=10)
    return f.dropna()


def test_hybrid_fits_all_three_components(synthetic_prices):
    """The hybrid must fit the regime detector + composite + HOMC."""
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(
        regime_n_states=3,
        regime_n_iter=30,
        homc_order=3,  # shorter so synthetic data supports it
    )
    m.fit(feats)
    assert m.fitted_
    assert m.regime_detector is not None and m.regime_detector.fitted_
    assert m.composite is not None and m.composite.fitted_
    assert m.homc is not None and m.homc.fitted_


def test_hybrid_predict_state_sets_active_component(synthetic_prices):
    """predict_state should set the active component based on the routed regime."""
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(regime_n_states=3, regime_n_iter=30, homc_order=3).fit(feats)
    state = m.predict_state(feats)
    # At least one component is now active — either homc (if current regime
    # routes to homc) or composite (otherwise).
    assert m.active_component_name in ("homc", "composite")
    assert m.last_regime_label in ("bear", "neutral", "bull")
    # predict_next must work after predict_state
    probs = m.predict_next(state)
    assert isinstance(probs, np.ndarray)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_hybrid_state_returns_delegates(synthetic_prices):
    """state_returns_ must match whichever component is currently active."""
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(regime_n_states=3, regime_n_iter=30, homc_order=3).fit(feats)
    m.predict_state(feats)
    if m.active_component_name == "composite":
        np.testing.assert_array_equal(m.state_returns_, m.composite.state_returns_)
    else:
        np.testing.assert_array_equal(m.state_returns_, m.homc.state_returns_)


def test_hybrid_custom_routing(synthetic_prices):
    """User-supplied routing must be honored."""
    feats = _features(synthetic_prices)
    # Route ALL regimes to HOMC
    m = HybridRegimeModel(
        regime_n_states=3,
        regime_n_iter=30,
        homc_order=3,
        routing={"bear": "homc", "neutral": "homc", "bull": "homc"},
    ).fit(feats)
    m.predict_state(feats)
    assert m.active_component_name == "homc"


def test_hybrid_routing_validation():
    """Invalid routing should be rejected at construction."""
    with pytest.raises(ValueError, match="routing"):
        HybridRegimeModel(routing={"bear": "foo", "neutral": "composite", "bull": "homc"})
    with pytest.raises(ValueError, match="routing"):
        HybridRegimeModel(routing={"bear": "composite"})  # missing neutral/bull
    with pytest.raises(ValueError, match="routing"):
        HybridRegimeModel(routing={"up": "homc", "down": "composite"})  # bad labels


def test_hybrid_save_load(tmp_path, synthetic_prices):
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(regime_n_states=3, regime_n_iter=30, homc_order=3).fit(feats)
    m.predict_state(feats)  # set active component

    path = tmp_path / "hybrid.json"
    m.save(path)
    m2 = HybridRegimeModel.load(path)
    assert m2.fitted_
    assert m2.regime_detector is not None
    assert m2.composite is not None
    assert m2.homc is not None
    assert m2.routing == m.routing
    # Reloaded model should produce the same regime classification
    m2.predict_state(feats)
    assert m2.last_regime_label == m.last_regime_label


def test_engine_runs_with_hybrid_backend(synthetic_prices):
    """End-to-end: BacktestEngine should run a hybrid config."""
    cfg = BacktestConfig(
        model_type="hybrid",
        train_window=300,
        retrain_freq=60,
        return_bins=3,
        volatility_bins=3,
        n_states=3,  # HOMC n_states=3 so synthetic data supports it
        order=3,
        vol_window=10,
        n_iter=30,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(synthetic_prices, symbol="TEST")
    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0
    assert len(result.benchmark_curve) > 0


def test_hybrid_engine_no_lookahead(synthetic_prices):
    """Hybrid must satisfy the same lookahead regression as the single models.

    Equity curve up to bar N must be bit-identical regardless of how much
    future data is in the input.
    """
    cfg = BacktestConfig(
        model_type="hybrid",
        train_window=300,
        retrain_freq=60,
        return_bins=3,
        volatility_bins=3,
        n_states=3,
        order=3,
        vol_window=10,
        n_iter=30,
    )

    cutoff = 500
    short_result = BacktestEngine(cfg).run(
        synthetic_prices.iloc[:cutoff], symbol="TEST"
    )
    long_result = BacktestEngine(cfg).run(synthetic_prices, symbol="TEST")

    short_eq = short_result.equity_curve.iloc[:-2]
    common = short_eq.index.intersection(long_result.equity_curve.index)
    assert len(common) > 100, f"too few common bars ({len(common)})"

    pd.testing.assert_series_equal(
        short_eq.loc[common],
        long_result.equity_curve.loc[common],
        check_names=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_hybrid_default_routing_is_conservative():
    """DEFAULT_ROUTING should send bear AND neutral to composite (bear-resistant)
    and only bull to HOMC (bull-participation). This encodes the Tier-0b
    finding that composite is the safer default."""
    assert DEFAULT_ROUTING["bear"] == "composite"
    assert DEFAULT_ROUTING["neutral"] == "composite"
    assert DEFAULT_ROUTING["bull"] == "homc"


def test_hybrid_vol_routing_fits_and_predicts(synthetic_prices):
    """Vol-based routing should work without an HMM regime detector."""
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(
        routing_strategy="vol",
        vol_quantile_threshold=0.75,
        homc_order=3,
    ).fit(feats)
    assert m.fitted_
    assert m.regime_detector is None
    assert m._vol_threshold_value is not None
    # Call predict_state — should route based on current vol
    state = m.predict_state(feats)
    assert m.active_component_name in ("homc", "composite")
    probs = m.predict_next(state)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_hybrid_vol_routing_selects_by_vol_threshold(synthetic_prices):
    """Feeding a high-vol tail should route to composite; a low-vol tail
    should route to homc."""
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(
        routing_strategy="vol",
        vol_quantile_threshold=0.5,  # median split for this test
        homc_order=3,
    ).fit(feats)

    # Force a high-vol observation by scaling the vol column
    high_vol = feats.copy()
    high_vol["volatility_20d"] = high_vol["volatility_20d"] * 10
    m.predict_state(high_vol)
    assert m.last_regime_label == "bear"
    assert m.active_component_name == "composite"

    # And a low-vol observation
    low_vol = feats.copy()
    low_vol["volatility_20d"] = low_vol["volatility_20d"] * 0.01
    m.predict_state(low_vol)
    assert m.last_regime_label == "bull"
    assert m.active_component_name == "homc"


def test_hybrid_vol_routing_no_lookahead(synthetic_prices):
    """Vol routing must also satisfy the lookahead regression."""
    cfg = BacktestConfig(
        model_type="hybrid",
        train_window=300,
        retrain_freq=60,
        return_bins=3,
        volatility_bins=3,
        n_states=3,
        order=3,
        vol_window=10,
        hybrid_routing_strategy="vol",
    )

    cutoff = 500
    short_result = BacktestEngine(cfg).run(
        synthetic_prices.iloc[:cutoff], symbol="TEST"
    )
    long_result = BacktestEngine(cfg).run(synthetic_prices, symbol="TEST")

    short_eq = short_result.equity_curve.iloc[:-2]
    common = short_eq.index.intersection(long_result.equity_curve.index)
    pd.testing.assert_series_equal(
        short_eq.loc[common],
        long_result.equity_curve.loc[common],
        check_names=False,
        rtol=1e-9,
        atol=1e-9,
    )
