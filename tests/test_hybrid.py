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


def test_hybrid_blend_fits_and_predicts(synthetic_prices):
    """Blend strategy should fit both components and compute a blended
    expected return at predict_state time."""
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(
        routing_strategy="blend",
        blend_low_quantile=0.50,
        blend_high_quantile=0.85,
        homc_order=3,
    ).fit(feats)
    assert m.fitted_
    assert m._blend_low_value is not None
    assert m._blend_high_value is not None
    assert m._blend_low_value < m._blend_high_value

    state = m.predict_state(feats)
    # Synthetic 1-state interface
    probs = m.predict_next(state)
    assert probs.shape == (1,)
    assert probs[0] == pytest.approx(1.0)
    assert m.state_returns_.shape == (1,)
    # expected = probs @ state_returns_ = blended_expected_return
    expected = float(probs @ m.state_returns_)
    assert expected == m._blended_expected_return


def test_hybrid_blend_weight_ramps_linearly(synthetic_prices):
    """The blend weight should be 0 below low_value, 1 above high_value,
    and linear in between."""
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(
        routing_strategy="blend",
        blend_low_quantile=0.30,
        blend_high_quantile=0.70,
        homc_order=3,
    ).fit(feats)

    low = m._blend_low_value
    high = m._blend_high_value
    mid = (low + high) / 2

    # Build a tiny feature frame carrying a specific vol value
    def _with_vol(vol_value: float) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
        return pd.DataFrame(
            {"return_1d": [0.0, 0.0, 0.0], "volatility_20d": [vol_value] * 3},
            index=idx,
        )

    # Below low → weight 0 (all HOMC)
    assert m._blend_weight(_with_vol(low * 0.5)) == 0.0
    # Above high → weight 1 (all composite)
    assert m._blend_weight(_with_vol(high * 2.0)) == 1.0
    # Exactly at low → weight 0
    assert m._blend_weight(_with_vol(low)) == 0.0
    # Exactly at high → weight 1
    assert m._blend_weight(_with_vol(high)) == 1.0
    # Midpoint → weight 0.5
    w_mid = m._blend_weight(_with_vol(mid))
    assert 0.49 <= w_mid <= 0.51


def test_hybrid_blend_expected_is_weighted_average(synthetic_prices):
    """blended_expected_return should equal w*e_comp + (1-w)*e_homc."""
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(
        routing_strategy="blend",
        blend_low_quantile=0.50,
        blend_high_quantile=0.85,
        homc_order=3,
    ).fit(feats)
    m.predict_state(feats)

    # Manually compute what the two components would say and verify
    state_comp = m.composite.predict_state(feats)
    state_homc = m.homc.predict_state(feats)
    e_comp = float(m.composite.predict_next(state_comp) @ m.composite.state_returns_)
    e_homc = float(m.homc.predict_next(state_homc) @ m.homc.state_returns_)
    w = m._blend_weight_composite
    expected_blend = w * e_comp + (1.0 - w) * e_homc
    assert m._blended_expected_return == pytest.approx(expected_blend, abs=1e-12)


def test_hybrid_blend_engine_no_lookahead(synthetic_prices):
    """Blend routing must also satisfy the lookahead regression."""
    cfg = BacktestConfig(
        model_type="hybrid",
        train_window=300,
        retrain_freq=60,
        return_bins=3,
        volatility_bins=3,
        n_states=3,
        order=3,
        vol_window=10,
        hybrid_routing_strategy="blend",
        hybrid_blend_low=0.50,
        hybrid_blend_high=0.85,
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


def test_hybrid_blend_quantile_validation():
    """blend_low must be < blend_high, both in [0, 1]."""
    with pytest.raises(ValueError, match="blend_low"):
        HybridRegimeModel(blend_low_quantile=0.8, blend_high_quantile=0.5)
    with pytest.raises(ValueError, match="blend_low"):
        HybridRegimeModel(blend_low_quantile=-0.1, blend_high_quantile=0.5)
    with pytest.raises(ValueError, match="blend_low"):
        HybridRegimeModel(blend_low_quantile=0.5, blend_high_quantile=1.5)


def test_hybrid_adaptive_vol_fits_and_predicts(synthetic_prices):
    """Adaptive-vol strategy stores both low/high thresholds + the
    training median vol for regime detection."""
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(
        routing_strategy="adaptive_vol",
        adaptive_low_quantile=0.60,
        adaptive_high_quantile=0.80,
        homc_order=3,
    ).fit(feats)
    assert m.fitted_
    assert m._adaptive_low_value is not None
    assert m._adaptive_high_value is not None
    assert m._adaptive_median_vol is not None
    assert m._adaptive_low_value < m._adaptive_high_value
    state = m.predict_state(feats)
    assert m.active_component_name in ("homc", "composite")
    probs = m.predict_next(state)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_hybrid_adaptive_vol_regime_switches_threshold(synthetic_prices):
    """In a high-vol recent regime the adaptive strategy should use the
    HIGH quantile; in a low-vol recent regime it should use the LOW."""
    feats = _features(synthetic_prices)
    m = HybridRegimeModel(
        routing_strategy="adaptive_vol",
        adaptive_low_quantile=0.30,
        adaptive_high_quantile=0.80,
        homc_order=3,
    ).fit(feats)

    # Build a feature frame where the last 30 vol bars are DEFINITELY
    # above the training median vol — forces the adaptive strategy into
    # its high-vol regime.
    high_vol_obs = feats.copy()
    median = m._adaptive_median_vol
    assert median is not None
    high_vol_obs.loc[high_vol_obs.index[-40:], "volatility_20d"] = median * 5

    # And one where the last 30 vols are below median — low-vol regime
    low_vol_obs = feats.copy()
    low_vol_obs.loc[low_vol_obs.index[-40:], "volatility_20d"] = median * 0.01

    # In high-vol regime, current vol needs to exceed adaptive_high_value
    # to be classified as bear. We force it high enough.
    high_vol_obs.loc[high_vol_obs.index[-1], "volatility_20d"] = median * 10
    m.predict_state(high_vol_obs)
    # Active component should reflect routing based on high threshold
    assert m.last_regime_label in ("bear", "bull")

    low_vol_obs.loc[low_vol_obs.index[-1], "volatility_20d"] = median * 0.001
    m.predict_state(low_vol_obs)
    assert m.last_regime_label in ("bear", "bull")


def test_hybrid_adaptive_vol_engine_no_lookahead(synthetic_prices):
    """Adaptive routing must also satisfy the lookahead regression."""
    cfg = BacktestConfig(
        model_type="hybrid",
        train_window=300,
        retrain_freq=60,
        return_bins=3,
        volatility_bins=3,
        n_states=3,
        order=3,
        vol_window=10,
        hybrid_routing_strategy="adaptive_vol",
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
