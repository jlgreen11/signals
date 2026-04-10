"""Tests for the CompositeMarkovChain (1st-order, 2D state encoder)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.composite import CompositeMarkovChain


def _feats(prices: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=prices.index)
    f["return_1d"] = log_returns(prices["close"])
    f["volatility_20d"] = rolling_volatility(f["return_1d"], window=20)
    return f.dropna()


def test_composite_fit_and_predict_next(synthetic_prices):
    feats = _feats(synthetic_prices)
    m = CompositeMarkovChain(return_bins=3, volatility_bins=3).fit(feats)
    assert m.fitted_
    assert m.n_states == 9
    assert m.transmat_.shape == (9, 9)
    assert np.allclose(m.transmat_.sum(axis=1), 1.0)
    probs = m.predict_next(0)
    assert probs.shape == (9,)
    assert np.isclose(probs.sum(), 1.0)


def test_composite_predict_state(synthetic_prices):
    feats = _feats(synthetic_prices)
    m = CompositeMarkovChain(return_bins=3, volatility_bins=3).fit(feats)
    s = m.predict_state(feats)
    assert isinstance(s, int)
    assert 0 <= s < 9


def test_composite_save_load_roundtrip(tmp_path, synthetic_prices):
    feats = _feats(synthetic_prices)
    m = CompositeMarkovChain(return_bins=3, volatility_bins=3).fit(feats)
    path = tmp_path / "composite.json"
    m.save(path)
    m2 = CompositeMarkovChain.load(path)
    assert m2.n_states == m.n_states
    np.testing.assert_array_almost_equal(m2.transmat_, m.transmat_)
    np.testing.assert_array_almost_equal(m2.state_returns_, m.state_returns_)
    # Decoded current state should match
    s1 = m.predict_state(feats)
    s2 = m2.predict_state(feats)
    assert s1 == s2


def test_composite_steady_state_sums_to_one(synthetic_prices):
    feats = _feats(synthetic_prices)
    m = CompositeMarkovChain(return_bins=3, volatility_bins=3).fit(feats)
    steady = m.steady_state()
    assert np.isclose(steady.sum(), 1.0, atol=1e-6)
