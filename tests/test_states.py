"""Tests for state encoders (used internally by HOMC)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.states import CompositeStateEncoder, QuantileStateEncoder


def _features(prices):
    f = pd.DataFrame(index=prices.index)
    f["return_1d"] = log_returns(prices["close"])
    f["volatility"] = rolling_volatility(f["return_1d"], window=20)
    return f.dropna()


def test_quantile_encoder_partitions_evenly(synthetic_prices):
    feats = _features(synthetic_prices)
    enc = QuantileStateEncoder(n_bins=5)
    states = enc.fit_transform(feats).dropna().astype(int)
    assert set(states.unique()) == {0, 1, 2, 3, 4}
    fractions = states.value_counts(normalize=True)
    assert (fractions > 0.15).all() and (fractions < 0.25).all()


def test_quantile_encoder_deterministic(synthetic_prices):
    feats = _features(synthetic_prices)
    enc1 = QuantileStateEncoder(n_bins=5).fit(feats)
    enc2 = QuantileStateEncoder(n_bins=5).fit(feats)
    np.testing.assert_array_equal(enc1.edges_, enc2.edges_)


def test_quantile_encoder_label_5bins():
    enc = QuantileStateEncoder(n_bins=5)
    assert enc.label(0) == "deep-bear"
    assert enc.label(2) == "neutral"
    assert enc.label(4) == "deep-bull"


def test_composite_encoder_produces_9_states(synthetic_prices):
    feats = _features(synthetic_prices)
    enc = CompositeStateEncoder(return_bins=3, volatility_bins=3)
    states = enc.fit_transform(feats).dropna().astype(int)
    assert states.min() >= 0
    assert states.max() < 9
    assert enc.n_states == 9


def test_composite_encoder_labels():
    enc = CompositeStateEncoder(return_bins=3, volatility_bins=3)
    # state 0 = (return bin 0, vol bin 0) = bear-calm
    assert enc.label(0) == "bear-calm"
    assert enc.label(8) == "bull-panic"
