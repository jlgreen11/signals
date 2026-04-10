"""Tests for HigherOrderMarkovChain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.homc import HigherOrderMarkovChain


def _features(prices: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=prices.index)
    f["return_1d"] = log_returns(prices["close"])
    f["volatility_20d"] = rolling_volatility(f["return_1d"], window=20)
    return f.dropna()


def test_homc_fits_and_predicts(synthetic_prices):
    feats = _features(synthetic_prices)
    chain = HigherOrderMarkovChain(n_states=5, order=3).fit(feats)
    assert chain.fitted_
    state = chain.predict_state(feats)
    assert isinstance(state, tuple)
    assert len(state) == 3
    assert all(0 <= s < 5 for s in state)


def test_homc_predict_next_sums_to_one(synthetic_prices):
    feats = _features(synthetic_prices)
    chain = HigherOrderMarkovChain(n_states=5, order=2).fit(feats)
    state = chain.predict_state(feats)
    probs = chain.predict_next(state)
    assert probs.shape == (5,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_homc_unseen_history_uses_marginal():
    # Build a tiny dataset where one history is unseen
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, 200))
    df = pd.DataFrame({"return_1d": rets, "close": (1 + rets).cumprod() * 100})
    chain = HigherOrderMarkovChain(n_states=3, order=2).fit(df, feature_col="return_1d")
    impossible = (0, 0, 0)  # wrong length tuple should error
    with pytest.raises(ValueError):
        chain.predict_next(impossible)
    # A 2-tuple that may not have appeared falls back to marginal
    fallback = chain.predict_next((0, 2))
    assert fallback.sum() == pytest.approx(1.0, abs=1e-6)


def test_homc_save_load(tmp_path, synthetic_prices):
    feats = _features(synthetic_prices)
    chain = HigherOrderMarkovChain(n_states=4, order=2).fit(feats)
    path = tmp_path / "homc.json"
    chain.save(path)
    loaded = HigherOrderMarkovChain.load(path)
    assert loaded.n_states == chain.n_states
    assert loaded.order == chain.order
    np.testing.assert_array_equal(loaded.state_returns_, chain.state_returns_)
    np.testing.assert_array_equal(loaded.marginal_, chain.marginal_)
    # Same prediction
    state = chain.predict_state(feats)
    np.testing.assert_array_equal(
        loaded.predict_next(state), chain.predict_next(state)
    )


def test_homc_top_rules(synthetic_prices):
    feats = _features(synthetic_prices)
    chain = HigherOrderMarkovChain(n_states=3, order=2).fit(feats)
    rules = chain.top_rules(k=5)
    assert len(rules) <= 5
    for r in rules:
        assert "history" in r
        assert "p_next" in r
        assert 0.0 <= r["p_next"] <= 1.0


def test_homc_steady_state_sums_to_one(synthetic_prices):
    feats = _features(synthetic_prices)
    chain = HigherOrderMarkovChain(n_states=4, order=2).fit(feats)
    ss = chain.steady_state()
    assert ss.shape == (4,)
    assert ss.sum() == pytest.approx(1.0, abs=1e-6)
