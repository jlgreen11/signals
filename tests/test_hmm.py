"""Tests for HiddenMarkovModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.hmm import HiddenMarkovModel


def _features(prices: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=prices.index)
    f["return_1d"] = log_returns(prices["close"])
    f["volatility"] = rolling_volatility(f["return_1d"], window=20)
    return f.dropna()


def test_hmm_fits_and_decodes(synthetic_prices):
    feats = _features(synthetic_prices)
    hmm = HiddenMarkovModel(n_states=3, n_iter=50).fit(
        feats, feature_cols=["return_1d", "volatility"]
    )
    assert hmm.fitted_
    assert hmm.transmat_.shape == (3, 3)
    np.testing.assert_allclose(hmm.transmat_.sum(axis=1), 1.0, atol=1e-6)
    assert hmm.steady_state().sum() == pytest.approx(1.0, abs=1e-6)
    state = hmm.predict_state(feats)
    assert 0 <= state < 3


def test_hmm_predict_next_distribution(synthetic_prices):
    feats = _features(synthetic_prices)
    hmm = HiddenMarkovModel(n_states=3, n_iter=50).fit(
        feats, feature_cols=["return_1d", "volatility"]
    )
    probs = hmm.predict_next(0)
    assert probs.shape == (3,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_hmm_save_load(tmp_path, synthetic_prices):
    feats = _features(synthetic_prices)
    hmm = HiddenMarkovModel(n_states=3, n_iter=50).fit(
        feats, feature_cols=["return_1d", "volatility"]
    )
    path = tmp_path / "hmm.pkl"
    hmm.save(path)
    hmm2 = HiddenMarkovModel.load(path)
    np.testing.assert_array_equal(hmm.transmat_, hmm2.transmat_)
    np.testing.assert_array_equal(hmm.state_returns_, hmm2.state_returns_)
    assert hmm2.fitted_
    assert hmm2.predict_state(feats) == hmm.predict_state(feats)


def test_hmm_label_orders_by_return():
    hmm = HiddenMarkovModel(n_states=3)
    hmm.state_returns_ = np.array([0.01, -0.01, 0.0])
    hmm.fitted_ = True
    # Sorted ascending: index 1 (-0.01) → bear, index 2 (0.0) → neutral, index 0 (0.01) → bull
    assert hmm.label(1) == "bear"
    assert hmm.label(2) == "neutral"
    assert hmm.label(0) == "bull"


def test_hmm_n_step_distribution(synthetic_prices):
    feats = _features(synthetic_prices)
    hmm = HiddenMarkovModel(n_states=3, n_iter=50).fit(
        feats, feature_cols=["return_1d", "volatility"]
    )
    p3 = hmm.n_step(0, n=3)
    assert p3.sum() == pytest.approx(1.0, abs=1e-6)


def test_hmm_n_init_keeps_best_ll(synthetic_prices):
    """Multi-start HMM should never produce a worse LL than single-start
    on the same seed range."""
    feats = _features(synthetic_prices)
    single = HiddenMarkovModel(n_states=3, n_iter=50, n_init=1).fit(
        feats, feature_cols=["return_1d", "volatility"]
    )
    multi = HiddenMarkovModel(n_states=3, n_iter=50, n_init=4).fit(
        feats, feature_cols=["return_1d", "volatility"]
    )
    # Multi-start covers the single-start seed and at least 3 others, so its
    # best LL must be >= the single-start LL.
    assert multi.log_likelihood_ is not None
    assert single.log_likelihood_ is not None
    assert multi.log_likelihood_ >= single.log_likelihood_ - 1e-9


def test_hmm_strict_convergence_raises_when_too_few_iters(synthetic_prices):
    """If strict_convergence is on and EM bails out before converging,
    fit() should raise rather than silently return a half-fit model."""
    feats = _features(synthetic_prices)
    # Force non-convergence: tight tolerance + only 2 EM iterations.
    hmm = HiddenMarkovModel(
        n_states=4,
        n_iter=2,
        tol=1e-15,
        strict_convergence=True,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        hmm.fit(feats, feature_cols=["return_1d", "volatility"])
