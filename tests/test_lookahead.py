"""Strict no-lookahead regression tests.

If extending the input past bar N changes anything in the output up to bar N,
some part of the feature/encoder/model/signal pipeline is using future data.
These tests are the cheapest possible canary for the entire class of
data-leakage bugs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility


def test_log_returns_no_lookahead(synthetic_prices):
    """Extending input must not change historical return values."""
    cutoff = 400
    short = log_returns(synthetic_prices["close"].iloc[:cutoff])
    long_ = log_returns(synthetic_prices["close"])
    pd.testing.assert_series_equal(
        short,
        long_.iloc[:cutoff],
        check_names=False,
    )


def test_rolling_volatility_no_lookahead(synthetic_prices):
    """Rolling vol must use right-aligned windows only — no centered windows."""
    rets = log_returns(synthetic_prices["close"])
    cutoff = 400
    short = rolling_volatility(rets.iloc[:cutoff], window=10)
    long_ = rolling_volatility(rets, window=10)
    pd.testing.assert_series_equal(
        short,
        long_.iloc[:cutoff],
        check_names=False,
    )


@pytest.mark.parametrize(
    "model_type,extra_kwargs",
    [
        ("composite", {"return_bins": 3, "volatility_bins": 3}),
        ("homc", {"n_states": 3, "order": 2}),
        ("hmm", {"n_states": 3, "n_iter": 30}),
    ],
)
def test_engine_no_lookahead(synthetic_prices, model_type, extra_kwargs):
    """Canonical lookahead regression test.

    The equity curve up to bar N must be bit-identical regardless of whether
    the engine was given prices[:N] or prices[:N+150]. Any leakage in
    features, encoding, model fitting, or signal generation will break this.
    """
    cfg_short = BacktestConfig(
        model_type=model_type,
        train_window=200,
        retrain_freq=40,
        **extra_kwargs,
    )
    cfg_long = BacktestConfig(
        model_type=model_type,
        train_window=200,
        retrain_freq=40,
        **extra_kwargs,
    )

    cutoff = 450
    short_result = BacktestEngine(cfg_short).run(
        synthetic_prices.iloc[:cutoff], symbol="TEST"
    )
    long_result = BacktestEngine(cfg_long).run(synthetic_prices, symbol="TEST")

    # Drop the final two bars of the short run: BacktestEngine.run() flattens
    # the position at the end of the input, which artificially perturbs equity
    # at the truncation point. Everything before that should be identical.
    short_eq = short_result.equity_curve.iloc[:-2]
    common = short_eq.index.intersection(long_result.equity_curve.index)
    assert len(common) > 100, f"too few common bars to validate ({len(common)})"

    pd.testing.assert_series_equal(
        short_eq.loc[common],
        long_result.equity_curve.loc[common],
        check_names=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_composite_encoder_only_uses_train_window(synthetic_prices):
    """The CompositeStateEncoder must compute its quantile bins from the
    training window passed in — never from the full feature DataFrame."""
    from signals.features.returns import log_returns
    from signals.features.volatility import rolling_volatility
    from signals.model.composite import CompositeMarkovChain

    feats = pd.DataFrame(index=synthetic_prices.index)
    feats["return_1d"] = log_returns(synthetic_prices["close"])
    feats["volatility"] = rolling_volatility(feats["return_1d"], window=10)
    feats = feats.dropna()

    train_short = feats.iloc[:300]
    train_full = feats

    m_short = CompositeMarkovChain(return_bins=3, volatility_bins=3, alpha=0.5)
    m_short.fit(
        train_short,
        return_feature="return_1d",
        volatility_feature="volatility",
        return_col="return_1d",
    )
    m_full = CompositeMarkovChain(return_bins=3, volatility_bins=3, alpha=0.5)
    m_full.fit(
        train_full,
        return_feature="return_1d",
        volatility_feature="volatility",
        return_col="return_1d",
    )

    # If the encoder were leaking the future, fitting on `train_full` would
    # produce different bin edges from `train_short` — and we'd see the same
    # quantile edges produced regardless of input. Here we want the opposite:
    # the edges should differ, proving that fit() actually consumed only its
    # input.
    enc_short = m_short._encoder
    enc_full = m_full._encoder
    assert not (
        (enc_short.return_edges_ == enc_full.return_edges_).all()
        and (enc_short.vol_edges_ == enc_full.vol_edges_).all()
    ), "encoder edges identical for different inputs — fit() may be ignoring its argument"
