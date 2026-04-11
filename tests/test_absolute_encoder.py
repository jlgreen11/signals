"""Tests for AbsoluteGranularityEncoder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.model.states import AbsoluteGranularityEncoder


def _fake_returns(seed: int = 42, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"return_1d": rng.normal(0.0, 0.03, n)}, index=idx)


def test_absolute_encoder_fit_produces_symmetric_edges():
    df = _fake_returns()
    enc = AbsoluteGranularityEncoder(bin_width=0.01)
    enc.fit(df)
    # Bin boundaries should be integer multiples of bin_width.
    interior = enc.edges_[1:-1]
    ratios = interior / 0.01
    assert np.allclose(ratios, np.round(ratios)), \
        f"interior edges must be integer multiples of bin_width: {interior}"
    # n_states matches len(edges) - 1.
    assert enc.n_states == len(enc.edges_) - 1


def test_absolute_encoder_bin_width_controls_granularity():
    df = _fake_returns()
    wide = AbsoluteGranularityEncoder(bin_width=0.02).fit(df)
    narrow = AbsoluteGranularityEncoder(bin_width=0.005).fit(df)
    # Narrower bins → more bins.
    assert narrow.n_states > wide.n_states


def test_absolute_encoder_transform_round_trip_is_stable():
    df = _fake_returns()
    enc = AbsoluteGranularityEncoder(bin_width=0.01)
    s1 = enc.fit_transform(df)
    s2 = enc.transform(df)
    pd.testing.assert_series_equal(s1, s2)


def test_absolute_encoder_oob_values_are_clipped_to_extremes():
    df = _fake_returns()
    enc = AbsoluteGranularityEncoder(bin_width=0.01, buffer_bins=1)
    enc.fit(df)
    # Inject an extreme value far outside the training range.
    df2 = df.copy()
    df2.loc[df2.index[0], "return_1d"] = 10.0  # 1000% return, way beyond any bin
    states = enc.transform(df2)
    # The extreme value maps to the top bin (n_states - 1) due to clipping.
    assert int(states.iloc[0]) == enc.n_states - 1


def test_absolute_encoder_zero_centred():
    """A value of 0.0 should fall on a bin boundary, not be misplaced."""
    df = _fake_returns()
    enc = AbsoluteGranularityEncoder(bin_width=0.01)
    enc.fit(df)
    # The boundary at 0.0 should exist in interior edges (since fit data
    # includes returns near zero).
    assert 0.0 in enc.edges_[1:-1]


def test_absolute_encoder_label_shows_range():
    df = _fake_returns()
    enc = AbsoluteGranularityEncoder(bin_width=0.01)
    enc.fit(df)
    mid_state = enc.n_states // 2
    label = enc.label(mid_state)
    assert "%" in label, f"label should include percentages: {label}"


def test_absolute_encoder_deterministic_across_reruns():
    df = _fake_returns()
    e1 = AbsoluteGranularityEncoder(bin_width=0.01).fit(df)
    e2 = AbsoluteGranularityEncoder(bin_width=0.01).fit(df)
    np.testing.assert_array_equal(e1.edges_, e2.edges_)
    assert e1.n_states == e2.n_states


def test_absolute_encoder_plumbed_through_homc():
    """HigherOrderMarkovChain should accept the absolute encoder and
    rewrite n_states to match the encoder's actual bin count."""
    from signals.model.homc import HigherOrderMarkovChain

    df = _fake_returns()
    # Pass a custom encoder with a specific bin width.
    enc = AbsoluteGranularityEncoder(bin_width=0.01)
    model = HigherOrderMarkovChain(
        n_states=5,     # nominal; will be rewritten
        order=3,
        alpha=1.0,
        encoder=enc,
    )
    model.fit(df, feature_col="return_1d", return_col="return_1d")
    # n_states should now match the absolute encoder's actual bin count,
    # NOT the initial nominal value 5.
    assert model.n_states == enc.n_states
    assert model.n_states > 5, (
        "absolute encoder at 0.01 bin width should produce more than 5 "
        "states on N(0, 0.03) returns"
    )
    # Transitions and state_returns arrays should be correctly sized.
    assert model.state_returns_.shape == (model.n_states,)
    assert all(len(probs) == model.n_states for probs in model.transitions_.values())
