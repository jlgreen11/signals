"""Sunset warnings for the Markov model classes.

Round 4: the Markov path is sunset but not deleted. Direct usage of
HigherOrderMarkovChain / CompositeMarkovChain / HiddenMarkovModel must
emit a DeprecationWarning pointing the user to the production hybrid
bundle or a vol filter alternative. Usage *inside* `HybridRegimeModel`
must NOT warn — the hybrid is the endorsed production path and still
the project's best-known Sharpe.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth_df():
    rng = np.random.default_rng(42)
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "return_1d": rng.normal(0.0, 0.03, n),
        "volatility": rng.uniform(0.01, 0.05, n),
    }, index=idx)


def test_homc_direct_instantiation_warns():
    from signals.model.homc import HigherOrderMarkovChain
    with pytest.warns(DeprecationWarning, match="SUNSET"):
        HigherOrderMarkovChain()


def test_composite_direct_instantiation_warns():
    from signals.model.composite import CompositeMarkovChain
    with pytest.warns(DeprecationWarning, match="SUNSET"):
        CompositeMarkovChain()


def test_hmm_direct_instantiation_warns():
    from signals.model.hmm import HiddenMarkovModel
    with pytest.warns(DeprecationWarning, match="SUNSET"):
        HiddenMarkovModel()


def test_hybrid_fit_does_not_emit_sunset_warning(synth_df):
    """The hybrid legitimately uses HOMC/composite internally. Those
    instantiations must NOT emit the sunset warning."""
    from signals.model.hybrid import HybridRegimeModel

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = HybridRegimeModel(
            routing_strategy="vol",
            homc_n_states=5,
            homc_order=3,
        )
        h.fit(synth_df)

    sunset_warns = [w for w in caught if "SUNSET" in str(w.message)]
    assert sunset_warns == [], (
        f"Hybrid fit should not emit sunset warnings, got {len(sunset_warns)}: "
        f"{[str(w.message)[:80] for w in sunset_warns]}"
    )


def test_sunset_suppression_flag_restores_after_hybrid_fit(synth_df):
    """The hybrid sets the module-level suppression flag during fit.
    Verify it's restored afterward so subsequent direct instantiation
    still warns."""
    from signals.model import homc as homc_module
    from signals.model.hybrid import HybridRegimeModel

    assert not homc_module._SUPPRESS_SUNSET_WARNING, (
        "suppression flag should start False"
    )
    h = HybridRegimeModel(routing_strategy="vol", homc_n_states=5, homc_order=3)
    h.fit(synth_df)
    assert not homc_module._SUPPRESS_SUNSET_WARNING, (
        "suppression flag should be restored to False after fit"
    )
    # And direct instantiation after the hybrid fit still warns.
    with pytest.warns(DeprecationWarning, match="SUNSET"):
        from signals.model.homc import HigherOrderMarkovChain
        HigherOrderMarkovChain()


def test_sunset_suppression_flag_restores_after_hybrid_fit_exception(synth_df):
    """If the hybrid's fit raises, the suppression flag must still be
    restored (finally clause)."""
    from signals.model import homc as homc_module
    from signals.model.hybrid import HybridRegimeModel

    h = HybridRegimeModel(routing_strategy="vol", homc_n_states=5, homc_order=3)
    # Pass an empty DataFrame to trigger a fit error.
    empty = pd.DataFrame(columns=["return_1d", "volatility"])
    with pytest.raises(Exception):  # noqa: B017
        h.fit(empty)
    assert not homc_module._SUPPRESS_SUNSET_WARNING, (
        "suppression flag should be restored even when fit raises"
    )
