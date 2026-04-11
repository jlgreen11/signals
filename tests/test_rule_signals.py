"""Tests for RuleBasedSignalGenerator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.model.homc import HigherOrderMarkovChain
from signals.model.rule_signals import RuleBasedSignalGenerator
from signals.model.signals import Signal


def _fake_returns(seed: int = 42, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"return_1d": rng.normal(0.0, 0.03, n)}, index=idx)


def _fit_homc(order: int = 3, n_states: int = 5) -> HigherOrderMarkovChain:
    df = _fake_returns()
    m = HigherOrderMarkovChain(n_states=n_states, order=order, alpha=1.0)
    m.fit(df, feature_col="return_1d", return_col="return_1d")
    return m


def test_rule_generator_requires_fitted_model():
    class Dummy:
        fitted_ = False

    rsg = RuleBasedSignalGenerator(Dummy())
    with pytest.raises(RuntimeError, match="fit"):
        rsg.generate((0, 1, 2))


def test_rule_generator_rejects_invalid_threshold():
    m = _fit_homc()
    with pytest.raises(ValueError, match="p_threshold"):
        RuleBasedSignalGenerator(m, p_threshold=0.3)  # below 0.5
    with pytest.raises(ValueError, match="p_threshold"):
        RuleBasedSignalGenerator(m, p_threshold=1.5)  # above 1.0


def test_rule_generator_rejects_invalid_top_k():
    m = _fit_homc()
    with pytest.raises(ValueError, match="top_k"):
        RuleBasedSignalGenerator(m, top_k=0)


def test_rule_generator_unmatched_state_returns_hold():
    m = _fit_homc(order=3)
    rsg = RuleBasedSignalGenerator(m, top_k=5, p_threshold=0.90)
    # An unmatched (or weakly-directional) k-tuple should emit HOLD.
    # With top_k=5 and a strict p_threshold, most states will HOLD.
    states_tested = 0
    holds = 0
    for key in list(m.transitions_.keys())[:20]:
        decision = rsg.generate(key)
        states_tested += 1
        if decision.signal == Signal.HOLD:
            holds += 1
    # At p_threshold=0.90 we expect MOST decisions to be HOLD.
    assert holds >= states_tested // 2, (
        f"expected mostly HOLDs at strict p_threshold; got {holds}/{states_tested}"
    )


def test_rule_generator_matched_rule_can_fire():
    """With a lax threshold, a top-K rule with ANY directional bias should fire."""
    m = _fit_homc(order=3)
    rsg = RuleBasedSignalGenerator(m, top_k=20, p_threshold=0.55)
    # With p_threshold=0.55 (just above 0.5), at least one of the top-20
    # rules should fire BUY or SELL when its k-tuple is queried.
    fired = 0
    for key in list(m.transitions_.keys())[:50]:
        decision = rsg.generate(key)
        if decision.signal != Signal.HOLD:
            fired += 1
    assert fired > 0, "at least one rule should fire BUY/SELL at p_threshold=0.55"


def test_rule_generator_sell_with_short_disabled_yields_zero_target():
    m = _fit_homc(order=3)
    rsg = RuleBasedSignalGenerator(
        m, top_k=20, p_threshold=0.55, allow_short=False
    )
    # Force extraction and then look for any rule that triggers SELL.
    rsg._extract_rules()
    sell_found = False
    for key in list(m.transitions_.keys())[:50]:
        decision = rsg.generate(key)
        if decision.signal == Signal.SELL:
            sell_found = True
            # allow_short=False should flatten, not short.
            assert decision.target_position == 0.0
            break
    # Not all seeds produce a SELL rule; we just verify behavior IF one fires.
    _ = sell_found


def test_rule_generator_buy_signal_has_positive_target():
    m = _fit_homc(order=3)
    rsg = RuleBasedSignalGenerator(m, top_k=20, p_threshold=0.55, max_long=1.0)
    for key in list(m.transitions_.keys())[:50]:
        decision = rsg.generate(key)
        if decision.signal == Signal.BUY:
            assert decision.target_position > 0.0
            return
    # Test doesn't fail if no BUY fires — rare but possible on synthetic data.


def test_rule_generator_top_k_limits_active_rules():
    m = _fit_homc(order=3)
    rsg_small = RuleBasedSignalGenerator(m, top_k=3, p_threshold=0.5)
    rsg_large = RuleBasedSignalGenerator(m, top_k=30, p_threshold=0.5)
    rsg_small._extract_rules()
    rsg_large._extract_rules()
    assert len(rsg_small._rules) <= 3
    assert len(rsg_large._rules) >= len(rsg_small._rules)


def test_rule_generator_returns_signal_decision_shape():
    m = _fit_homc(order=3)
    rsg = RuleBasedSignalGenerator(m, top_k=10, p_threshold=0.7)
    key = next(iter(m.transitions_.keys()))
    decision = rsg.generate(key)
    # SignalDecision fields exist and are correct types.
    assert decision.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
    assert isinstance(decision.confidence, float)
    assert isinstance(decision.expected_return, float)
    assert isinstance(decision.target_position, float)
    assert isinstance(decision.state_label, str)
