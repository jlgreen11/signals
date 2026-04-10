"""Tests for the SignalGenerator."""

from __future__ import annotations

import numpy as np

from signals.model.signals import Signal, SignalGenerator


class _StubModel:
    """Minimal model implementing the SignalGenerator interface."""

    def __init__(self, state_returns):
        self.n_states = len(state_returns)
        self.state_returns_ = np.array(state_returns)
        self.fitted_ = True

    def predict_next(self, state):
        # Identity transition: each state always returns to itself
        v = np.zeros(self.n_states)
        v[int(state)] = 1.0
        return v

    def label(self, state):
        return f"s{int(state)}"


def test_buy_when_expected_return_above_threshold():
    model = _StubModel([0.01, 0.0, -0.01])
    gen = SignalGenerator(model=model, buy_threshold_bps=20, sell_threshold_bps=-20)
    decision = gen.generate(current_state=0)
    assert decision.signal == Signal.BUY


def test_sell_when_expected_return_below_threshold():
    model = _StubModel([0.01, 0.0, -0.01])
    gen = SignalGenerator(model=model, buy_threshold_bps=20, sell_threshold_bps=-20)
    decision = gen.generate(current_state=2)
    assert decision.signal == Signal.SELL


def test_hold_when_inside_threshold_band():
    model = _StubModel([0.0001, 0.0, -0.0001])
    gen = SignalGenerator(model=model, buy_threshold_bps=20, sell_threshold_bps=-20)
    decision = gen.generate(current_state=1)
    assert decision.signal == Signal.HOLD


def test_signal_decision_carries_label():
    model = _StubModel([0.01, -0.01])
    gen = SignalGenerator(model=model, buy_threshold_bps=20, sell_threshold_bps=-20)
    decision = gen.generate(current_state=0)
    assert decision.state_label == "s0"


def test_target_position_long_when_buy():
    model = _StubModel([0.01, 0.0, -0.01])
    gen = SignalGenerator(
        model=model, buy_threshold_bps=20, sell_threshold_bps=-20,
        target_scale_bps=50.0, allow_short=True,
    )
    decision = gen.generate(current_state=0)
    assert decision.target_position > 0
    assert decision.target_position <= 1.0


def test_target_position_short_when_sell_and_allowed():
    model = _StubModel([0.01, 0.0, -0.01])
    gen = SignalGenerator(
        model=model, buy_threshold_bps=20, sell_threshold_bps=-20,
        allow_short=True,
    )
    decision = gen.generate(current_state=2)
    assert decision.target_position < 0
    assert decision.target_position >= -1.0


def test_target_position_zero_when_short_disallowed():
    model = _StubModel([0.01, 0.0, -0.01])
    gen = SignalGenerator(
        model=model, buy_threshold_bps=20, sell_threshold_bps=-20,
        allow_short=False,
    )
    decision = gen.generate(current_state=2)
    # SELL signal still fires (so an existing long can exit), but target is 0.
    assert decision.signal == Signal.SELL
    assert decision.target_position == 0.0


def test_target_position_zero_on_hold():
    model = _StubModel([0.0001, 0.0, -0.0001])
    gen = SignalGenerator(model=model, buy_threshold_bps=20, sell_threshold_bps=-20)
    decision = gen.generate(current_state=1)
    assert decision.target_position == 0.0
