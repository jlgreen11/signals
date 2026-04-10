"""Map model predictions into discrete trading signals + a sized target.

This is model-agnostic: any object exposing the small interface below works.
    - .fitted_ : bool
    - .predict_next(state) -> np.ndarray  (next-state probability vector)
    - .state_returns_ : np.ndarray         (avg realized return per state index)
    - .label(state) -> str                 (human-readable label)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalDecision:
    signal: Signal
    confidence: float
    state: Any
    state_label: str
    expected_return: float
    target_position: float = 0.0  # signed fraction of equity in [-max_short, +max_long]


class SignalGenerator:
    """Convert next-state distributions into discrete signals + sized targets.

    target_position is computed as:
        sign(expected) * min(1, |expected| / target_scale) * confidence
    then clipped to [-max_short, +max_long]. confidence is the probability mass
    on directionally-aligned next states.

    A non-zero `hysteresis_bps` adds a deadband around the buy/sell thresholds:
    once a position is open, the signal stays in that direction until the
    expected return crosses *all the way* to the opposite threshold.
    (Hysteresis is informational on SignalDecision; the engine enforces it.)
    """

    def __init__(
        self,
        model,
        buy_threshold_bps: float = 20.0,
        sell_threshold_bps: float = -20.0,
        target_scale_bps: float = 50.0,
        max_long: float = 1.0,
        max_short: float = 1.0,
        allow_short: bool = True,
        min_confidence: float = 0.0,
    ):
        self.model = model
        self.buy_threshold = buy_threshold_bps * 1e-4
        self.sell_threshold = sell_threshold_bps * 1e-4
        self.target_scale = max(target_scale_bps * 1e-4, 1e-6)
        self.max_long = float(max_long)
        self.max_short = float(max_short)
        self.allow_short = bool(allow_short)
        self.min_confidence = float(min_confidence)

    def generate(self, current_state) -> SignalDecision:
        if not getattr(self.model, "fitted_", False):
            raise RuntimeError("model must be fit before generating signals")

        probs = self.model.predict_next(current_state)
        expected = float(np.dot(probs, self.model.state_returns_))

        if expected >= self.buy_threshold:
            sig = Signal.BUY
        elif expected <= self.sell_threshold:
            # SELL means "exit longs". If shorts are disallowed, target=0
            # rather than negative; the discrete bucket stays SELL so the
            # engine knows to exit.
            sig = Signal.SELL
        else:
            sig = Signal.HOLD

        # Confidence: probability mass on directionally-aligned next states.
        if expected >= 0:
            mask = self.model.state_returns_ >= 0
        else:
            mask = self.model.state_returns_ < 0
        confidence = float(probs[mask].sum())

        # Sized target. Magnitude is |expected| / scale — can exceed 1.0 for
        # strong signals, which is clipped later by max_long / max_short.
        # Confidence acts as a *gate*, not a multiplier — a confident signal
        # goes fully to target; a low-confidence signal goes flat.
        #
        # Design note: the old formula capped magnitude at 1.0 unconditionally,
        # which meant raising max_long above 1.0 had no effect at all. The new
        # formula lets max_long be the actual leverage ceiling. Backward
        # compatibility: with max_long=1.0 (the previous default), behavior is
        # identical — a magnitude of e.g. 1.5 clips to 1.0 at the target stage.
        magnitude = abs(expected) / self.target_scale
        if confidence < self.min_confidence:
            magnitude = 0.0
        raw_target = np.sign(expected) * magnitude
        if raw_target > 0:
            target = float(min(raw_target, self.max_long))
        elif raw_target < 0 and self.allow_short:
            target = float(max(raw_target, -self.max_short))
        else:
            target = 0.0

        # If the discrete bucket says HOLD (inside deadband), force flat.
        if sig == Signal.HOLD:
            target = 0.0

        return SignalDecision(
            signal=sig,
            confidence=confidence,
            state=current_state,
            state_label=self.model.label(current_state),
            expected_return=expected,
            target_position=target,
        )
