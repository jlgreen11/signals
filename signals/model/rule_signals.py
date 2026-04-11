"""RuleBasedSignalGenerator — Nascimento-style rule extraction.

IMPROVEMENTS.md Tier-1 #4 / SKEPTIC_REVIEW follow-up Experiment 2.

The project's `SignalGenerator` (in `signals/model/signals.py`) trades
the full marginal expectation from every Markov-chain state. The
Nascimento et al. (2022) paper's actual proposal is *rule extraction*:
trade only when the current k-tuple matches one of the most frequent
training rules with a strong directional consensus. Otherwise HOLD.

This generator implements that approach:

  1. At fit time (via `attach_rules`), freeze the top-K rules from the
     chain's transitions table, ranked by training support (how many
     times the k-tuple appeared). For each rule, compute P(next_up) =
     sum of probabilities for states with positive avg return.
  2. At signal time, look up the current k-tuple. If it's in the
     top-K AND P(next_up) >= p_threshold, emit BUY. If it's in the
     top-K AND P(next_down) >= p_threshold, emit SELL. Otherwise
     HOLD (target = 0 or hold-preserves-position depending on engine
     config).

Unlike SignalGenerator, this class does NOT use a dot product with
state_returns_ — it uses a rule-match-then-vote decision. That's the
qualitative difference the Nascimento paper hinges on: most of the
time you don't trade, because most k-tuples you observe at inference
aren't in your top-K training set.

Implements the same interface as SignalGenerator so the engine can
swap it in transparently. The engine wires SignalGenerator at the
signal-generation call site; to use this class, you need to either
patch the engine or run the rule-based evaluation in a dedicated
script that bypasses the default generator.

Parameters
----------
model : HigherOrderMarkovChain
    A fitted HOMC instance. The generator extracts rules from its
    `transitions_` + `state_returns_` arrays.
top_k : int
    How many top-supported rules to extract. Nascimento paper uses
    10-20 depending on order. Default 20.
p_threshold : float
    Minimum direction probability to trigger a trade. Default 0.7
    (matches the paper).
max_long : float
    Position size on BUY. Default 1.0.
max_short : float
    Position size on SELL. Default 1.0. Only used if `allow_short`.
allow_short : bool
    Whether SELL on a matching rule emits short (vs flat). Default
    False, matching the rest of the project.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from signals.model.signals import Signal, SignalDecision


class RuleBasedSignalGenerator:
    """Emit BUY/SELL only when the current k-tuple matches a top-K rule.

    Interface-compatible with `SignalGenerator.generate(state)`.
    """

    def __init__(
        self,
        model: Any,
        top_k: int = 20,
        p_threshold: float = 0.7,
        max_long: float = 1.0,
        max_short: float = 1.0,
        allow_short: bool = False,
    ):
        if not 0.5 <= p_threshold <= 1.0:
            raise ValueError("p_threshold must be in [0.5, 1.0]")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.model = model
        self.top_k = int(top_k)
        self.p_threshold = float(p_threshold)
        self.max_long = float(max_long)
        self.max_short = float(max_short)
        self.allow_short = bool(allow_short)
        self._rules: dict[tuple[int, ...], dict] = {}
        self._rules_ready = False

    def _extract_rules(self) -> None:
        """Freeze the top-K rules from the model's current transitions.

        Rules are ranked by the count of how often their k-tuple
        appeared in training — i.e. by support. Since HOMC stores
        normalized probability rows rather than raw counts, we
        approximate support by how many times the counts differ from
        the uniform Laplace prior — which isn't perfect but is
        monotonic in support for reasonable alpha values. A cleaner
        fix would be to store raw counts at fit time; a follow-up can
        do that if rule extraction becomes load-bearing.

        Direction probabilities P(up) and P(down) are computed as the
        total probability mass on next-states whose avg realized
        return is strictly positive or negative, respectively.
        """
        model = self.model
        if not getattr(model, "fitted_", False):
            raise RuntimeError("model must be fit before extracting rules")
        if not hasattr(model, "transitions_") or not hasattr(model, "state_returns_"):
            raise RuntimeError(
                "RuleBasedSignalGenerator requires a HOMC-style model with "
                "`transitions_` and `state_returns_`"
            )

        state_returns = model.state_returns_
        up_mask = state_returns > 0
        down_mask = state_returns < 0

        # Score each rule by max directional consensus × rough support
        # proxy (distance from uniform prior). A rule that's both
        # frequently observed AND strongly directional scores highest.
        scored: list[tuple[float, tuple[int, ...], np.ndarray, float, float]] = []
        n_states = model.n_states
        uniform = 1.0 / n_states
        for key, probs in model.transitions_.items():
            p_up = float(probs[up_mask].sum()) if up_mask.any() else 0.0
            p_down = float(probs[down_mask].sum()) if down_mask.any() else 0.0
            # Support proxy: L1 distance from uniform (larger = seen more
            # in training). Correct interpretation is conditional on the
            # Laplace alpha used at fit time; this proxy is monotonic in
            # actual support for reasonable alpha < 2.
            support_proxy = float(np.sum(np.abs(probs - uniform)))
            strength = max(p_up, p_down)
            # Rank key: big support AND strong direction dominates.
            score = support_proxy * strength
            scored.append((score, key, probs, p_up, p_down))

        scored.sort(key=lambda t: t[0], reverse=True)
        self._rules = {}
        for score, key, probs, p_up, p_down in scored[: self.top_k]:
            # Expected return across all next-states (for diagnostics only;
            # trading decision uses p_up/p_down only).
            expected = float(np.dot(probs, state_returns))
            self._rules[key] = {
                "probs": probs,
                "p_up": p_up,
                "p_down": p_down,
                "expected_return": expected,
                "score": float(score),
            }
        self._rules_ready = True

    def generate(self, current_state: Any) -> SignalDecision:
        if not getattr(self.model, "fitted_", False):
            raise RuntimeError("model must be fit before generating signals")
        if not self._rules_ready:
            self._extract_rules()

        # Normalize the state to a tuple key
        if isinstance(current_state, tuple):
            key = tuple(int(x) for x in current_state)
        else:
            key = (int(current_state),)

        rule = self._rules.get(key)
        state_label = self.model.label(current_state)

        if rule is None:
            # k-tuple is not in the top-K → HOLD.
            return SignalDecision(
                signal=Signal.HOLD,
                confidence=0.0,
                state=current_state,
                state_label=f"{state_label} (unmatched)",
                expected_return=0.0,
                target_position=0.0,
            )

        p_up = rule["p_up"]
        p_down = rule["p_down"]
        expected = rule["expected_return"]

        if p_up >= self.p_threshold:
            return SignalDecision(
                signal=Signal.BUY,
                confidence=p_up,
                state=current_state,
                state_label=f"{state_label} (rule-buy p_up={p_up:.2f})",
                expected_return=expected,
                target_position=float(self.max_long),
            )
        if p_down >= self.p_threshold:
            target = -float(self.max_short) if self.allow_short else 0.0
            return SignalDecision(
                signal=Signal.SELL,
                confidence=p_down,
                state=current_state,
                state_label=f"{state_label} (rule-sell p_down={p_down:.2f})",
                expected_return=expected,
                target_position=target,
            )
        # Rule matched but no strong direction → HOLD.
        return SignalDecision(
            signal=Signal.HOLD,
            confidence=max(p_up, p_down),
            state=current_state,
            state_label=f"{state_label} (rule-weak p_up={p_up:.2f} p_down={p_down:.2f})",
            expected_return=expected,
            target_position=0.0,
        )
