"""Model package: composite + HMM + HOMC backends with shared signal interface."""

from signals.model.composite import CompositeMarkovChain
from signals.model.hmm import HiddenMarkovModel
from signals.model.homc import HigherOrderMarkovChain
from signals.model.signals import Signal, SignalDecision, SignalGenerator
from signals.model.states import CompositeStateEncoder, QuantileStateEncoder

__all__ = [
    "CompositeMarkovChain",
    "CompositeStateEncoder",
    "HiddenMarkovModel",
    "HigherOrderMarkovChain",
    "QuantileStateEncoder",
    "Signal",
    "SignalDecision",
    "SignalGenerator",
]
