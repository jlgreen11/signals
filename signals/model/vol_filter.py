"""NaiveVolFilter — the null hypothesis baseline for the signals project.

This model answers the question: "Does the Markov chain know something a
simple vol threshold doesn't?"

The logic is trivially simple:
  - Compute trailing realized vol over `vol_window` bars during fit
  - Compute the `quantile`-th percentile of training vol as the threshold
  - At prediction time: if current vol >= threshold, go flat (state 0);
    otherwise, hold long (state 1)

This is EXACTLY what the hybrid's vol-routing does, minus the Markov chain.
If the hybrid doesn't beat this model, the Markov chain adds no value and
the entire composite/HOMC/HMM machinery is decoration on top of a vol filter.

Conforms to the standard model interface so BacktestEngine and SignalGenerator
can swap it in transparently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Saturated state returns so SignalGenerator thresholds always trigger.
# State 0 = high vol (flat/sell), state 1 = low vol (long/buy).
_STATE_RETURNS = np.array([-1.0, +1.0])


class NaiveVolFilter:
    """Go flat when trailing vol is high, hold long otherwise.

    This is the simplest possible implementation of the insight that
    drives the hybrid model's vol routing. If the hybrid can't beat
    this, the Markov chain adds no value.
    """

    def __init__(self, vol_window: int = 10, quantile: float = 0.50):
        if vol_window < 2:
            raise ValueError("vol_window must be >= 2")
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.vol_window = int(vol_window)
        self.quantile = float(quantile)
        self.n_states = 2
        self.state_returns_ = _STATE_RETURNS.copy()
        self.state_counts_: np.ndarray = np.zeros(2, dtype=np.int64)
        self.fitted_: bool = False
        self._vol_threshold: float = 0.0

    def fit(
        self,
        observations: pd.DataFrame,
        **_ignored,
    ) -> NaiveVolFilter:
        """Compute the vol quantile threshold from training data."""
        if "volatility" in observations.columns:
            vols = observations["volatility"].dropna()
        elif "return_1d" in observations.columns:
            vols = (
                observations["return_1d"]
                .rolling(window=self.vol_window, min_periods=self.vol_window)
                .std()
                .dropna()
            )
        else:
            raise ValueError(
                "observations must contain 'volatility' or 'return_1d' column"
            )

        if len(vols) < 10:
            raise ValueError(f"Need >= 10 vol observations to fit; got {len(vols)}")

        self._vol_threshold = float(np.quantile(vols.values, self.quantile))

        # Record state frequencies for introspection
        high_vol = vols >= self._vol_threshold
        self.state_counts_ = np.array(
            [int(high_vol.sum()), int((~high_vol).sum())], dtype=np.int64
        )
        self.fitted_ = True
        return self

    def predict_state(self, observations: pd.DataFrame) -> int:
        """Return 0 (high vol → flat) or 1 (low vol → long)."""
        if not self.fitted_:
            raise RuntimeError("NaiveVolFilter not fit")

        if "volatility" in observations.columns:
            vols = observations["volatility"].dropna()
        elif "return_1d" in observations.columns:
            vols = (
                observations["return_1d"]
                .rolling(window=self.vol_window, min_periods=self.vol_window)
                .std()
                .dropna()
            )
        else:
            raise ValueError(
                "observations must contain 'volatility' or 'return_1d' column"
            )

        if vols.empty:
            return 0  # no data → defensive (flat)

        current_vol = float(vols.iloc[-1])
        return 0 if current_vol >= self._vol_threshold else 1

    def predict_next(self, state: int) -> np.ndarray:
        """Deterministic one-hot: state drives the signal directly."""
        s = int(state)
        if s not in (0, 1):
            raise ValueError(f"state must be 0 or 1; got {s}")
        out = np.zeros(2)
        out[s] = 1.0
        return out

    def expected_next_return(self, state: int) -> float:
        return float(self.state_returns_[int(state)])

    def label(self, state) -> str:
        s = int(state)
        if s == 0:
            return "high_vol_flat"
        if s == 1:
            return "low_vol_long"
        return f"state_{s}"

    @property
    def vol_threshold(self) -> float:
        return self._vol_threshold

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "NaiveVolFilter",
            "vol_window": self.vol_window,
            "quantile": self.quantile,
            "vol_threshold": self._vol_threshold,
            "state_counts": self.state_counts_.tolist(),
            "fitted": self.fitted_,
        }
        p.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path | str) -> NaiveVolFilter:
        d = json.loads(Path(path).read_text())
        obj = cls(vol_window=d["vol_window"], quantile=d["quantile"])
        obj._vol_threshold = float(d["vol_threshold"])
        obj.state_counts_ = np.array(d["state_counts"], dtype=np.int64)
        obj.fitted_ = bool(d["fitted"])
        return obj
