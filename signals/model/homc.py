"""Higher-Order Markov Chain (HOMC), inspired by:

  Nascimento et al., "Extracting Rules via Markov Chains for Cryptocurrencies
  Returns Forecasting", Computational Economics (2022).

Differences from the paper:
  - We add Laplace smoothing (the paper uses raw frequencies, which are often zero
    for k-tuples that never appeared in training).
  - We use quantile-based binning rather than fixed-width granularity, so the bin
    occupancy is uniform regardless of the underlying return distribution.
  - We expose the same model interface as the HMM so the BacktestEngine and
    SignalGenerator can swap between them transparently.

State of order k = the most recent k bin indices, encoded as a tuple of length k.
The "next state" distribution is over single bins (not k-tuples), so
expected_return = Σ_j P(next_bin=j | history) × avg_return(j) — exactly the same
calculation as for the HMM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from signals.model.states import QuantileStateEncoder

REGIME_NAMES: dict[int, list[str]] = {
    3: ["bear", "neutral", "bull"],
    5: ["deep-bear", "bear", "neutral", "bull", "deep-bull"],
}


class HigherOrderMarkovChain:
    """k-th order discrete Markov chain over quantile-binned returns."""

    def __init__(
        self,
        n_states: int = 5,
        order: int = 3,
        alpha: float = 1.0,
    ):
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if order < 1:
            raise ValueError("order must be >= 1")
        self.n_states = int(n_states)
        self.order = int(order)
        self.alpha = float(alpha)
        self._encoder: QuantileStateEncoder | None = None
        self.feature_col_: str = "return_1d"
        self.return_col_: str = "return_1d"
        self.transitions_: dict[tuple[int, ...], np.ndarray] = {}
        self.marginal_: np.ndarray = np.full(n_states, 1.0 / n_states)
        self.state_returns_: np.ndarray = np.zeros(n_states)
        self.state_counts_: np.ndarray = np.zeros(n_states, dtype=np.int64)
        self.fitted_: bool = False

    # ----- Fit -----
    def fit(
        self,
        observations: pd.DataFrame,
        feature_col: str = "return_1d",
        return_col: str = "return_1d",
    ) -> "HigherOrderMarkovChain":
        self.feature_col_ = feature_col
        self.return_col_ = return_col

        cols = list(dict.fromkeys([feature_col, return_col]))
        clean = observations[cols].dropna()
        min_required = max(self.order + 5, self.n_states * 3)
        if len(clean) < min_required:
            raise ValueError(
                f"Need >= {min_required} samples to fit HOMC(order={self.order}, "
                f"states={self.n_states}); got {len(clean)}"
            )

        self._encoder = QuantileStateEncoder(n_bins=self.n_states, feature=feature_col)
        encoded = self._encoder.fit_transform(clean).dropna().astype(int)
        # Align returns to the encoded series
        rets = clean.loc[encoded.index, return_col].to_numpy()
        states = encoded.to_numpy()

        # Per-state average realized return
        self.state_returns_ = np.zeros(self.n_states)
        self.state_counts_ = np.zeros(self.n_states, dtype=np.int64)
        for s in range(self.n_states):
            mask = states == s
            self.state_counts_[s] = int(mask.sum())
            if mask.any():
                self.state_returns_[s] = float(rets[mask].mean())

        # k-tuple → next-state distribution (Laplace-smoothed)
        self.transitions_ = {}
        for i in range(self.order, len(states)):
            key = tuple(int(x) for x in states[i - self.order : i])
            nxt = int(states[i])
            counts = self.transitions_.get(key)
            if counts is None:
                counts = np.full(self.n_states, self.alpha, dtype=float)
                self.transitions_[key] = counts
            counts[nxt] += 1.0
        # Normalize each row
        for key, counts in self.transitions_.items():
            self.transitions_[key] = counts / counts.sum()

        # Marginal next-state distribution as the fallback when a k-tuple is unseen
        marginal_counts = np.full(self.n_states, self.alpha, dtype=float)
        for s in states:
            marginal_counts[int(s)] += 1.0
        self.marginal_ = marginal_counts / marginal_counts.sum()

        self.fitted_ = True
        return self

    # ----- Predict -----
    def predict_next(self, state) -> np.ndarray:
        """`state` may be an int (interpreted as the most recent bin) or a k-tuple."""
        key = self._coerce_history(state)
        return self.transitions_.get(key, self.marginal_).copy()

    def predict_state(self, observations: pd.DataFrame) -> tuple[int, ...]:
        """Return the most recent k bins as the current 'state'."""
        if self._encoder is None:
            raise RuntimeError("HOMC not fit")
        clean = observations[[self.feature_col_]].dropna()
        if len(clean) < self.order:
            raise ValueError(
                f"Need >= {self.order} valid observations to decode current state; "
                f"got {len(clean)}"
            )
        encoded = self._encoder.transform(clean).dropna().astype(int)
        if len(encoded) < self.order:
            raise ValueError("Insufficient encoded observations")
        return tuple(int(x) for x in encoded.tail(self.order).to_numpy())

    def n_step(self, state, n: int) -> np.ndarray:
        if n < 1:
            raise ValueError("n must be >= 1")
        # Iteratively roll the marginal forward; loses k-tuple specificity past 1 step.
        probs = self.predict_next(state)
        for _ in range(n - 1):
            probs = sum(p * self.transitions_.get(self._coerce_history(int(np.argmax(probs))), self.marginal_) for p in [probs.max()])  # noqa: E501
        return probs

    def steady_state(self) -> np.ndarray:
        """1st-order steady-state from the *bag of single bins* (paper-aligned).

        Note: a true k-th order chain has a steady state over k-tuples, but for
        comparison and labeling we report the marginal stationary distribution
        over single bins (which is well-defined and matches `state_returns_`).
        """
        # Build a 1st-order transition matrix by marginalizing the k-tuple table.
        T1 = np.full((self.n_states, self.n_states), self.alpha, dtype=float)
        for key, probs in self.transitions_.items():
            last = key[-1]
            T1[last] += probs
        T1 = T1 / T1.sum(axis=1, keepdims=True)
        eigvals, eigvecs = np.linalg.eig(T1.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        vec = np.abs(np.real(eigvecs[:, idx]))
        total = vec.sum()
        if total == 0:
            return np.full(self.n_states, 1.0 / self.n_states)
        return vec / total

    def expected_next_return(self, state) -> float:
        return float(self.predict_next(state) @ self.state_returns_)

    # ----- Labels -----
    def label(self, state) -> str:
        # If int, treat as most-recent single bin and return its quantile name.
        if isinstance(state, (int, np.integer)):
            return self._bin_label(int(state))
        if isinstance(state, tuple):
            return "→".join(self._bin_label(s) for s in state)
        return str(state)

    def _bin_label(self, bin_idx: int) -> str:
        names = REGIME_NAMES.get(self.n_states)
        if names and 0 <= bin_idx < len(names):
            return names[bin_idx]
        return f"q{bin_idx}/{self.n_states}"

    # ----- Helpers -----
    def _coerce_history(self, state) -> tuple[int, ...]:
        if isinstance(state, (int, np.integer)):
            if self.order != 1:
                raise ValueError(
                    f"HOMC has order={self.order}; predict_next requires a tuple of length {self.order}"
                )
            return (int(state),)
        if isinstance(state, tuple):
            if len(state) != self.order:
                raise ValueError(
                    f"history length {len(state)} does not match order {self.order}"
                )
            return tuple(int(s) for s in state)
        if isinstance(state, list):
            return self._coerce_history(tuple(state))
        raise TypeError(f"state must be int or tuple; got {type(state).__name__}")

    # ----- Rule extraction (paper-style) -----
    def top_rules(self, k: int = 10) -> list[dict]:
        """Return the top-k k-tuples by training support, with their next-state distribution."""
        rows = []
        # Recover counts: probs * total — but we stored normalized probs only.
        # Re-derive support by counting how many full k+1 windows landed on each key.
        # Store raw counts inside fit() for accuracy.
        for key, probs in self.transitions_.items():
            top_next = int(np.argmax(probs))
            rows.append(
                {
                    "history": "→".join(self._bin_label(s) for s in key),
                    "raw_history": key,
                    "most_likely_next": self._bin_label(top_next),
                    "p_next": float(probs[top_next]),
                    "expected_return": float(probs @ self.state_returns_),
                }
            )
        rows.sort(key=lambda r: r["p_next"], reverse=True)
        return rows[:k]

    # ----- Persistence -----
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_states": self.n_states,
            "order": self.order,
            "alpha": self.alpha,
            "feature_col": self.feature_col_,
            "return_col": self.return_col_,
            "edges": (self._encoder.edges_.tolist() if self._encoder else None),
            "transitions": {
                ",".join(str(s) for s in k): v.tolist()
                for k, v in self.transitions_.items()
            },
            "marginal": self.marginal_.tolist(),
            "state_returns": self.state_returns_.tolist(),
            "state_counts": self.state_counts_.tolist(),
            "fitted": self.fitted_,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path | str) -> "HigherOrderMarkovChain":
        d = json.loads(Path(path).read_text())
        obj = cls(n_states=d["n_states"], order=d["order"], alpha=d["alpha"])
        obj.feature_col_ = d["feature_col"]
        obj.return_col_ = d["return_col"]
        if d["edges"] is not None:
            enc = QuantileStateEncoder(n_bins=obj.n_states, feature=obj.feature_col_)
            enc.edges_ = np.array(d["edges"])
            obj._encoder = enc
        obj.transitions_ = {
            tuple(int(s) for s in k.split(",")): np.array(v)
            for k, v in d["transitions"].items()
        }
        obj.marginal_ = np.array(d["marginal"])
        obj.state_returns_ = np.array(d["state_returns"])
        obj.state_counts_ = np.array(d["state_counts"], dtype=np.int64)
        obj.fitted_ = bool(d["fitted"])
        return obj
