"""1st-order discrete Markov chain over the CompositeStateEncoder.

This is the original model from Phase 1 (return-bin × volatility-bin states),
re-implemented behind the same interface as HMM and HOMC so that the
BacktestEngine can swap between all three transparently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from signals.model.states import CompositeStateEncoder


class CompositeMarkovChain:
    """1st-order Markov chain over composite (return × volatility) states."""

    def __init__(
        self,
        return_bins: int = 3,
        volatility_bins: int = 3,
        alpha: float = 1.0,
    ):
        if return_bins < 2 or volatility_bins < 2:
            raise ValueError("return_bins and volatility_bins must be >= 2")
        self.return_bins = int(return_bins)
        self.volatility_bins = int(volatility_bins)
        self.n_states = self.return_bins * self.volatility_bins
        self.alpha = float(alpha)
        self._encoder: CompositeStateEncoder | None = None
        self.return_feature_: str = "return_1d"
        self.volatility_feature_: str = "volatility_20d"
        self.return_col_: str = "return_1d"
        self.transmat_: np.ndarray = np.zeros((self.n_states, self.n_states))
        self.state_returns_: np.ndarray = np.zeros(self.n_states)
        self.state_counts_: np.ndarray = np.zeros(self.n_states, dtype=np.int64)
        self.fitted_: bool = False

    # ----- Fit -----
    def fit(
        self,
        observations: pd.DataFrame,
        return_feature: str = "return_1d",
        volatility_feature: str = "volatility_20d",
        return_col: str = "return_1d",
    ) -> "CompositeMarkovChain":
        self.return_feature_ = return_feature
        self.volatility_feature_ = volatility_feature
        self.return_col_ = return_col

        cols = list(dict.fromkeys([return_feature, volatility_feature, return_col]))
        clean = observations[cols].dropna()
        if len(clean) < self.n_states * 3:
            raise ValueError(
                f"Need >= {self.n_states * 3} samples to fit composite chain; got {len(clean)}"
            )

        self._encoder = CompositeStateEncoder(
            return_bins=self.return_bins,
            volatility_bins=self.volatility_bins,
            return_feature=return_feature,
            volatility_feature=volatility_feature,
        )
        encoded = self._encoder.fit_transform(clean).dropna().astype(int)
        rets = clean.loc[encoded.index, return_col].to_numpy()
        states = encoded.to_numpy()

        # Per-state realized return
        self.state_returns_ = np.zeros(self.n_states)
        self.state_counts_ = np.zeros(self.n_states, dtype=np.int64)
        for s in range(self.n_states):
            mask = states == s
            self.state_counts_[s] = int(mask.sum())
            if mask.any():
                self.state_returns_[s] = float(rets[mask].mean())

        # Laplace-smoothed transition counts
        T = np.full((self.n_states, self.n_states), self.alpha, dtype=float)
        for i in range(len(states) - 1):
            T[int(states[i]), int(states[i + 1])] += 1.0
        self.transmat_ = T / T.sum(axis=1, keepdims=True)

        self.fitted_ = True
        return self

    # ----- Predict -----
    def predict_next(self, state) -> np.ndarray:
        s = int(state)
        self._check_state(s)
        return self.transmat_[s].copy()

    def predict_state(self, observations: pd.DataFrame) -> int:
        if self._encoder is None:
            raise RuntimeError("CompositeMarkovChain not fit")
        clean = observations[[self.return_feature_, self.volatility_feature_]].dropna()
        if clean.empty:
            raise ValueError("No valid observations to decode current state")
        encoded = self._encoder.transform(clean).dropna().astype(int)
        if encoded.empty:
            raise ValueError("No encoded observations")
        return int(encoded.iloc[-1])

    def n_step(self, state, n: int) -> np.ndarray:
        if n < 1:
            raise ValueError("n must be >= 1")
        v = np.zeros(self.n_states)
        v[int(state)] = 1.0
        return v @ np.linalg.matrix_power(self.transmat_, n)

    def steady_state(self) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eig(self.transmat_.T)
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
        if self._encoder is None:
            return f"state_{int(state)}"
        return self._encoder.label(int(state))

    def _check_state(self, state: int) -> None:
        if not (0 <= state < self.n_states):
            raise ValueError(f"state {state} out of range [0, {self.n_states - 1}]")

    # ----- Persistence -----
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "return_bins": self.return_bins,
            "volatility_bins": self.volatility_bins,
            "alpha": self.alpha,
            "return_feature": self.return_feature_,
            "volatility_feature": self.volatility_feature_,
            "return_col": self.return_col_,
            "return_edges": (
                self._encoder.return_edges_.tolist()
                if self._encoder and self._encoder.return_edges_ is not None
                else None
            ),
            "vol_edges": (
                self._encoder.vol_edges_.tolist()
                if self._encoder and self._encoder.vol_edges_ is not None
                else None
            ),
            "transmat": self.transmat_.tolist(),
            "state_returns": self.state_returns_.tolist(),
            "state_counts": self.state_counts_.tolist(),
            "fitted": self.fitted_,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path | str) -> "CompositeMarkovChain":
        d = json.loads(Path(path).read_text())
        obj = cls(
            return_bins=d["return_bins"],
            volatility_bins=d["volatility_bins"],
            alpha=d["alpha"],
        )
        obj.return_feature_ = d["return_feature"]
        obj.volatility_feature_ = d["volatility_feature"]
        obj.return_col_ = d["return_col"]
        if d["return_edges"] is not None and d["vol_edges"] is not None:
            enc = CompositeStateEncoder(
                return_bins=obj.return_bins,
                volatility_bins=obj.volatility_bins,
                return_feature=obj.return_feature_,
                volatility_feature=obj.volatility_feature_,
            )
            enc.return_edges_ = np.array(d["return_edges"])
            enc.vol_edges_ = np.array(d["vol_edges"])
            obj._encoder = enc
        obj.transmat_ = np.array(d["transmat"])
        obj.state_returns_ = np.array(d["state_returns"])
        obj.state_counts_ = np.array(d["state_counts"], dtype=np.int64)
        obj.fitted_ = bool(d["fitted"])
        return obj
