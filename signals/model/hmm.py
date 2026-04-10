"""Gaussian Hidden Markov Model for regime detection.

Discovers latent market regimes from continuous observations (e.g. return + volatility)
using Baum-Welch (EM). At inference time, the most likely current hidden state is
recovered via Viterbi.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class HiddenMarkovModel:
    """Gaussian-emission HMM with a uniform interface for the SignalGenerator.

    Required interface:
        - n_states, fitted_, state_returns_
        - predict_state(observations) -> int   (current latent state)
        - predict_next(state) -> ndarray       (next-state distribution)
        - label(state) -> str
        - steady_state() -> ndarray
        - save / load
    """

    REGIME_NAMES: dict[int, list[str]] = {
        2: ["bear", "bull"],
        3: ["bear", "neutral", "bull"],
        4: ["deep-bear", "weak-bear", "weak-bull", "deep-bull"],
        5: ["deep-bear", "bear", "neutral", "bull", "deep-bull"],
        6: ["deep-bear", "bear", "weak-bear", "weak-bull", "bull", "deep-bull"],
    }

    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 200,
        random_state: int = 42,
        covariance_type: str = "diag",
        n_init: int = 1,
        tol: float = 1e-3,
        strict_convergence: bool = False,
    ):
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.random_state = int(random_state)
        self.covariance_type = covariance_type
        self.n_init = max(1, int(n_init))
        self.tol = float(tol)
        self.strict_convergence = bool(strict_convergence)
        self._model = None
        self._scaler = None  # sklearn StandardScaler, fit on training features
        self.state_returns_: np.ndarray = np.zeros(n_states)
        self.state_counts_: np.ndarray = np.zeros(n_states, dtype=np.int64)
        self.feature_means_: np.ndarray | None = None
        self.feature_cols_: list[str] = []
        self.return_col_: str = "return_1d"
        self.fitted_: bool = False
        self.converged_: bool = False
        self.log_likelihood_: float | None = None

    # ----- Fit -----
    def fit(
        self,
        observations: pd.DataFrame,
        feature_cols: list[str] | None = None,
        return_col: str = "return_1d",
    ) -> HiddenMarkovModel:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler

        if feature_cols is None:
            feature_cols = [c for c in observations.columns if c != return_col]
            if not feature_cols:
                feature_cols = [return_col]
        self.feature_cols_ = list(feature_cols)
        self.return_col_ = return_col

        cols = list(dict.fromkeys(self.feature_cols_ + [return_col]))
        clean = observations[cols].dropna()
        if len(clean) < self.n_states * 5:
            raise ValueError(
                f"Need >= {self.n_states * 5} samples to fit HMM; got {len(clean)}"
            )

        X_raw = clean[self.feature_cols_].to_numpy()
        rets = clean[return_col].to_numpy()

        # Standardize features so Gaussian emissions don't get pulled around by scale.
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X_raw)

        # Multi-start: fit n_init times with distinct seeds, keep the model
        # with the highest log-likelihood. Single restart still costs the same
        # as before (n_init=1 default), so existing callers are unaffected.
        best_model = None
        best_ll = -np.inf
        for trial in range(self.n_init):
            candidate = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state + trial,
                tol=self.tol,
            )
            try:
                candidate.fit(X)
                ll = float(candidate.score(X))
            except Exception:
                continue
            if ll > best_ll:
                best_ll = ll
                best_model = candidate
        if best_model is None:
            raise RuntimeError(
                f"HMM failed to fit on all {self.n_init} restart(s)"
            )
        self._model = best_model
        # hmmlearn's monitor_.converged is True even when EM exhausted n_iter
        # without LL convergence (it conflates "stopped" with "converged"). Do
        # the real check: LL delta on the final step must be below tol.
        history = list(self._model.monitor_.history)
        ll_converged = (
            len(history) >= 2 and (history[-1] - history[-2]) < self.tol
        )
        self.converged_ = ll_converged
        self.log_likelihood_ = best_ll if best_ll > -np.inf else None
        if self.strict_convergence and not self.converged_:
            raise RuntimeError(
                f"HMM did not converge after {self.n_iter} iter × {self.n_init} restart(s); "
                f"increase n_iter, n_init, or relax tol"
            )

        states = self._model.predict(X)
        self.state_returns_ = np.zeros(self.n_states)
        self.state_counts_ = np.zeros(self.n_states, dtype=np.int64)
        # Store *raw* feature means for human interpretation.
        self.feature_means_ = np.zeros((self.n_states, len(self.feature_cols_)))
        for s in range(self.n_states):
            mask = states == s
            self.state_counts_[s] = int(mask.sum())
            if mask.any():
                self.state_returns_[s] = float(rets[mask].mean())
                self.feature_means_[s] = X_raw[mask].mean(axis=0)
        self.fitted_ = True
        return self

    # ----- Predict -----
    @property
    def transmat_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("HMM not fit")
        return self._model.transmat_

    @property
    def T(self) -> np.ndarray:
        return self.transmat_

    def predict_next(self, state: int) -> np.ndarray:
        self._check_state(state)
        return self.transmat_[int(state)].copy()

    def predict_state(self, observations: pd.DataFrame) -> int:
        """Most likely current hidden state via Viterbi over the supplied window."""
        if self._model is None:
            raise RuntimeError("HMM not fit")
        clean = observations[self.feature_cols_].dropna()
        if clean.empty:
            raise ValueError("No valid observations to decode")
        X = clean.to_numpy()
        if self._scaler is not None:
            X = self._scaler.transform(X)
        states = self._model.predict(X)
        return int(states[-1])

    def n_step(self, state: int, n: int) -> np.ndarray:
        self._check_state(state)
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

    def expected_next_return(self, state: int) -> float:
        return float(self.predict_next(state) @ self.state_returns_)

    # ----- Labels -----
    def label(self, state) -> str:
        s = int(state)
        self._check_state(s)
        if not self.fitted_:
            return f"state_{s}"
        # Rank states by avg return: lowest = bear, highest = bull
        ranks = np.argsort(np.argsort(self.state_returns_))
        rank = int(ranks[s])
        names = self.REGIME_NAMES.get(self.n_states)
        if names is None:
            return f"regime_{rank}/{self.n_states}"
        return names[rank]

    def _check_state(self, state) -> None:
        s = int(state)
        if not (0 <= s < self.n_states):
            raise ValueError(f"state {s} out of range [0, {self.n_states - 1}]")

    # ----- Persistence -----
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "n_states": self.n_states,
                    "n_iter": self.n_iter,
                    "random_state": self.random_state,
                    "covariance_type": self.covariance_type,
                    "n_init": self.n_init,
                    "tol": self.tol,
                    "strict_convergence": self.strict_convergence,
                    "model": self._model,
                    "scaler": self._scaler,
                    "state_returns": self.state_returns_,
                    "state_counts": self.state_counts_,
                    "feature_means": self.feature_means_,
                    "feature_cols": self.feature_cols_,
                    "return_col": self.return_col_,
                    "fitted": self.fitted_,
                    "converged": self.converged_,
                    "log_likelihood": self.log_likelihood_,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path | str) -> HiddenMarkovModel:
        with Path(path).open("rb") as fh:
            d = pickle.load(fh)
        obj = cls(
            n_states=d["n_states"],
            n_iter=d["n_iter"],
            random_state=d["random_state"],
            covariance_type=d["covariance_type"],
            n_init=d.get("n_init", 1),
            tol=d.get("tol", 1e-3),
            strict_convergence=d.get("strict_convergence", False),
        )
        obj._model = d["model"]
        obj._scaler = d.get("scaler")
        obj.state_returns_ = np.asarray(d["state_returns"])
        obj.state_counts_ = np.asarray(d["state_counts"])
        obj.feature_means_ = np.asarray(d["feature_means"])
        obj.feature_cols_ = list(d["feature_cols"])
        obj.return_col_ = d["return_col"]
        obj.fitted_ = bool(d["fitted"])
        obj.converged_ = bool(d.get("converged", False))
        obj.log_likelihood_ = d.get("log_likelihood")
        return obj
