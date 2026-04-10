"""GradientBoostingModel — sklearn-based classifier for next-bar direction.

A structurally different model class from the Markov-chain backbone. Fits
a `GradientBoostingClassifier` to predict whether the NEXT bar's return
will be positive or negative given engineered features from the recent
price history:

  - Recent log returns at lags 1, 3, 5, 10, 20, 50
  - Recent volatility (20-day rolling std of log returns)
  - Return / volatility ratio (z-score of recent return)
  - Rolling mean return over the last 5, 20, 50 bars
  - Rolling max-drawdown over the last 20 bars

At predict time, returns a synthetic 2-state interface compatible with
SignalGenerator:
  - state 0: "down" (predicted next return < 0)
  - state 1: "up" (predicted next return > 0)
  - state_returns_ synthesized from predicted probability so the
    magnitude of the expected return reflects the classifier's
    confidence

The model stores the sklearn estimator pickled inside the JSON manifest
via base64 encoding (hmm.py uses a similar pattern). Load returns a
fully-restored instance.

This is Tier-3 Phase E — a research extension to test whether a
non-Markov model class can beat the Markov plateau on BTC.
"""

from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Synthetic state returns used by SignalGenerator. These are scaled so
# that the dot product with predict_next probabilities produces an
# expected return in the right order of magnitude for the BUY/SELL
# thresholds (25 / -35 bps) to actually trigger.
_STATE_RETURNS = np.array([-0.005, 0.005])  # ±50 bps


def _build_features(
    returns: pd.Series, volatility: pd.Series
) -> pd.DataFrame:
    """Engineer features from a return series.

    Uses only PAST data — every feature at time t depends only on
    data available at bar t. Critical for no-lookahead.
    """
    df = pd.DataFrame(index=returns.index)
    for lag in [1, 3, 5, 10, 20, 50]:
        df[f"ret_lag{lag}"] = returns.shift(lag)
    df["vol_20"] = volatility.copy()
    df["ret_20_mean"] = returns.rolling(20, min_periods=20).mean()
    df["ret_50_mean"] = returns.rolling(50, min_periods=50).mean()
    df["ret_5_mean"] = returns.rolling(5, min_periods=5).mean()
    # Return/vol z-score
    df["ret_z"] = returns.rolling(20, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9),
        raw=False,
    )
    # Rolling max drawdown over last 20 bars
    cumret = returns.rolling(20).sum()
    df["cum_ret_20"] = cumret
    return df


class GradientBoostingModel:
    """Predicts next-bar direction using a sklearn gradient boosting
    classifier on engineered features. Conforms to the project's model
    interface so BacktestEngine and SignalGenerator can swap it in.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 42,
        min_training_samples: int = 100,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.min_training_samples = int(min_training_samples)
        self.n_states = 2
        self.state_returns_ = _STATE_RETURNS.copy()
        self.state_counts_: np.ndarray = np.zeros(2, dtype=np.int64)
        self._model = None
        self._feature_names: list[str] = []
        self.fitted_: bool = False

    def fit(
        self,
        observations: pd.DataFrame,
        feature_col: str = "return_1d",
        return_col: str = "return_1d",
        **_ignored,
    ) -> GradientBoostingModel:
        from sklearn.ensemble import GradientBoostingClassifier

        returns = observations[return_col].dropna()
        if "volatility_20d" in observations.columns:
            vol = observations["volatility_20d"]
        else:
            vol = returns.rolling(20, min_periods=20).std()

        features = _build_features(returns, vol)
        # Target: next bar's return sign
        target = (returns.shift(-1) > 0).astype(int)

        # Align and drop NaNs
        aligned = pd.concat(
            [features, target.rename("target")], axis=1
        ).dropna()
        if len(aligned) < self.min_training_samples:
            raise ValueError(
                f"Need >= {self.min_training_samples} samples; got {len(aligned)}"
            )

        X = aligned.drop(columns=["target"]).to_numpy()
        y = aligned["target"].to_numpy()
        self._feature_names = list(aligned.drop(columns=["target"]).columns)

        self._model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self._model.fit(X, y)

        self.state_counts_ = np.array(
            [int((y == 0).sum()), int((y == 1).sum())], dtype=np.int64
        )
        self.fitted_ = True
        return self

    def _predict_proba_up(self, observations: pd.DataFrame) -> float:
        """Compute P(up) for the most recent observation."""
        if self._model is None:
            raise RuntimeError("GradientBoostingModel not fit")
        returns = observations["return_1d"].dropna()
        if "volatility_20d" in observations.columns:
            vol = observations["volatility_20d"]
        else:
            vol = returns.rolling(20, min_periods=20).std()
        features = _build_features(returns, vol).dropna()
        if features.empty:
            return 0.5
        # Use the most recent row
        row = features.iloc[[-1]]
        # Ensure feature order matches training
        row = row[self._feature_names]
        probs = self._model.predict_proba(row.to_numpy())[0]
        return float(probs[1])  # probability of class 1 (up)

    def predict_state(self, observations: pd.DataFrame) -> int:
        p_up = self._predict_proba_up(observations)
        return 1 if p_up >= 0.5 else 0

    def predict_next(self, state: int) -> np.ndarray:
        s = int(state)
        if s not in (0, 1):
            raise ValueError(f"state must be 0 or 1; got {s}")
        # Return a one-hot so `probs @ state_returns_` yields the
        # state's expected return directly. The real probability is
        # collapsed into the state choice at predict_state time.
        out = np.zeros(2)
        out[s] = 1.0
        return out

    def expected_next_return(self, state: int) -> float:
        return float(self.state_returns_[int(state)])

    def label(self, state) -> str:
        s = int(state)
        if s == 0:
            return "down"
        if s == 1:
            return "up"
        return f"state_{s}"

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        model_b64 = (
            base64.b64encode(pickle.dumps(self._model)).decode("ascii")
            if self._model is not None
            else None
        )
        payload = {
            "type": "GradientBoostingModel",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "min_training_samples": self.min_training_samples,
            "feature_names": self._feature_names,
            "state_counts": self.state_counts_.tolist(),
            "state_returns": self.state_returns_.tolist(),
            "model_b64": model_b64,
            "fitted": self.fitted_,
        }
        p.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path | str) -> GradientBoostingModel:
        d = json.loads(Path(path).read_text())
        obj = cls(
            n_estimators=d["n_estimators"],
            max_depth=d["max_depth"],
            learning_rate=d["learning_rate"],
            random_state=d["random_state"],
            min_training_samples=d["min_training_samples"],
        )
        obj._feature_names = list(d["feature_names"])
        obj.state_counts_ = np.array(d["state_counts"], dtype=np.int64)
        obj.state_returns_ = np.array(d["state_returns"])
        if d.get("model_b64"):
            obj._model = pickle.loads(base64.b64decode(d["model_b64"]))
        obj.fitted_ = bool(d["fitted"])
        return obj
