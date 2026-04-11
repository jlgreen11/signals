"""EnsembleModel — combine multiple model backends via weighted averaging.

Runs several underlying models in parallel, averages their expected-
return predictions (weighted by configurable weights), and presents
the standard interface to the engine.

The per-bar flow:
  1. Each component is fit on the same training window at fit() time
  2. At predict_state(), every component predicts its own state and
     expected return
  3. The ensemble's expected return is the weighted average of the
     components' expected returns
  4. A synthetic 1-state interface exposes the averaged expected return
     via state_returns_ and predict_next(), so SignalGenerator picks
     it up unchanged

Use cases:
  - Combine composite (bear defense) + HOMC (bull participation) +
    boost (ML-based) to hedge against each individual model's failure
    modes
  - Ensemble different hyperparameter settings of the same model
    (bagging-like variance reduction)

Default configuration: 3-way ensemble of composite-3×3, HOMC@order=5,
GradientBoosting with 100 estimators, equal weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signals.model.boost import GradientBoostingModel
from signals.model.composite import CompositeMarkovChain
from signals.model.homc import HigherOrderMarkovChain


class EnsembleModel:
    """Weighted-average ensemble of multiple model backends.

    Presents a synthetic 1-state interface where state_returns_ carries
    the blended expected return. SignalGenerator's formula
    `expected = probs @ state_returns_` then yields
    `1.0 * blended_expected = blended_expected` directly.
    """

    def __init__(
        self,
        components: list[tuple[str, Any, float]] | None = None,
    ):
        """
        components: list of (name, model_instance, weight) tuples. The
        weights must sum to 1.0. If None, builds a default 3-way
        ensemble of composite + HOMC + boost.
        """
        if components is None:
            components = self._default_components()
        total = sum(w for _, _, w in components)
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"weights must sum to 1.0; got {total:.4f}")
        self.components = list(components)
        self.n_states = 1
        self.state_returns_ = np.array([0.0])
        self.fitted_: bool = False
        self._blended_expected: float = 0.0

    @staticmethod
    def _default_components() -> list[tuple[str, Any, float]]:
        return [
            (
                "composite",
                CompositeMarkovChain(return_bins=3, volatility_bins=3, alpha=0.01),
                1.0 / 3,
            ),
            (
                "homc",
                HigherOrderMarkovChain(n_states=5, order=5, alpha=1.0),
                1.0 / 3,
            ),
            (
                "boost",
                GradientBoostingModel(n_estimators=100, max_depth=3),
                1.0 / 3,
            ),
        ]

    def fit(
        self,
        observations: pd.DataFrame,
        feature_col: str = "return_1d",
        return_col: str = "return_1d",
        **_ignored,
    ) -> EnsembleModel:
        for name, model, _ in self.components:
            try:
                if isinstance(model, CompositeMarkovChain):
                    model.fit(
                        observations,
                        return_feature="return_1d",
                        volatility_feature="volatility_20d",
                        return_col=return_col,
                    )
                elif isinstance(model, HigherOrderMarkovChain | GradientBoostingModel):
                    model.fit(
                        observations,
                        feature_col=feature_col,
                        return_col=return_col,
                    )
                else:
                    model.fit(observations)
            except Exception as e:
                print(f"ensemble component {name!r} failed to fit: {e}")
        self.fitted_ = True
        return self

    def _component_expected(
        self,
        name: str,
        model: Any,
        observations: pd.DataFrame,
    ) -> float:
        """Compute the component's expected-return prediction for the
        current observation."""
        if not getattr(model, "fitted_", False):
            return 0.0
        try:
            state = model.predict_state(observations)
            probs = model.predict_next(state)
            expected = float(probs @ model.state_returns_)
            return expected
        except Exception as e:
            print(f"ensemble component {name!r} failed to predict: {e}")
            return 0.0

    def predict_state(self, observations: pd.DataFrame) -> int:
        if not self.fitted_:
            raise RuntimeError("EnsembleModel not fit")
        # Compute each component's expected return and blend
        blended = 0.0
        total_w = 0.0
        for name, model, weight in self.components:
            e = self._component_expected(name, model, observations)
            blended += weight * e
            total_w += weight
        if total_w > 0:
            blended /= total_w
        self._blended_expected = blended
        return 0  # synthetic single-state

    def predict_next(self, state: int) -> np.ndarray:
        return np.array([1.0])

    @property
    def state_returns_(self) -> np.ndarray:
        # Dynamic property — computed from the most recent predict_state
        return np.array([self._blended_expected])

    @state_returns_.setter
    def state_returns_(self, value: np.ndarray) -> None:
        # Ignored — this class's state_returns_ is computed dynamically
        pass

    def expected_next_return(self, state: int) -> float:
        return float(self._blended_expected)

    def label(self, state) -> str:
        direction = "up" if self._blended_expected >= 0 else "down"
        return f"ensemble/{direction}({self._blended_expected * 1e4:+.1f}bps)"

    def save(self, path: Path | str) -> None:
        """Save the ensemble manifest and each component's serialized form."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        base = p.with_suffix("")
        manifest = {
            "type": "EnsembleModel",
            "fitted": self.fitted_,
            "components": [
                {
                    "name": name,
                    "type": type(model).__name__,
                    "weight": weight,
                    "file": f"{base.name}.{name}",
                }
                for name, model, weight in self.components
            ],
        }
        p.write_text(json.dumps(manifest, indent=2))
        for name, model, _ in self.components:
            component_path = base.with_name(f"{base.name}.{name}")
            if hasattr(model, "save") and isinstance(
                model,
                GradientBoostingModel | CompositeMarkovChain | HigherOrderMarkovChain,
            ):
                model.save(component_path.with_suffix(".json"))

    @staticmethod
    def weights_from_sharpes(
        sharpes: dict[str, float],
        floor: float = 0.0,
        negative_policy: str = "floor",
    ) -> dict[str, float]:
        """Convert a dict of component Sharpes to normalized weights.

        This is the building block for a performance-weighted ensemble:
        run each component in a validation window, pass the resulting
        per-component Sharpe ratios here, and the output is a weight
        dict that can be passed back into `EnsembleModel` via its
        `components=[...]` constructor argument.

        Parameters
        ----------
        sharpes
            Dict mapping component name → recent out-of-sample Sharpe.
        floor
            Minimum weight to assign each component after normalization.
            A floor > 0 prevents any single component from being
            completely zeroed out.
        negative_policy
            How to handle negative Sharpes:
              - "floor" (default): clip negatives to 0, then normalize.
                Components with Sharpe <= 0 get only the `floor` weight.
              - "shift": add |min_sharpe| + epsilon to all components so
                the worst becomes barely positive. Preserves rank but
                keeps all components alive.
              - "keep": leave negatives as-is. Results may be nonsense
                if the sum is near zero or negative — use at your own
                risk.

        Returns
        -------
        dict[str, float]
            Weights summing to 1.0, one entry per input component.
        """
        if not sharpes:
            return {}
        if negative_policy == "shift":
            min_s = min(sharpes.values())
            offset = -min_s + 1e-6 if min_s < 0 else 0.0
            scores = {k: v + offset for k, v in sharpes.items()}
        elif negative_policy == "floor":
            scores = {k: max(0.0, v) for k, v in sharpes.items()}
        elif negative_policy == "keep":
            scores = dict(sharpes)
        else:
            raise ValueError(f"unknown negative_policy: {negative_policy!r}")

        total = sum(scores.values())
        n = len(scores)
        if total <= 0:
            # All components are at the floor — fall back to equal weight
            return {k: 1.0 / n for k in sharpes}

        raw = {k: v / total for k, v in scores.items()}
        if floor <= 0:
            return raw

        # Apply floor: each component gets at least `floor`. Remaining
        # weight (1 - n*floor) is distributed proportionally.
        if floor * n >= 1.0:
            # Floor is too aggressive — fall back to equal weight.
            return {k: 1.0 / n for k in sharpes}
        remaining = 1.0 - floor * n
        weighted = {k: floor + remaining * raw[k] for k in sharpes}
        # Renormalize to correct any tiny FP drift.
        total_w = sum(weighted.values())
        return {k: v / total_w for k, v in weighted.items()}

    @classmethod
    def load(cls, path: Path | str) -> EnsembleModel:
        p = Path(path)
        manifest = json.loads(p.read_text())
        base = p.with_suffix("")
        components: list[tuple[str, Any, float]] = []
        for c in manifest["components"]:
            name = c["name"]
            weight = c["weight"]
            component_path = base.with_name(f"{base.name}.{name}.json")
            if c["type"] == "CompositeMarkovChain":
                model = CompositeMarkovChain.load(component_path)
            elif c["type"] == "HigherOrderMarkovChain":
                model = HigherOrderMarkovChain.load(component_path)
            elif c["type"] == "GradientBoostingModel":
                model = GradientBoostingModel.load(component_path)
            else:
                continue
            components.append((name, model, weight))
        obj = cls(components=components)
        obj.fitted_ = bool(manifest.get("fitted", False))
        return obj
