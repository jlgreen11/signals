"""HybridRegimeModel — regime-aware ensemble of composite + HOMC.

Tier-0b evidence (see scripts/HOMC_TIER0B_COMPREHENSIVE.md) showed that:

  - HOMC@order=5/window=1000 is a bull-regime specialist. On a 16-window
    random BTC evaluation it beats composite on 11/16 windows aggregate
    (mean Sharpe 1.39 vs 1.10), but it loses every bear window (2018 late
    crash, 2022 crypto winter) to composite.

  - Composite-3×3 is bear-resistant. In the same three bear windows it
    delivers +20% to +28% returns while HOMC loses 14-62%.

  - The two models are complementary, not ranked. A regime-router that
    uses HOMC in bulls and composite in bears should beat both single
    models across the random-window evaluation.

This module implements that router. An HMM regime detector classifies each
bar into {bear, neutral, bull} based on return+volatility features, and the
hybrid delegates all prediction calls to one of the underlying models based
on a configurable routing table:

    default routing = {"bear": "composite", "neutral": "composite", "bull": "homc"}

The routing is explicitly conservative on the neutral bucket because the
dominant risk in the Tier-0b random-window eval was HOMC's bear liability,
not its bull opportunity cost. Making the router err toward composite when
unsure preserves the bear defense without giving up much bull participation.

All model-interface calls (`predict_state`, `predict_next`, `state_returns_`,
`label`) delegate to whichever component is currently active — that's the
component selected by the most recent `predict_state()` call. The engine's
flow (predict_state → generate → predict_next) ensures the active component
is always set before predict_next is invoked.

WALK-FORWARD: Because `predict_state` is called once per bar and sets the
active component before `predict_next` is called, the regime decision is
made fresh per bar — no cached state crosses bar boundaries. Retraining the
regime detector on each refit cycle automatically updates its idea of what
counts as bear/neutral/bull.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signals.model.composite import CompositeMarkovChain
from signals.model.hmm import HiddenMarkovModel
from signals.model.homc import HigherOrderMarkovChain
from signals.utils.logging import get_logger

log = get_logger(__name__)


DEFAULT_ROUTING: dict[str, str] = {
    "bear": "composite",
    "neutral": "composite",
    "bull": "homc",
}

# Routing strategies:
#   "hmm"           — use the HMM regime detector's label (original design)
#   "vol"           — hard binary switch at a FIXED vol quantile
#   "blend"         — continuous linear ramp between low and high quantiles
#   "adaptive_vol"  — hard binary switch where the vol quantile itself
#                     adapts based on the recent vol regime. In a
#                     high-vol recent regime (recent vol > training vol
#                     median) it uses a HIGH threshold quantile (more
#                     permissive of HOMC so we participate in
#                     high-vol-but-still-bullish rallies). In a low-vol
#                     recent regime it uses a LOW threshold quantile
#                     (more aggressive bear-defense via composite). This
#                     was the Tier-B #4 hypothesis: fixed quantiles might
#                     be suboptimal because the right threshold depends
#                     on the overall vol environment, not just the
#                     per-bar value.
ROUTING_STRATEGIES = ("hmm", "vol", "blend", "adaptive_vol")


class HybridRegimeModel:
    """Regime-routed ensemble over composite + HOMC.

    Exposes the same interface as the single models so BacktestEngine and
    SignalGenerator can swap it in transparently.
    """

    def __init__(
        self,
        regime_n_states: int = 3,
        regime_n_iter: int = 200,
        regime_random_state: int = 42,
        composite_return_bins: int = 3,
        composite_volatility_bins: int = 3,
        composite_alpha: float = 0.01,
        composite_train_window: int = 252,
        homc_n_states: int = 5,
        homc_order: int = 5,
        homc_alpha: float = 1.0,
        routing: dict[str, str] | None = None,
        routing_strategy: str = "hmm",
        vol_quantile_threshold: float = 0.75,
        blend_low_quantile: float = 0.50,
        blend_high_quantile: float = 0.85,
        adaptive_low_quantile: float = 0.60,
        adaptive_high_quantile: float = 0.80,
        adaptive_lookback: int = 30,
    ):
        if regime_n_states not in (2, 3):
            raise ValueError("regime_n_states must be 2 or 3")
        if routing_strategy not in ROUTING_STRATEGIES:
            raise ValueError(
                f"routing_strategy must be one of {ROUTING_STRATEGIES}"
            )
        # SKEPTIC_REVIEW.md § C5: HMM-based routing was shown in Tier-0c
        # (HOMC_TIER0C_HYBRID_RESULTS.md) to whipsaw catastrophically on
        # ambiguous regimes (median Sharpe 0.37 vs H-Vol's 1.92). It is
        # retained for historical comparisons but should NOT be used as a
        # production config. Emit a runtime warning so anyone who stumbles
        # into it via the CLI sees the deprecation.
        if routing_strategy == "hmm":
            log.warning(
                "routing_strategy='hmm' is DEPRECATED — HMM-based regime "
                "routing whipsaws on ambiguous regimes (see "
                "HOMC_TIER0C_HYBRID_RESULTS.md; median Sharpe 0.37 vs "
                "H-Vol's 1.92). Use routing_strategy='vol' for production."
            )
        self.regime_n_states = int(regime_n_states)
        self.regime_n_iter = int(regime_n_iter)
        self.regime_random_state = int(regime_random_state)
        self.composite_return_bins = int(composite_return_bins)
        self.composite_volatility_bins = int(composite_volatility_bins)
        self.composite_alpha = float(composite_alpha)
        # Composite is tuned for a 252-bar window ("tightened defaults").
        # Training it on the full 1000-bar slice that HOMC requires blunts
        # its bear-regime sensitivity — verified empirically against the
        # 16-window random eval. Fit composite on just the trailing
        # composite_train_window bars of whatever the engine passes in.
        self.composite_train_window = int(composite_train_window)
        self.homc_n_states = int(homc_n_states)
        self.homc_order = int(homc_order)
        self.homc_alpha = float(homc_alpha)
        self.routing = dict(routing) if routing is not None else dict(DEFAULT_ROUTING)
        self.routing_strategy = routing_strategy
        self.vol_quantile_threshold = float(vol_quantile_threshold)
        self.blend_low_quantile = float(blend_low_quantile)
        self.blend_high_quantile = float(blend_high_quantile)
        if not 0 <= self.blend_low_quantile < self.blend_high_quantile <= 1:
            raise ValueError(
                "blend_low_quantile must be < blend_high_quantile and both in [0, 1]"
            )
        self.adaptive_low_quantile = float(adaptive_low_quantile)
        self.adaptive_high_quantile = float(adaptive_high_quantile)
        self.adaptive_lookback = int(adaptive_lookback)
        if not 0 <= self.adaptive_low_quantile < self.adaptive_high_quantile <= 1:
            raise ValueError(
                "adaptive_low_quantile must be < adaptive_high_quantile and both in [0, 1]"
            )
        self._vol_threshold_value: float | None = None
        self._blend_low_value: float | None = None
        self._blend_high_value: float | None = None
        self._adaptive_low_value: float | None = None
        self._adaptive_high_value: float | None = None
        self._adaptive_median_vol: float | None = None
        # Blend strategy precomputes the blended expected return at
        # predict_state time and exposes it via a synthetic 1-state
        # interface in predict_next/state_returns_. These fields hold
        # the current per-bar values set by predict_state.
        self._blended_expected_return: float = 0.0
        self._blend_weight_composite: float = 0.0
        self._validate_routing()

        self.regime_detector: HiddenMarkovModel | None = None
        self.composite: CompositeMarkovChain | None = None
        self.homc: HigherOrderMarkovChain | None = None

        self._active_component_name: str = "composite"  # safe default
        self._active_component: Any = None
        self._last_regime_label: str = "neutral"

        self.fitted_: bool = False

    def _validate_routing(self) -> None:
        allowed_regimes = {"bear", "neutral", "bull"}
        if self.regime_n_states == 2:
            allowed_regimes = {"bear", "bull"}
        for regime, component in self.routing.items():
            if regime not in allowed_regimes:
                raise ValueError(
                    f"routing regime {regime!r} not in {allowed_regimes}"
                )
            if component not in ("composite", "homc"):
                raise ValueError(
                    f"routing component {component!r} must be 'composite' or 'homc'"
                )
        for regime in allowed_regimes:
            if regime not in self.routing:
                raise ValueError(f"routing missing regime {regime!r}")

    # ----- Fit -----
    def fit(
        self,
        observations: pd.DataFrame,
        feature_col: str = "return_1d",
        return_col: str = "return_1d",
        **_ignored: Any,
    ) -> HybridRegimeModel:
        """Fit regime detector + both components on the same training window.

        Unknown kwargs are accepted and ignored so the engine can pass the
        union of all model-specific fit kwargs without caring which subset
        applies.
        """
        # 1. Regime signal: either HMM-based or volatility-based
        if self.routing_strategy == "hmm":
            self.regime_detector = HiddenMarkovModel(
                n_states=self.regime_n_states,
                n_iter=self.regime_n_iter,
                random_state=self.regime_random_state,
            )
            self.regime_detector.fit(
                observations,
                feature_cols=["return_1d", "volatility_20d"],
                return_col="return_1d",
            )
        else:
            # Vol-based routing: compute vol quantile(s) from the training
            # window. The "vol" strategy uses a single threshold for hard
            # switching; the "blend" strategy uses a low/high pair that
            # defines a linear ramp; the "adaptive_vol" strategy uses a
            # low/high pair plus a recent-vol pivot that selects between
            # them based on the realized vol regime.
            vols = observations["volatility_20d"].dropna()
            if len(vols) < 10:
                raise ValueError("Not enough vol observations to set threshold")
            self._vol_threshold_value = float(
                vols.quantile(self.vol_quantile_threshold)
            )
            self._blend_low_value = float(vols.quantile(self.blend_low_quantile))
            self._blend_high_value = float(vols.quantile(self.blend_high_quantile))
            self._adaptive_low_value = float(
                vols.quantile(self.adaptive_low_quantile)
            )
            self._adaptive_high_value = float(
                vols.quantile(self.adaptive_high_quantile)
            )
            self._adaptive_median_vol = float(vols.median())
            self.regime_detector = None

        # 2. Composite component — trained on the trailing
        # composite_train_window bars only, matching the standalone
        # composite's tuning. Training on the full input slice blunts the
        # bear-regime detection.
        self.composite = CompositeMarkovChain(
            return_bins=self.composite_return_bins,
            volatility_bins=self.composite_volatility_bins,
            alpha=self.composite_alpha,
        )
        composite_slice = observations.iloc[-self.composite_train_window :]
        self.composite.fit(
            composite_slice,
            return_feature="return_1d",
            volatility_feature="volatility_20d",
            return_col="return_1d",
        )

        # 3. HOMC component
        self.homc = HigherOrderMarkovChain(
            n_states=self.homc_n_states,
            order=self.homc_order,
            alpha=self.homc_alpha,
        )
        self.homc.fit(
            observations,
            feature_col="return_1d",
            return_col="return_1d",
        )

        self.fitted_ = True
        return self

    # ----- Predict -----
    def _regime_label(self, observations: pd.DataFrame) -> str:
        """Classify the current bar's regime via the configured strategy."""
        if self.routing_strategy == "hmm":
            assert self.regime_detector is not None
            state = self.regime_detector.predict_state(observations)
            return self.regime_detector.label(state)
        if self.routing_strategy == "adaptive_vol":
            # Adaptive: pick the low or high quantile threshold based on
            # the RECENT vol regime. If the last `adaptive_lookback` bars
            # of vol have a mean above the training median, we're in a
            # high-vol environment — use the high quantile (more
            # permissive, HOMC runs more often even on choppy days).
            # Otherwise use the low quantile (more bear-defensive,
            # composite runs on smaller vol spikes).
            assert self._adaptive_low_value is not None
            assert self._adaptive_high_value is not None
            assert self._adaptive_median_vol is not None
            vol_series = observations["volatility_20d"].dropna()
            if vol_series.empty:
                return "neutral"
            current_vol = float(vol_series.iloc[-1])
            recent_vols = vol_series.iloc[-self.adaptive_lookback :]
            recent_mean = float(recent_vols.mean())
            in_high_vol_regime = recent_mean > self._adaptive_median_vol
            threshold = (
                self._adaptive_high_value
                if in_high_vol_regime
                else self._adaptive_low_value
            )
            return "bear" if current_vol >= threshold else "bull"
        # "vol" or "blend": get the latest non-NaN 20d vol and compare
        # to threshold.
        assert self._vol_threshold_value is not None
        vol_series = observations["volatility_20d"].dropna()
        if vol_series.empty:
            return "neutral"
        current_vol = float(vol_series.iloc[-1])
        return "bear" if current_vol >= self._vol_threshold_value else "bull"

    def _select_component(self, regime_label: str) -> tuple[str, Any]:
        """Route regime → component instance."""
        component_name = self.routing.get(regime_label, "composite")
        if component_name == "homc":
            assert self.homc is not None
            return component_name, self.homc
        assert self.composite is not None
        return component_name, self.composite

    def _blend_weight(self, observations: pd.DataFrame) -> float:
        """Return the composite weight in [0, 1] for the current bar's vol.

        0 = full HOMC, 1 = full composite. Linear ramp between
        blend_low_value (w=0) and blend_high_value (w=1).
        """
        assert self._blend_low_value is not None
        assert self._blend_high_value is not None
        vol_series = observations["volatility_20d"].dropna()
        if vol_series.empty:
            return 0.5
        current_vol = float(vol_series.iloc[-1])
        lo, hi = self._blend_low_value, self._blend_high_value
        if current_vol <= lo:
            return 0.0
        if current_vol >= hi:
            return 1.0
        return (current_vol - lo) / (hi - lo)

    def predict_state(self, observations: pd.DataFrame) -> Any:
        if not self.fitted_:
            raise RuntimeError("HybridRegimeModel not fit")

        if self.routing_strategy == "blend":
            # Continuous blend: run both components, compute their expected
            # returns, and synthesize a weighted blend. The active component
            # is marked as "blend" and predict_next/state_returns_ return a
            # 1-state synthetic interface carrying the blended expected.
            assert self.composite is not None and self.homc is not None
            state_comp = self.composite.predict_state(observations)
            state_homc = self.homc.predict_state(observations)
            e_comp = float(
                self.composite.predict_next(state_comp)
                @ self.composite.state_returns_
            )
            e_homc = float(
                self.homc.predict_next(state_homc) @ self.homc.state_returns_
            )
            w = self._blend_weight(observations)
            self._blend_weight_composite = w
            self._blended_expected_return = w * e_comp + (1.0 - w) * e_homc
            if w >= 0.5:
                self._last_regime_label = "bear"
                self._active_component_name = f"blend(c={w:.2f})"
            else:
                self._last_regime_label = "bull"
                self._active_component_name = f"blend(h={1 - w:.2f})"
            # Active component is self — delegate predict_next/state_returns_
            # to the synthetic interface below.
            self._active_component = self
            return 0  # synthetic state index

        regime_label = self._regime_label(observations)
        self._last_regime_label = regime_label
        component_name, component = self._select_component(regime_label)
        self._active_component_name = component_name
        self._active_component = component
        return component.predict_state(observations)

    def predict_next(self, state: Any) -> np.ndarray:
        if self._active_component is None:
            raise RuntimeError(
                "predict_state must be called before predict_next — active "
                "component is not set"
            )
        if self.routing_strategy == "blend":
            # Synthetic 1-state interface: returns [1.0] so the SignalGenerator
            # dot-product expected = 1.0 * blended_expected_return.
            return np.array([1.0])
        return self._active_component.predict_next(state)

    @property
    def state_returns_(self) -> np.ndarray:
        if self.routing_strategy == "blend" and self.fitted_:
            # Synthetic 1-state vector carrying the per-bar blended
            # expected return. SignalGenerator computes
            #     expected = probs @ state_returns_ = 1 * blended_expected
            # and then the direction mask is [True] or [False] depending on
            # the sign, giving confidence = 1.0. Confidence of 1.0 is OK
            # because the components' individual confidences are already
            # baked into their expected returns (a weak signal has a small
            # |expected|), and SignalGenerator's magnitude formula
            # (|expected| / scale) sizes positions proportionally.
            return np.array([self._blended_expected_return])
        if self._active_component is None:
            if self.composite is not None:
                return self.composite.state_returns_
            return np.zeros(0)
        return self._active_component.state_returns_

    def label(self, state: Any) -> str:
        if self.routing_strategy == "blend":
            return f"{self._last_regime_label}/{self._active_component_name}"
        if self._active_component is None:
            return "hybrid/?"
        base = self._active_component.label(state)
        return f"{self._last_regime_label}/{self._active_component_name}:{base}"

    def expected_next_return(self, state: Any) -> float:
        if self.routing_strategy == "blend":
            return float(self._blended_expected_return)
        if self._active_component is None:
            raise RuntimeError("predict_state must be called first")
        return self._active_component.expected_next_return(state)

    # ----- Introspection -----
    @property
    def n_states(self) -> int:
        """n_states of the currently active component, or composite's as default."""
        if self._active_component is not None:
            return int(self._active_component.n_states)
        if self.composite is not None:
            return int(self.composite.n_states)
        return 0

    @property
    def last_regime_label(self) -> str:
        return self._last_regime_label

    @property
    def active_component_name(self) -> str:
        return self._active_component_name

    # ----- Persistence -----
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        base = path.with_suffix("")
        manifest = {
            "type": "HybridRegimeModel",
            "regime_n_states": self.regime_n_states,
            "regime_n_iter": self.regime_n_iter,
            "regime_random_state": self.regime_random_state,
            "composite_return_bins": self.composite_return_bins,
            "composite_volatility_bins": self.composite_volatility_bins,
            "composite_alpha": self.composite_alpha,
            "composite_train_window": self.composite_train_window,
            "homc_n_states": self.homc_n_states,
            "homc_order": self.homc_order,
            "homc_alpha": self.homc_alpha,
            "routing": self.routing,
            "routing_strategy": self.routing_strategy,
            "vol_quantile_threshold": self.vol_quantile_threshold,
            "vol_threshold_value": self._vol_threshold_value,
            "blend_low_quantile": self.blend_low_quantile,
            "blend_high_quantile": self.blend_high_quantile,
            "blend_low_value": self._blend_low_value,
            "blend_high_value": self._blend_high_value,
            "adaptive_low_quantile": self.adaptive_low_quantile,
            "adaptive_high_quantile": self.adaptive_high_quantile,
            "adaptive_lookback": self.adaptive_lookback,
            "adaptive_low_value": self._adaptive_low_value,
            "adaptive_high_value": self._adaptive_high_value,
            "adaptive_median_vol": self._adaptive_median_vol,
            "fitted": self.fitted_,
            "files": {
                "regime": f"{base.name}.regime.pkl",
                "composite": f"{base.name}.composite.json",
                "homc": f"{base.name}.homc.json",
            },
        }
        path.write_text(json.dumps(manifest, indent=2))
        if self.regime_detector is not None and self.routing_strategy == "hmm":
            self.regime_detector.save(base.with_name(base.name + ".regime.pkl"))
        if self.composite is not None:
            self.composite.save(base.with_name(base.name + ".composite.json"))
        if self.homc is not None:
            self.homc.save(base.with_name(base.name + ".homc.json"))

    @classmethod
    def load(cls, path: Path | str) -> HybridRegimeModel:
        path = Path(path)
        manifest = json.loads(path.read_text())
        obj = cls(
            regime_n_states=manifest["regime_n_states"],
            regime_n_iter=manifest["regime_n_iter"],
            regime_random_state=manifest["regime_random_state"],
            composite_return_bins=manifest["composite_return_bins"],
            composite_volatility_bins=manifest["composite_volatility_bins"],
            composite_alpha=manifest["composite_alpha"],
            composite_train_window=manifest.get("composite_train_window", 252),
            homc_n_states=manifest["homc_n_states"],
            homc_order=manifest["homc_order"],
            homc_alpha=manifest["homc_alpha"],
            routing=manifest["routing"],
            routing_strategy=manifest.get("routing_strategy", "hmm"),
            vol_quantile_threshold=manifest.get("vol_quantile_threshold", 0.75),
            blend_low_quantile=manifest.get("blend_low_quantile", 0.50),
            blend_high_quantile=manifest.get("blend_high_quantile", 0.85),
            adaptive_low_quantile=manifest.get("adaptive_low_quantile", 0.60),
            adaptive_high_quantile=manifest.get("adaptive_high_quantile", 0.80),
            adaptive_lookback=manifest.get("adaptive_lookback", 30),
        )
        base = path.with_suffix("")
        if obj.routing_strategy == "hmm":
            obj.regime_detector = HiddenMarkovModel.load(
                base.with_name(base.name + ".regime.pkl")
            )
        else:
            obj._vol_threshold_value = manifest.get("vol_threshold_value")
            obj._blend_low_value = manifest.get("blend_low_value")
            obj._blend_high_value = manifest.get("blend_high_value")
            obj._adaptive_low_value = manifest.get("adaptive_low_value")
            obj._adaptive_high_value = manifest.get("adaptive_high_value")
            obj._adaptive_median_vol = manifest.get("adaptive_median_vol")
        obj.composite = CompositeMarkovChain.load(
            base.with_name(base.name + ".composite.json")
        )
        obj.homc = HigherOrderMarkovChain.load(
            base.with_name(base.name + ".homc.json")
        )
        obj.fitted_ = bool(manifest["fitted"])
        return obj
