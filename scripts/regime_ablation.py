"""Regime-filter ablation study — SKEPTIC_REVIEW.md Tier C4.

The hybrid model (production H-Vol @ q=0.70) claims that BOTH components
contribute: composite is the bear defender, HOMC is the bull participant,
and the vol-quantile router simply picks between them per bar.

C4 asks: if we replace each component with a constant signal and keep the
router, does the hybrid still produce a similar Sharpe? If yes, the Markov
components are decorative and the router alone is doing all the work.

This script evaluates four variants on BTC-USD via the 16-random-window
evaluator at seed 42:

  1. full              — H-Vol hybrid @ q=0.70 (control)
  2. composite_only    — vol router still switches; but when in "bull"
                         (HOMC branch), emit a constant "always long"
                         signal (target = +1.0)
  3. homc_only         — symmetric; when in "bear" (composite branch),
                         emit a constant "go flat" signal (target = 0)
  4. both_constants    — router → (flat, long) constants. Equivalent to a
                         pure vol filter / trend overlay with no Markov
                         structure at all; sanity-check row.

The wrapper classes subclass HybridRegimeModel and override predict_next +
state_returns_ so that, WHEN THE ROUTER PICKS THE ABLATED BRANCH, the
downstream SignalGenerator computes expected = probs @ state_returns_ as
a hard-coded constant:

  - "always long"  → expected = +0.02 (2%, far above buy threshold
                     25 bps and target_scale 20 bps); magnitude clips
                     to max_long = 1.0 at the portfolio layer.
  - "go flat"      → expected = -0.02 (2% loss, far below sell threshold
                     -35 bps); SELL signal → target forced to 0 (short
                     is disallowed in the production config).

Why SELL and not HOLD for the flat case? Because BacktestConfig sets
hold_preserves_position=True, which means HOLD keeps the current
position instead of flattening. SELL reliably drives target → 0 under
allow_short=False.

Why subclassing HybridRegimeModel and not monkey-patching it? Because
HybridRegimeModel's predict_state() sets self._active_component to the
raw composite/HOMC instance before returning its decoded state index.
SignalGenerator then reads model.predict_next(state) and
model.state_returns_, both of which delegate to _active_component. If
we intercept _active_component with a sentinel "self" and teach
predict_next/state_returns_ to return a synthetic 2-state vector in
that case, the ablation is invisible to SignalGenerator and every other
caller.

The synthetic state_returns_ vector is chosen as [-0.02, +0.02] and the
synthetic probs vector puts mass 1.0 on whichever index we want.
SignalGenerator's expected = probs @ state_returns_ = ±0.02, and its
confidence computation (mass on directionally-aligned states) gives
confidence = 1.0, so min_confidence gating cannot strip the signal.

All self-contained: no signals/model/ edits.

Usage:

    python scripts/regime_ablation.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.config import SETTINGS
from signals.data.storage import DataStore
from signals.model.hybrid import DEFAULT_ROUTING, HybridRegimeModel

# ----- Synthetic constants -----
#
# These magnitudes only need to clear the BUY / SELL thresholds in
# BacktestConfig (default ±25/±35 bps) and the SignalGenerator's
# target_scale (default 20 bps). A value of ±2% is 100× target_scale,
# so magnitude = |expected| / target_scale = 100, which clips to
# max_long = 1.0 at the signal generation stage. Clean constant target.
_CONSTANT_EXPECTED = 0.02
# Shared synthetic 2-state vector. Index 0 carries a negative return
# ("flat/sell" synthetic state) and index 1 carries a positive return
# ("long/buy" synthetic state). Whichever branch we're ablating picks
# one of these indices as its predict_state return value.
_SYNTHETIC_STATE_RETURNS = np.array([-_CONSTANT_EXPECTED, _CONSTANT_EXPECTED])
_SYNTHETIC_STATE_FLAT = 0   # SELL → target = 0 under allow_short=False
_SYNTHETIC_STATE_LONG = 1   # BUY  → target = max_long = 1.0


# A module-level sentinel, distinct from any real component instance.
# We set self._active_component = self when the router picks the
# ablated branch; predict_next / state_returns_ detect this and return
# the synthetic interface instead of delegating to composite/homc.
class _AblationHybridBase(HybridRegimeModel):
    """Base class for ablation wrappers.

    Subclasses decide which regime ("bear" / "bull") is ablated and
    which synthetic state the ablation emits.

    When the wrapper is the active component (i.e. the router picked
    the ablated branch), we store the synthetic state index in
    self._synthetic_state and make predict_next / state_returns_ /
    label return synthetic values keyed on that index.
    """

    # Subclasses override these:
    _ABLATED_COMPONENT: str = ""       # "composite" or "homc"
    _ABLATED_SYNTH_STATE: int = 0      # 0 (flat) or 1 (long)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Only used when self is the active component (ablated branch).
        self._synthetic_state: int = 0

    def predict_state(self, observations: pd.DataFrame) -> Any:
        if not self.fitted_:
            raise RuntimeError("HybridRegimeModel not fit")

        # Never takes the "blend" path — ablation runs on vol routing only.
        regime_label = self._regime_label(observations)
        self._last_regime_label = regime_label
        component_name, component = self._select_component(regime_label)

        if component_name == self._ABLATED_COMPONENT:
            # Router picked the ablated branch → substitute a constant
            # via the synthetic 2-state interface.
            self._active_component_name = f"{component_name}(ablated)"
            self._active_component = self  # sentinel: we are active
            self._synthetic_state = self._ABLATED_SYNTH_STATE
            return self._synthetic_state

        # Router picked the non-ablated branch → delegate normally.
        self._active_component_name = component_name
        self._active_component = component
        return component.predict_state(observations)

    def predict_next(self, state: Any) -> np.ndarray:
        if self._active_component is self:
            # Hard one-hot on the synthetic state so
            # expected = probs @ state_returns_ = _SYNTHETIC_STATE_RETURNS[state]
            # and confidence (mass on aligned states) = 1.0.
            probs = np.zeros(2)
            probs[int(state)] = 1.0
            return probs
        return super().predict_next(state)

    @property
    def state_returns_(self) -> np.ndarray:
        if self._active_component is self:
            return _SYNTHETIC_STATE_RETURNS
        return super().state_returns_

    def label(self, state: Any) -> str:
        if self._active_component is self:
            tag = "flat" if int(state) == _SYNTHETIC_STATE_FLAT else "long"
            return f"{self._last_regime_label}/{self._active_component_name}:{tag}"
        return super().label(state)


class CompositeOnlyHybrid(_AblationHybridBase):
    """Ablates the HOMC branch — when the router picks HOMC, emit constant LONG."""

    _ABLATED_COMPONENT = "homc"
    _ABLATED_SYNTH_STATE = _SYNTHETIC_STATE_LONG


class HomcOnlyHybrid(_AblationHybridBase):
    """Ablates the composite branch — when the router picks composite, emit constant FLAT."""

    _ABLATED_COMPONENT = "composite"
    _ABLATED_SYNTH_STATE = _SYNTHETIC_STATE_FLAT


class BothConstantsHybrid(HybridRegimeModel):
    """Ablates BOTH components — router → (flat, long) constants.

    This is the "pure vol filter" baseline: go long when vol is below
    the quantile threshold, go flat when vol is above it. No Markov
    structure anywhere. Included as the floor for what the router can
    produce on its own.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._synthetic_state: int = 0

    def predict_state(self, observations: pd.DataFrame) -> Any:
        if not self.fitted_:
            raise RuntimeError("HybridRegimeModel not fit")
        regime_label = self._regime_label(observations)
        self._last_regime_label = regime_label
        component_name, _ = self._select_component(regime_label)
        # Composite branch → flat constant; HOMC branch → long constant.
        if component_name == "homc":
            self._synthetic_state = _SYNTHETIC_STATE_LONG
        else:
            self._synthetic_state = _SYNTHETIC_STATE_FLAT
        self._active_component_name = f"{component_name}(ablated)"
        self._active_component = self
        return self._synthetic_state

    def predict_next(self, state: Any) -> np.ndarray:
        probs = np.zeros(2)
        probs[int(state)] = 1.0
        return probs

    @property
    def state_returns_(self) -> np.ndarray:
        if self._active_component is self:
            return _SYNTHETIC_STATE_RETURNS
        # Before predict_state is called — fall back to composite's.
        if self.composite is not None:
            return self.composite.state_returns_
        return np.zeros(0)

    def label(self, state: Any) -> str:
        tag = "flat" if int(state) == _SYNTHETIC_STATE_FLAT else "long"
        return f"{self._last_regime_label}/{self._active_component_name}:{tag}"


# ----- Custom engine factory -----
#
# BacktestEngine._make_model() is the single point where the hybrid
# class is instantiated. Subclassing the engine and overriding that
# one method is enough to swap in any of the ablation wrappers while
# reusing every other engine code path (fit kwargs, walk-forward loop,
# portfolio, metrics).
class _AblationEngine(BacktestEngine):
    def __init__(self, config: BacktestConfig, hybrid_cls: type[HybridRegimeModel]):
        super().__init__(config)
        self._hybrid_cls = hybrid_cls

    def _make_model(self):
        cfg = self.config
        if cfg.model_type != "hybrid":
            return super()._make_model()
        return self._hybrid_cls(
            regime_n_states=3,
            regime_n_iter=cfg.n_iter,
            regime_random_state=cfg.random_state,
            composite_return_bins=cfg.return_bins,
            composite_volatility_bins=cfg.volatility_bins,
            composite_alpha=cfg.laplace_alpha,
            homc_n_states=cfg.n_states if cfg.n_states >= 2 else 5,
            homc_order=cfg.order,
            homc_alpha=max(cfg.laplace_alpha, 1.0),
            routing=cfg.hybrid_routing or dict(DEFAULT_ROUTING),
            routing_strategy=cfg.hybrid_routing_strategy,
            vol_quantile_threshold=cfg.hybrid_vol_quantile,
            blend_low_quantile=cfg.hybrid_blend_low,
            blend_high_quantile=cfg.hybrid_blend_high,
            adaptive_low_quantile=cfg.hybrid_adaptive_low,
            adaptive_high_quantile=cfg.hybrid_adaptive_high,
            adaptive_lookback=cfg.hybrid_adaptive_lookback,
        )


# ----- Shared evaluation config -----

SYMBOL = "BTC-USD"
SYMBOL_START = pd.Timestamp("2015-01-01", tz="UTC")
SYMBOL_END = pd.Timestamp("2024-12-31", tz="UTC")
SEED = 42
N_WINDOWS = 16
SIX_MONTHS = 126
VOL_WINDOW = 10
HOMC_TRAIN_WINDOW = 1000
WARMUP_PAD = 5


def _hybrid_cfg() -> BacktestConfig:
    """Production H-Vol @ q=0.70 — same as random_window_eval's `hvol`."""
    return BacktestConfig(
        model_type="hybrid",
        train_window=HOMC_TRAIN_WINDOW,
        retrain_freq=21,
        n_states=5,
        order=5,
        return_bins=3,
        volatility_bins=3,
        vol_window=VOL_WINDOW,
        laplace_alpha=0.01,
        hybrid_routing_strategy="vol",
        hybrid_vol_quantile=0.70,
    )


@dataclass
class Variant:
    name: str
    hybrid_cls: type[HybridRegimeModel]


VARIANTS: list[Variant] = [
    Variant("full",           HybridRegimeModel),
    Variant("composite_only", CompositeOnlyHybrid),
    Variant("homc_only",      HomcOnlyHybrid),
    Variant("both_constants", BothConstantsHybrid),
]


def _run_on_window(
    variant: Variant,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
) -> Metrics:
    cfg = _hybrid_cfg()
    slice_start = start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]

    engine = _AblationEngine(cfg, variant.hybrid_cls)
    try:
        result = engine.run(engine_input, symbol=symbol)
    except Exception as e:
        print(f"  [{variant.name}] engine error: {e}")
        return compute_metrics(pd.Series(dtype=float), [])

    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(eq_rebased, [])


def _evaluate() -> pd.DataFrame:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= SYMBOL_START) & (prices.index <= SYMBOL_END)]
    if prices.empty:
        raise ValueError(f"No data for {SYMBOL} in the requested date range")

    print(f"{SYMBOL}: {len(prices)} bars  "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    min_start = HOMC_TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = len(prices) - SIX_MONTHS - 1
    if max_start - min_start < N_WINDOWS:
        raise ValueError(
            f"{SYMBOL} has too few bars for {N_WINDOWS} {SIX_MONTHS}-bar windows"
        )

    rng = random.Random(SEED)
    starts = sorted(rng.sample(range(min_start, max_start), N_WINDOWS))

    rows: list[dict] = []
    for i, start_i in enumerate(starts, start=1):
        end_i = start_i + SIX_MONTHS
        eval_window = prices.iloc[start_i:end_i]
        print(f"  window {i}/{N_WINDOWS}: "
              f"{eval_window.index[0].date()} → {eval_window.index[-1].date()}")

        for variant in VARIANTS:
            m = _run_on_window(variant, prices, start_i, end_i, SYMBOL)
            rows.append({
                "symbol": SYMBOL,
                "variant": variant.name,
                "window_idx": i,
                "start": eval_window.index[0].date(),
                "end": eval_window.index[-1].date(),
                "sharpe": float(m.sharpe),
                "cagr": float(m.cagr),
                "max_dd": float(m.max_drawdown),
                "n_trades": int(m.n_trades),
            })

    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame) -> pd.DataFrame:
    print()
    print("=" * 80)
    print(f"Regime ablation aggregate — {SYMBOL} — "
          f"{N_WINDOWS} random 6-month windows, seed {SEED}")
    print("=" * 80)
    print(f"{'variant':<18} "
          f"{'median Sh':>10} {'mean Sh':>10} "
          f"{'median CAGR':>13} {'mean MDD':>11}")
    print("-" * 80)

    agg_rows = []
    for variant in VARIANTS:
        sub = df[df["variant"] == variant.name]
        agg = {
            "variant": variant.name,
            "median_sharpe": float(sub["sharpe"].median()),
            "mean_sharpe": float(sub["sharpe"].mean()),
            "median_cagr": float(sub["cagr"].median()),
            "mean_mdd": float(sub["max_dd"].mean()),
            "n_windows": int(len(sub)),
        }
        agg_rows.append(agg)
        print(f"{variant.name:<18} "
              f"{agg['median_sharpe']:>+10.2f} "
              f"{agg['mean_sharpe']:>+10.2f} "
              f"{agg['median_cagr'] * 100:>+12.1f}% "
              f"{agg['mean_mdd'] * 100:>+10.1f}%")

    return pd.DataFrame(agg_rows)


def _print_interpretation(agg: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("Interpretation (SKEPTIC_REVIEW.md Tier C4)")
    print("=" * 80)

    def _med(name: str) -> float:
        return float(agg.loc[agg["variant"] == name, "median_sharpe"].iloc[0])

    full = _med("full")
    comp_only = _med("composite_only")
    homc_only = _med("homc_only")
    both = _med("both_constants")

    def _verdict(delta: float, component: str) -> str:
        if delta >= 0.3:
            return (
                f"  {component} CONTRIBUTES: replacing it with a constant "
                f"loses {delta:+.2f} Sharpe vs full hybrid."
            )
        if delta < 0.1:
            return (
                f"  {component} IS DECORATIVE: replacing it with a constant "
                f"loses only {delta:+.2f} Sharpe — within noise."
            )
        return (
            f"  {component} is AMBIGUOUS: replacing it with a constant "
            f"loses {delta:+.2f} Sharpe — between the 0.1 noise floor and "
            f"the 0.3 contribution threshold."
        )

    # "composite_only" ablates HOMC → tests HOMC's contribution.
    # "homc_only"      ablates composite → tests composite's contribution.
    delta_homc = full - comp_only
    delta_composite = full - homc_only

    print(f"  full hybrid median Sharpe        : {full:+.2f}")
    print(f"  composite_only (HOMC ablated)    : {comp_only:+.2f}  "
          f"(Δ vs full: {-delta_homc:+.2f})")
    print(f"  homc_only (composite ablated)    : {homc_only:+.2f}  "
          f"(Δ vs full: {-delta_composite:+.2f})")
    print(f"  both_constants (pure vol filter) : {both:+.2f}  "
          f"(Δ vs full: {both - full:+.2f})")
    print()
    print(_verdict(delta_homc, "HOMC branch"))
    print(_verdict(delta_composite, "Composite branch"))
    print()
    if both >= full - 0.1:
        print("  BOTH components together are decorative: the pure vol filter")
        print("  baseline (both_constants) matches the full hybrid within")
        print("  0.1 Sharpe. The router is doing all the work.")
    elif both >= full - 0.3:
        print("  The pure vol filter captures most of the hybrid's Sharpe.")
        print("  The Markov components add modest marginal value on top of")
        print("  the router.")
    else:
        print("  The pure vol filter is meaningfully worse than the full")
        print(f"  hybrid ({both - full:+.2f} Sharpe). The Markov components")
        print("  carry real signal beyond what the router alone provides.")


def main() -> None:
    df = _evaluate()

    # Save raw per-window results.
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "regime_ablation.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved per-window results: {out_path}")

    agg = _print_table(df)
    _print_interpretation(agg)


if __name__ == "__main__":
    main()
