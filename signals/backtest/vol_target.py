"""Volatility-targeting overlay — post-signal position sizer.

Scales a raw target position (produced by SignalGenerator) so that the
*expected* portfolio volatility over the next bar approximates a fixed
annualized target. This is a standard risk-budgeting technique that:

  - Reduces position size during high-vol regimes (crash protection)
  - Increases position size during low-vol regimes (capital efficiency)

It is a known-winning technique in the literature — Moskowitz et al.
"Time Series Momentum" (2012) popularized vol-targeted sizing on top of
simpler signals. It has NOT yet been applied to this project's BTC
strategies; it is one of the "what I would do differently" items the
Tier 3 writeup flagged.

This overlay is applied as a POST-signal multiplier:

    scale = target_annual_vol / (realized_daily_vol * sqrt(periods_per_year))
    sized_target = raw_target * clip(scale, min_scale, max_scale)

Critical design rules:

  1. **No lookahead.** The overlay consumes only realized vol up to the
     decision bar t (same rule the rest of the engine follows). The
     engine passes in `row["volatility"]` which is already computed
     with a trailing window.

  2. **Scale cap.** Without a max_scale cap, low-vol regimes can produce
     arbitrarily high leverage. Default cap is 2.0 (2x the raw target).
     Above that, realistic-execution assumptions (margin, liquidity,
     wipeout risk) break down.

  3. **Annualization factor is asset-specific.** BTC trades 365d/year;
     US equities 252. Default is 365 because this overlay ships first
     for BTC, but `periods_per_year` is configurable.

  4. **Zero-vol fallback.** If realized vol is 0 or negative (can happen
     in degenerate warmup cases), pass raw_target through unchanged
     rather than dividing by zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VolTargetConfig:
    """Configuration for the vol-targeting overlay.

    Attributes
    ----------
    enabled : bool
        If False, `apply_vol_target` returns the raw_target unchanged.
    annual_target : float
        Target annualized portfolio volatility (e.g., 0.20 = 20% vol).
    periods_per_year : int
        Annualization factor. 365 for crypto (BTC trades daily), 252
        for US equities, 260 for FX.
    max_scale : float
        Upper cap on the vol-targeting multiplier. Default 2.0 means
        the overlay can at most *double* the raw target position.
    min_scale : float
        Lower floor on the multiplier. Default 0.0 means the overlay
        can reduce the target to zero but never flip its sign.
    """

    enabled: bool = False
    annual_target: float = 0.20
    periods_per_year: int = 365
    max_scale: float = 2.0
    min_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.annual_target < 0:
            raise ValueError(f"annual_target must be >= 0, got {self.annual_target}")
        if self.periods_per_year <= 0:
            raise ValueError(
                f"periods_per_year must be > 0, got {self.periods_per_year}"
            )
        if self.max_scale < self.min_scale:
            raise ValueError(
                f"max_scale ({self.max_scale}) must be >= min_scale ({self.min_scale})"
            )


def apply_vol_target(
    raw_target: float,
    realized_daily_vol: float,
    config: VolTargetConfig,
) -> float:
    """Return the vol-targeted position given a raw target and realized vol.

    Parameters
    ----------
    raw_target : float
        Position fraction emitted by SignalGenerator (e.g., +1.0 for
        max long, 0.0 for flat, -0.5 for half short).
    realized_daily_vol : float
        Trailing realized daily return stdev (e.g., the engine's
        `volatility` column at bar t). Must be non-negative.
    config : VolTargetConfig
        Overlay configuration.

    Returns
    -------
    float
        Sized target position. Sign is preserved; only magnitude
        changes. If `config.enabled` is False, raw_target is returned
        unchanged.
    """
    if not config.enabled:
        return raw_target
    if raw_target == 0.0:
        return 0.0
    if realized_daily_vol is None or realized_daily_vol <= 0:
        # Degenerate vol reading — pass through rather than produce a
        # nonsense infinite scale.
        return raw_target
    annualized = realized_daily_vol * math.sqrt(config.periods_per_year)
    if annualized <= 0:
        return raw_target
    scale = config.annual_target / annualized
    if scale < config.min_scale:
        scale = config.min_scale
    elif scale > config.max_scale:
        scale = config.max_scale
    return raw_target * scale
