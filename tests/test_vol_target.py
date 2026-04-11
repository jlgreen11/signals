"""Tests for the volatility-targeting overlay."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.vol_target import VolTargetConfig, apply_vol_target

# ---------------------------------------------------------------------------
# Unit tests for apply_vol_target
# ---------------------------------------------------------------------------


def test_disabled_returns_raw_target_unchanged() -> None:
    cfg = VolTargetConfig(enabled=False, annual_target=0.20)
    assert apply_vol_target(1.0, realized_daily_vol=0.05, config=cfg) == 1.0
    assert apply_vol_target(-0.5, realized_daily_vol=0.05, config=cfg) == -0.5


def test_zero_target_is_passthrough() -> None:
    cfg = VolTargetConfig(enabled=True, annual_target=0.20)
    assert apply_vol_target(0.0, realized_daily_vol=0.05, config=cfg) == 0.0


def test_zero_vol_is_passthrough() -> None:
    cfg = VolTargetConfig(enabled=True, annual_target=0.20)
    assert apply_vol_target(1.0, realized_daily_vol=0.0, config=cfg) == 1.0
    assert apply_vol_target(1.0, realized_daily_vol=-0.01, config=cfg) == 1.0


def test_high_vol_reduces_position() -> None:
    """Realized vol of 80% annualized with 20% target → 0.25x scale."""
    cfg = VolTargetConfig(
        enabled=True,
        annual_target=0.20,
        periods_per_year=365,
        max_scale=2.0,
        min_scale=0.0,
    )
    daily_vol = 0.80 / math.sqrt(365)
    sized = apply_vol_target(1.0, daily_vol, cfg)
    assert sized == pytest.approx(0.25, rel=1e-6)


def test_low_vol_increases_position_subject_to_cap() -> None:
    """10% realized vs 20% target → uncapped scale 2.0, capped at 2.0."""
    cfg = VolTargetConfig(
        enabled=True,
        annual_target=0.20,
        periods_per_year=365,
        max_scale=2.0,
    )
    daily_vol = 0.10 / math.sqrt(365)
    sized = apply_vol_target(1.0, daily_vol, cfg)
    assert sized == pytest.approx(2.0, rel=1e-6)


def test_very_low_vol_hits_cap() -> None:
    """5% realized vs 20% target would be 4.0x, but cap enforces 2.0x."""
    cfg = VolTargetConfig(
        enabled=True, annual_target=0.20, periods_per_year=365, max_scale=2.0
    )
    daily_vol = 0.05 / math.sqrt(365)
    sized = apply_vol_target(1.0, daily_vol, cfg)
    assert sized == pytest.approx(2.0, rel=1e-6)


def test_negative_target_preserves_sign() -> None:
    cfg = VolTargetConfig(
        enabled=True, annual_target=0.20, periods_per_year=365, max_scale=2.0
    )
    daily_vol = 0.40 / math.sqrt(365)  # 2x over target → scale 0.5
    sized = apply_vol_target(-1.0, daily_vol, cfg)
    assert sized == pytest.approx(-0.5, rel=1e-6)


def test_min_scale_floor() -> None:
    cfg = VolTargetConfig(
        enabled=True,
        annual_target=0.20,
        periods_per_year=365,
        max_scale=2.0,
        min_scale=0.3,
    )
    daily_vol = 2.0 / math.sqrt(365)  # 200% vol → uncapped would be 0.1
    sized = apply_vol_target(1.0, daily_vol, cfg)
    assert sized == pytest.approx(0.3, rel=1e-6)


def test_annualization_factor_matters() -> None:
    """Changing periods_per_year from 252 to 365 changes the annualized vol."""
    daily_vol = 0.02  # 2% daily
    cfg252 = VolTargetConfig(
        enabled=True, annual_target=0.20, periods_per_year=252, max_scale=10.0
    )
    cfg365 = VolTargetConfig(
        enabled=True, annual_target=0.20, periods_per_year=365, max_scale=10.0
    )
    sized252 = apply_vol_target(1.0, daily_vol, cfg252)
    sized365 = apply_vol_target(1.0, daily_vol, cfg365)
    # 365 annualization produces higher annualized vol → smaller scale
    assert sized365 < sized252


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        VolTargetConfig(annual_target=-0.1)
    with pytest.raises(ValueError):
        VolTargetConfig(periods_per_year=0)
    with pytest.raises(ValueError):
        VolTargetConfig(max_scale=1.0, min_scale=2.0)


# ---------------------------------------------------------------------------
# Integration test: BacktestConfig pass-through
# ---------------------------------------------------------------------------


def _make_prices(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.001, 0.02, size=n)
    prices = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def test_engine_runs_with_vol_target_enabled() -> None:
    """Smoke test: engine produces a valid BacktestResult with overlay on."""
    cfg = BacktestConfig(
        model_type="composite",
        train_window=100,
        retrain_freq=21,
        return_bins=3,
        volatility_bins=3,
        vol_window=10,
        laplace_alpha=0.01,
        vol_target_enabled=True,
        vol_target_annual=0.20,
        vol_target_periods_per_year=365,
    )
    prices = _make_prices()
    result = BacktestEngine(cfg).run(prices, symbol="TEST")
    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0


def test_engine_disabled_matches_baseline() -> None:
    """With vol_target_enabled=False, equity curve must be identical to baseline."""
    base = BacktestConfig(
        model_type="composite",
        train_window=100,
        retrain_freq=21,
        return_bins=3,
        volatility_bins=3,
        vol_window=10,
        laplace_alpha=0.01,
    )
    overlay = BacktestConfig(
        model_type="composite",
        train_window=100,
        retrain_freq=21,
        return_bins=3,
        volatility_bins=3,
        vol_window=10,
        laplace_alpha=0.01,
        vol_target_enabled=False,  # explicit no-op
    )
    prices = _make_prices()
    r_base = BacktestEngine(base).run(prices, symbol="TEST")
    r_over = BacktestEngine(overlay).run(prices, symbol="TEST")
    pd.testing.assert_series_equal(
        r_base.equity_curve, r_over.equity_curve, check_names=False
    )


# ---------------------------------------------------------------------------
# Lookahead regression — splitting the data at bar k must not change any
# overlay-modified target before bar k.
# ---------------------------------------------------------------------------


def test_vol_target_no_lookahead() -> None:
    """Overlay uses only realized vol at bar t, so target at bar t must
    be identical whether the engine runs on prices[0:t+5] or prices[0:n]."""
    cfg = BacktestConfig(
        model_type="composite",
        train_window=100,
        retrain_freq=21,
        return_bins=3,
        volatility_bins=3,
        vol_window=10,
        laplace_alpha=0.01,
        vol_target_enabled=True,
        vol_target_annual=0.20,
        vol_target_periods_per_year=365,
    )
    prices_full = _make_prices(n=400)
    prices_short = prices_full.iloc[:200]

    r_full = BacktestEngine(cfg).run(prices_full, symbol="TEST")
    r_short = BacktestEngine(cfg).run(prices_short, symbol="TEST")

    overlap = r_full.signals.index.intersection(r_short.signals.index)
    assert len(overlap) > 0
    pd.testing.assert_series_equal(
        r_full.signals.loc[overlap, "target_position"],
        r_short.signals.loc[overlap, "target_position"],
        check_names=False,
    )
