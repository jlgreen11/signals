"""Tests for TimeSeriesMomentum model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.model.tsmom import TimeSeriesMomentum


def _make_prices(
    n: int = 300,
    drift: float = 0.001,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate a dict of synthetic price DataFrames for 3 assets."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    result: dict[str, pd.DataFrame] = {}
    for name, d in [("UP", drift), ("DOWN", -drift), ("FLAT", 0.0)]:
        rets = rng.normal(d, 0.01, size=n)
        close = 100.0 * np.exp(np.cumsum(rets))
        result[name] = pd.DataFrame({"close": close}, index=idx)
    return result


# ============================================================
# Signal computation
# ============================================================


def test_signal_positive_for_uptrend():
    """An asset with positive trailing return should get positive weight."""
    prices = _make_prices(n=300, drift=0.002)
    model = TimeSeriesMomentum(lookback_days=63, vol_window=21, risk_parity=False)
    as_of = prices["UP"].index[-1]
    weights = model.signals(prices, as_of)
    assert weights["UP"] > 0, "Uptrending asset should have positive weight"


def test_signal_flat_for_downtrend():
    """An asset with negative trailing return should get zero weight."""
    prices = _make_prices(n=300, drift=0.002)
    model = TimeSeriesMomentum(lookback_days=63, vol_window=21, risk_parity=False)
    as_of = prices["DOWN"].index[-1]
    weights = model.signals(prices, as_of)
    assert weights["DOWN"] == 0.0, "Downtrending asset should be flat"


def test_signal_weights_sum_to_one_or_zero():
    """Weights across all assets should sum to 1.0 (or 0 if all flat)."""
    prices = _make_prices(n=300, drift=0.002)
    model = TimeSeriesMomentum(lookback_days=63, vol_window=21)
    as_of = prices["UP"].index[-1]
    weights = model.signals(prices, as_of)
    total = sum(weights.values())
    # Either all zero (unlikely with an up-drifting asset) or sum to 1
    assert total == pytest.approx(0.0) or total == pytest.approx(1.0, abs=1e-10)


def test_risk_parity_weights_differ_from_equal():
    """With risk_parity=True, weights should differ from equal-weight."""
    # Create assets with very different volatilities
    rng = np.random.default_rng(99)
    idx = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    prices: dict[str, pd.DataFrame] = {}
    # Low-vol asset (strong uptrend)
    close_low = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.005, 300)))
    prices["LOWVOL"] = pd.DataFrame({"close": close_low}, index=idx)
    # High-vol asset (strong uptrend)
    close_high = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.04, 300)))
    prices["HIGHVOL"] = pd.DataFrame({"close": close_high}, index=idx)

    model_rp = TimeSeriesMomentum(lookback_days=63, vol_window=21, risk_parity=True)
    model_eq = TimeSeriesMomentum(lookback_days=63, vol_window=21, risk_parity=False)
    as_of = idx[-1]
    w_rp = model_rp.signals(prices, as_of)
    w_eq = model_eq.signals(prices, as_of)

    # Both assets trending up, so both should have positive weight
    # But risk-parity should give MORE weight to the low-vol asset
    if w_rp["LOWVOL"] > 0 and w_rp["HIGHVOL"] > 0:
        assert w_rp["LOWVOL"] > w_rp["HIGHVOL"], (
            "Risk-parity should overweight the low-vol asset"
        )
        # Equal-weight should be 50/50
        if w_eq["LOWVOL"] > 0 and w_eq["HIGHVOL"] > 0:
            assert w_eq["LOWVOL"] == pytest.approx(w_eq["HIGHVOL"], abs=1e-10)


def test_insufficient_history_returns_zero_weight():
    """Assets without enough history should get zero weight."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    close = np.linspace(100, 110, 10)
    prices = {"SHORT": pd.DataFrame({"close": close}, index=idx)}
    model = TimeSeriesMomentum(lookback_days=252, vol_window=63)
    weights = model.signals(prices, idx[-1])
    assert weights["SHORT"] == 0.0


# ============================================================
# Backtest
# ============================================================


def test_backtest_returns_series():
    """Backtest should return a non-empty Series with correct index bounds."""
    prices = _make_prices(n=400, drift=0.001)
    model = TimeSeriesMomentum(lookback_days=63, vol_window=21, rebalance_freq=21)
    start = prices["UP"].index[0]
    end = prices["UP"].index[-1]
    eq = model.backtest(prices, start, end, initial_cash=10_000.0)
    assert isinstance(eq, pd.Series)
    assert len(eq) > 100
    assert eq.iloc[0] == 10_000.0


def test_backtest_equity_never_negative():
    """With long-only positions, equity should never go negative."""
    prices = _make_prices(n=400, drift=-0.001)  # Even with down-drifting assets
    model = TimeSeriesMomentum(lookback_days=63, vol_window=21, rebalance_freq=21)
    start = prices["UP"].index[0]
    end = prices["UP"].index[-1]
    eq = model.backtest(prices, start, end, initial_cash=10_000.0)
    assert (eq >= 0).all(), "Equity should never be negative"


def test_backtest_all_flat_stays_at_initial():
    """If all assets are flat/down the whole time, equity stays near initial."""
    rng = np.random.default_rng(77)
    idx = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    # All assets declining — model should be 100% cash
    prices: dict[str, pd.DataFrame] = {}
    for name in ["A", "B", "C"]:
        close = 100.0 * np.exp(np.cumsum(rng.normal(-0.005, 0.005, 400)))
        prices[name] = pd.DataFrame({"close": close}, index=idx)

    model = TimeSeriesMomentum(lookback_days=63, vol_window=21, rebalance_freq=21)
    eq = model.backtest(prices, idx[0], idx[-1], initial_cash=10_000.0)
    # Should be very close to initial — no positions taken
    if len(eq) > 0:
        assert eq.iloc[-1] == pytest.approx(10_000.0, rel=0.01)


# ============================================================
# Parameter validation
# ============================================================


def test_rejects_bad_lookback():
    with pytest.raises(ValueError, match="lookback_days"):
        TimeSeriesMomentum(lookback_days=0)


def test_rejects_bad_vol_window():
    with pytest.raises(ValueError, match="vol_window"):
        TimeSeriesMomentum(vol_window=1)
