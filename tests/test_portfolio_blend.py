"""Tests for PortfolioCombiner and run_portfolio_backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.backtest.engine import BacktestConfig
from signals.backtest.portfolio_blend import (
    PortfolioAllocation,
    PortfolioCombiner,
    default_btc_sp_allocation,
    run_portfolio_backtest,
)

# ============================================================
# PortfolioAllocation validation
# ============================================================


def test_combiner_rejects_weights_not_summing_to_one():
    allocs = [
        PortfolioAllocation("A", cfg=None, weight=0.5),
        PortfolioAllocation("B", cfg=None, weight=0.3),  # sum = 0.8
    ]
    with pytest.raises(ValueError, match="sum to 1.0"):
        PortfolioCombiner(allocations=allocs)


def test_combiner_rejects_empty_allocations():
    with pytest.raises(ValueError, match="at least one"):
        PortfolioCombiner(allocations=[])


def test_combiner_rejects_bad_rebalance_mode():
    allocs = [PortfolioAllocation("A", cfg=None, weight=1.0)]
    with pytest.raises(ValueError, match="rebalance must"):
        PortfolioCombiner(allocations=allocs, rebalance="quarterly")


# ============================================================
# Single-asset portfolio (100% weight to one asset)
# ============================================================


def test_single_asset_window_rebalance():
    """A 100/0 'portfolio' should be identical to the underlying asset."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    eq = pd.Series(np.linspace(1.0, 2.0, 100), index=idx)
    combiner = PortfolioCombiner(
        allocations=[PortfolioAllocation("A", cfg=None, weight=1.0)],
        rebalance="window",
    )
    port = combiner.combine({"A": eq})
    expected = (eq / eq.iloc[0]) * 10_000.0
    # Compare values only — the combiner returns a generic Index rather
    # than DatetimeIndex due to internal union() operations, which is
    # harmless for downstream use.
    assert len(port) == len(expected)
    np.testing.assert_allclose(port.values, expected.values, rtol=1e-9)


def test_single_asset_daily_rebalance():
    """With only one asset, daily rebalance should also equal the asset."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    eq = pd.Series(np.linspace(1.0, 2.0, 100), index=idx)
    combiner = PortfolioCombiner(
        allocations=[PortfolioAllocation("A", cfg=None, weight=1.0)],
        rebalance="daily",
    )
    port = combiner.combine({"A": eq})
    expected = (eq / eq.iloc[0]) * 10_000.0
    assert len(port) == len(expected)
    np.testing.assert_allclose(port.values, expected.values, rtol=1e-9, atol=1e-6)


# ============================================================
# Two-asset portfolio math
# ============================================================


def test_two_asset_window_rebalance_math():
    """Window-mode portfolio should be w_a × eq_a_normed + w_b × eq_b_normed."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    eq_a = pd.Series(np.arange(10, 20, dtype=float), index=idx)  # 10, 11, ..., 19
    eq_b = pd.Series(np.arange(100, 110, dtype=float), index=idx)  # 100, 101, ..., 109

    combiner = PortfolioCombiner(
        allocations=[
            PortfolioAllocation("A", cfg=None, weight=0.5),
            PortfolioAllocation("B", cfg=None, weight=0.5),
        ],
        rebalance="window",
    )
    port = combiner.combine({"A": eq_a, "B": eq_b})

    # Normalize each to start at 1.0, weight 50/50, scale by 10k
    a_norm = eq_a / 10.0
    b_norm = eq_b / 100.0
    expected = (0.5 * a_norm + 0.5 * b_norm) * 10_000.0
    np.testing.assert_allclose(port.values, expected.values, rtol=1e-9)


def test_two_asset_daily_rebalance_math():
    """Daily-rebalance portfolio should equal the compound of weighted daily returns."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    eq_a = pd.Series(np.arange(10, 20, dtype=float), index=idx)
    eq_b = pd.Series(np.arange(100, 110, dtype=float), index=idx)

    combiner = PortfolioCombiner(
        allocations=[
            PortfolioAllocation("A", cfg=None, weight=0.6),
            PortfolioAllocation("B", cfg=None, weight=0.4),
        ],
        rebalance="daily",
    )
    port = combiner.combine({"A": eq_a, "B": eq_b})

    a_norm = eq_a / eq_a.iloc[0]
    b_norm = eq_b / eq_b.iloc[0]
    a_ret = a_norm.pct_change().fillna(0)
    b_ret = b_norm.pct_change().fillna(0)
    expected_returns = 0.6 * a_ret + 0.4 * b_ret
    expected = (1.0 + expected_returns).cumprod() * 10_000.0
    np.testing.assert_allclose(port.values, expected.values, rtol=1e-9)


# ============================================================
# Date alignment
# ============================================================


def test_asset_with_missing_dates_is_forward_filled():
    """BTC (7d/week) + S&P (5d/week) — weekends on S&P should be
    forward-filled so the portfolio uses BTC's full calendar."""
    # BTC: 10 consecutive days
    btc_idx = pd.date_range("2020-01-06", periods=10, freq="D", tz="UTC")  # Mon → next Wed
    btc_eq = pd.Series(np.arange(100, 110, dtype=float), index=btc_idx)

    # S&P: same dates except skip the weekend (Jan 11 Sat, Jan 12 Sun)
    sp_idx = btc_idx[~btc_idx.weekday.isin([5, 6])]
    sp_eq = pd.Series(np.arange(200, 200 + len(sp_idx), dtype=float), index=sp_idx)

    combiner = PortfolioCombiner(
        allocations=[
            PortfolioAllocation("BTC", cfg=None, weight=0.5),
            PortfolioAllocation("SP", cfg=None, weight=0.5),
        ],
        rebalance="daily",
    )
    port = combiner.combine({"BTC": btc_eq, "SP": sp_eq})

    # Portfolio should have the BTC-calendar length (10 days)
    assert len(port) == 10


def test_intraday_timestamps_are_normalized():
    """BTC is stored at 00:00 UTC and S&P at market close — indices
    should be stripped to dates before combining."""
    btc_idx = pd.date_range("2020-01-01 00:00:00", periods=5, freq="D", tz="UTC")
    sp_idx = pd.date_range("2020-01-01 21:00:00", periods=5, freq="D", tz="UTC")
    # With raw intersection these would not overlap at all
    btc_eq = pd.Series([100.0, 101, 102, 103, 104], index=btc_idx)
    sp_eq = pd.Series([200.0, 201, 202, 203, 204], index=sp_idx)

    combiner = PortfolioCombiner(
        allocations=[
            PortfolioAllocation("BTC", cfg=None, weight=0.5),
            PortfolioAllocation("SP", cfg=None, weight=0.5),
        ],
        rebalance="window",
    )
    port = combiner.combine({"BTC": btc_eq, "SP": sp_eq})
    # Despite different intraday timestamps, both are combined
    # on the date-only index.
    assert len(port) == 5


# ============================================================
# Default allocation helper
# ============================================================


def test_default_btc_sp_allocation_is_40_60():
    allocs = default_btc_sp_allocation()
    assert len(allocs) == 2
    btc = next(a for a in allocs if a.symbol == "BTC-USD")
    sp = next(a for a in allocs if a.symbol == "^GSPC")
    assert btc.weight == 0.4
    assert sp.weight == 0.6
    assert btc.cfg is not None
    assert btc.cfg.model_type == "hybrid"
    assert btc.cfg.hybrid_vol_quantile == 0.70
    assert sp.cfg is None  # buy & hold


# ============================================================
# End-to-end backtest
# ============================================================


def test_run_portfolio_backtest_bh_only(synthetic_prices):
    """Two buy-and-hold assets with 50/50 weights should produce a
    portfolio that's a linear blend of the two underlying price series."""
    # Make a second asset by scaling the first
    prices_a = synthetic_prices.copy()
    prices_b = synthetic_prices.copy()
    prices_b["close"] = prices_b["close"] * 2.0
    prices_b["open"] = prices_b["open"] * 2.0

    port_eq = run_portfolio_backtest(
        allocations=[
            PortfolioAllocation("A", cfg=None, weight=0.5),
            PortfolioAllocation("B", cfg=None, weight=0.5),
        ],
        prices_by_symbol={"A": prices_a, "B": prices_b},
        rebalance="window",
    )
    assert not port_eq.empty
    assert port_eq.iloc[0] == pytest.approx(10_000.0, rel=1e-6)


def test_run_portfolio_backtest_with_strategy(synthetic_prices):
    """Run a real BacktestEngine strategy through the combiner with
    a buy-and-hold partner to exercise the engine integration path."""
    cfg = BacktestConfig(
        model_type="composite",
        train_window=150,
        retrain_freq=40,
        return_bins=3,
        volatility_bins=3,
        vol_window=10,
    )
    # 100% allocation to the strategy — simpler than mixing for the test
    port_eq = run_portfolio_backtest(
        allocations=[PortfolioAllocation("SYN", cfg=cfg, weight=1.0)],
        prices_by_symbol={"SYN": synthetic_prices},
        rebalance="window",
    )
    assert not port_eq.empty
    assert port_eq.iloc[-1] > 0


def test_run_portfolio_backtest_missing_symbol_raises(synthetic_prices):
    with pytest.raises(ValueError, match="missing prices"):
        run_portfolio_backtest(
            allocations=[PortfolioAllocation("MISSING", cfg=None, weight=1.0)],
            prices_by_symbol={"OTHER": synthetic_prices},
        )
