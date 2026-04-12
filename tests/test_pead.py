"""Tests for PEAD (Post-Earnings Announcement Drift) strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.data.earnings import compute_surprise
from signals.model.pead import PEADStrategy, summarize_trades

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(
    start: str = "2023-01-01",
    n_days: int = 200,
    initial_price: float = 100.0,
    daily_drift: float = 0.0005,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic price DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, freq="B", tz="UTC")
    rets = rng.normal(daily_drift, 0.015, size=n_days)
    rets[0] = 0.0
    close = initial_price * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.uniform(-0.005, 0.005, n_days)),
            "high": close * (1 + rng.uniform(0, 0.01, n_days)),
            "low": close * (1 - rng.uniform(0, 0.01, n_days)),
            "close": close,
            "volume": rng.uniform(1e6, 5e6, n_days),
        },
        index=idx,
    )


def _make_earnings(
    ticker: str,
    dates: list[str],
    surprises_pct: list[float],
) -> pd.DataFrame:
    """Create a synthetic earnings DataFrame."""
    n = len(dates)
    return pd.DataFrame({
        "ticker": [ticker] * n,
        "report_date": pd.to_datetime(dates),
        "actual_eps": [1.5] * n,
        "estimated_eps": [1.5 / (1 + s / 100) for s in surprises_pct],
        "surprise": [1.5 - 1.5 / (1 + s / 100) for s in surprises_pct],
        "surprise_pct": surprises_pct,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeSurprise:
    """Test surprise calculation utility."""

    def test_positive_surprise(self):
        surprise, pct = compute_surprise(actual_eps=1.50, estimated_eps=1.00)
        assert surprise == pytest.approx(0.50)
        assert pct == pytest.approx(50.0)

    def test_negative_surprise(self):
        surprise, pct = compute_surprise(actual_eps=0.80, estimated_eps=1.00)
        assert surprise == pytest.approx(-0.20)
        assert pct == pytest.approx(-20.0)

    def test_zero_estimate_no_division_error(self):
        surprise, pct = compute_surprise(actual_eps=0.50, estimated_eps=0.0)
        assert surprise == pytest.approx(0.50)
        assert pct == pytest.approx(0.0)  # graceful fallback

    def test_exact_match(self):
        surprise, pct = compute_surprise(actual_eps=1.00, estimated_eps=1.00)
        assert surprise == pytest.approx(0.0)
        assert pct == pytest.approx(0.0)


class TestPEADTradeGeneration:
    """Test trade generation on synthetic data."""

    def test_positive_surprise_generates_long(self):
        prices = _make_prices()
        earnings = _make_earnings("TEST", ["2023-02-01"], [10.0])
        strategy = PEADStrategy(surprise_threshold_pct=5.0, hold_days=60)
        trades = strategy.generate_trades(earnings, {"TEST": prices})
        assert len(trades) == 1
        assert trades.iloc[0]["direction"] == "long"
        assert trades.iloc[0]["ticker"] == "TEST"

    def test_below_threshold_no_trade(self):
        prices = _make_prices()
        earnings = _make_earnings("TEST", ["2023-02-01"], [2.0])
        strategy = PEADStrategy(surprise_threshold_pct=5.0, hold_days=60)
        trades = strategy.generate_trades(earnings, {"TEST": prices})
        assert len(trades) == 0

    def test_negative_surprise_no_trade(self):
        """Long-only: negative surprises should NOT generate trades."""
        prices = _make_prices()
        earnings = _make_earnings("TEST", ["2023-02-01"], [-15.0])
        strategy = PEADStrategy(surprise_threshold_pct=5.0, hold_days=60)
        trades = strategy.generate_trades(earnings, {"TEST": prices})
        assert len(trades) == 0

    def test_multiple_events_multiple_trades(self):
        prices = _make_prices(n_days=300)
        earnings = _make_earnings(
            "TEST",
            ["2023-02-01", "2023-05-01", "2023-08-01"],
            [8.0, 12.0, 6.0],
        )
        strategy = PEADStrategy(surprise_threshold_pct=5.0, hold_days=30)
        trades = strategy.generate_trades(earnings, {"TEST": prices})
        assert len(trades) == 3

    def test_hold_days_respected(self):
        prices = _make_prices()
        earnings = _make_earnings("TEST", ["2023-02-01"], [10.0])
        strategy = PEADStrategy(surprise_threshold_pct=5.0, hold_days=20)
        trades = strategy.generate_trades(earnings, {"TEST": prices})
        assert len(trades) == 1
        entry = trades.iloc[0]["entry_date"]
        exit_ = trades.iloc[0]["exit_date"]
        # The exit should be ~20 trading days after entry
        entry_loc = prices.index.get_loc(entry)
        exit_loc = prices.index.get_loc(exit_)
        assert exit_loc - entry_loc == 20

    def test_net_return_accounts_for_costs(self):
        prices = _make_prices()
        earnings = _make_earnings("TEST", ["2023-02-01"], [10.0])
        strategy = PEADStrategy(
            surprise_threshold_pct=5.0, hold_days=60, cost_bps=5.0
        )
        trades = strategy.generate_trades(earnings, {"TEST": prices})
        assert len(trades) == 1
        row = trades.iloc[0]
        expected_net = row["gross_return"] - 2 * (5.0 / 10_000.0)
        assert row["net_return"] == pytest.approx(expected_net, abs=1e-10)


class TestPEADBacktest:
    """Test backtest equity curve properties."""

    def test_equity_starts_at_initial_cash(self):
        prices = _make_prices()
        earnings = _make_earnings("TEST", ["2023-02-01"], [10.0])
        strategy = PEADStrategy(surprise_threshold_pct=5.0, hold_days=60)
        equity = strategy.backtest(
            earnings, {"TEST": prices},
            start="2023-01-01", end="2023-12-01",
            initial_cash=10_000.0,
        )
        assert len(equity) > 0
        assert equity.iloc[0] == pytest.approx(10_000.0)

    def test_equity_never_negative(self):
        prices = _make_prices()
        earnings = _make_earnings(
            "TEST",
            ["2023-02-01", "2023-05-01"],
            [15.0, 20.0],
        )
        strategy = PEADStrategy(surprise_threshold_pct=5.0, hold_days=60)
        equity = strategy.backtest(
            earnings, {"TEST": prices},
            start="2023-01-01", end="2023-12-01",
        )
        assert (equity >= 0).all()

    def test_no_earnings_returns_flat_equity(self):
        """With no qualifying earnings, equity should stay at initial cash."""
        prices = _make_prices()
        empty_earnings = pd.DataFrame(
            columns=["ticker", "report_date", "actual_eps",
                      "estimated_eps", "surprise", "surprise_pct"]
        )
        strategy = PEADStrategy(surprise_threshold_pct=5.0, hold_days=60)
        equity = strategy.backtest(
            empty_earnings, {"TEST": prices},
            start="2023-01-01", end="2023-12-01",
        )
        assert len(equity) > 0
        # All values should be initial cash (no trades)
        assert (equity == equity.iloc[0]).all()

    def test_multi_ticker_backtest(self):
        prices_a = _make_prices(seed=42)
        prices_b = _make_prices(seed=99)
        earnings = pd.concat([
            _make_earnings("A", ["2023-03-01"], [12.0]),
            _make_earnings("B", ["2023-04-01"], [8.0]),
        ], ignore_index=True)
        strategy = PEADStrategy(
            surprise_threshold_pct=5.0, hold_days=30, max_positions=5
        )
        equity = strategy.backtest(
            earnings, {"A": prices_a, "B": prices_b},
            start="2023-01-01", end="2023-12-01",
        )
        assert len(equity) > 0
        assert equity.iloc[0] == pytest.approx(10_000.0)


class TestSummarizeTradesHelper:
    """Test the summarize_trades utility."""

    def test_empty_trades(self):
        stats = summarize_trades(pd.DataFrame())
        assert stats.n_trades == 0
        assert stats.win_rate == 0.0

    def test_summary_counts(self):
        trades = pd.DataFrame({
            "net_return": [0.05, -0.02, 0.10, -0.01, 0.03],
            "gross_return": [0.06, -0.01, 0.11, 0.0, 0.04],
        })
        stats = summarize_trades(trades)
        assert stats.n_trades == 5
        assert stats.n_winners == 3
        assert stats.n_losers == 2
        assert stats.win_rate == pytest.approx(0.6)
