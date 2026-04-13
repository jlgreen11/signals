"""Tests for CrossSectionalMomentum model."""

from __future__ import annotations

import pandas as pd

from signals.model.momentum import CrossSectionalMomentum


def _make_price_df(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Build a minimal DataFrame with a 'close' column and UTC DatetimeIndex."""
    dates = pd.bdate_range(start=start, periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame({"close": prices}, index=dates)


def _linear_prices(start: float, end: float, n: int) -> list[float]:
    """Generate n linearly spaced prices from start to end."""
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


class TestRanking:
    """Tests for the rank() method."""

    def test_top_n_selection(self) -> None:
        """Highest-return stocks should be selected as winners."""
        n = 300  # enough history for lookback=252 + skip=21
        prices = {
            "WINNER1": _make_price_df(_linear_prices(100, 200, n)),  # +100%
            "WINNER2": _make_price_df(_linear_prices(100, 180, n)),  # +80%
            "LOSER1": _make_price_df(_linear_prices(100, 110, n)),   # +10%
            "LOSER2": _make_price_df(_linear_prices(100, 90, n)),    # -10%
        }
        mom = CrossSectionalMomentum(lookback_days=252, skip_days=21, n_long=2,
                                     mode="classic")
        as_of = prices["WINNER1"].index[-1]
        weights = mom.rank(prices, as_of_date=as_of)

        # Top 2 should be WINNER1 and WINNER2
        assert weights["WINNER1"] > 0
        assert weights["WINNER2"] > 0
        assert weights["LOSER1"] == 0.0
        assert weights["LOSER2"] == 0.0

    def test_weights_sum_to_one(self) -> None:
        """Weights across the portfolio should sum to 1.0."""
        n = 300
        prices = {
            f"STOCK{i}": _make_price_df(
                _linear_prices(100, 100 + i * 10, n)
            )
            for i in range(10)
        }
        mom = CrossSectionalMomentum(n_long=3, mode="classic")
        as_of = list(prices.values())[0].index[-1]
        weights = mom.rank(prices, as_of_date=as_of)

        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_different_lookbacks_produce_different_rankings(self) -> None:
        """Changing the lookback period should (in general) change which stocks win."""
        # Stock A: crashed early, then recovered sharply in the last ~120 days
        # (strong short-term momentum, but negative over long horizon)
        prices_a = _linear_prices(100, 50, 380) + _linear_prices(50, 95, 120)
        # Stock B: steady slow climb over the whole period
        # (weaker short-term momentum, but positive over long horizon)
        prices_b = _linear_prices(100, 160, 500)

        n = 500
        prices = {
            "A": _make_price_df(prices_a[:n]),
            "B": _make_price_df(prices_b[:n]),
        }
        as_of = list(prices.values())[0].index[-1]

        # Short lookback (100 days, skip 5): A's recent surge dominates
        mom_short = CrossSectionalMomentum(lookback_days=100, skip_days=5, n_long=1,
                                           mode="classic")
        # Long lookback (400 days, skip 5): B's steady climb dominates
        mom_long = CrossSectionalMomentum(lookback_days=400, skip_days=5, n_long=1,
                                          mode="classic")

        weights_short = mom_short.rank(prices, as_of_date=as_of)
        weights_long = mom_long.rank(prices, as_of_date=as_of)

        # With different lookbacks, the winner should differ
        winner_short = max(weights_short, key=weights_short.get)  # type: ignore[arg-type]
        winner_long = max(weights_long, key=weights_long.get)  # type: ignore[arg-type]
        assert winner_short != winner_long, (
            "Different lookback periods should produce different rankings for this data"
        )

    def test_insufficient_history_excluded(self) -> None:
        """Stocks without enough history should be excluded from ranking."""
        prices = {
            "ENOUGH": _make_price_df(_linear_prices(100, 200, 300)),
            "TOO_SHORT": _make_price_df(_linear_prices(100, 200, 50)),
        }
        mom = CrossSectionalMomentum(lookback_days=252, skip_days=21, n_long=2,
                                     mode="classic")
        as_of = prices["ENOUGH"].index[-1]
        weights = mom.rank(prices, as_of_date=as_of)

        # ENOUGH should get all the weight, TOO_SHORT gets zero
        assert weights["ENOUGH"] > 0
        assert weights["TOO_SHORT"] == 0.0


class TestBacktest:
    """Tests for the backtest() method."""

    def test_equity_starts_at_initial_cash(self) -> None:
        """First point of equity curve should equal initial_cash."""
        n = 400
        prices = {
            f"S{i}": _make_price_df(_linear_prices(100, 100 + i * 5, n))
            for i in range(5)
        }
        mom = CrossSectionalMomentum(lookback_days=50, skip_days=5, n_long=2,
                                     rebalance_freq=21, mode="classic")
        start = str(prices["S0"].index[0].date())
        end = str(prices["S0"].index[-1].date())
        equity = mom.backtest(prices, start=start, end=end, initial_cash=10000.0)

        assert len(equity) > 0
        assert abs(equity.iloc[0] - 10000.0) < 1.0, (
            f"First equity point {equity.iloc[0]} should be ~10000"
        )

    def test_equity_index_is_monotonically_increasing(self) -> None:
        """Equity curve index should be sorted (no time travel)."""
        n = 400
        prices = {
            f"S{i}": _make_price_df(_linear_prices(100, 100 + i * 5, n))
            for i in range(5)
        }
        mom = CrossSectionalMomentum(lookback_days=50, skip_days=5, n_long=2,
                                     mode="classic")
        start = str(prices["S0"].index[0].date())
        end = str(prices["S0"].index[-1].date())
        equity = mom.backtest(prices, start=start, end=end)

        assert equity.index.is_monotonic_increasing

    def test_backtest_with_costs_less_than_without(self) -> None:
        """Transaction costs should reduce total return."""
        n = 400
        prices = {
            f"S{i}": _make_price_df(_linear_prices(100, 100 + i * 10, n))
            for i in range(6)
        }
        start = str(prices["S0"].index[0].date())
        end = str(prices["S0"].index[-1].date())

        mom_costly = CrossSectionalMomentum(
            lookback_days=50, skip_days=5, n_long=2, rebalance_freq=21,
            commission_bps=10.0, slippage_bps=10.0, mode="classic",
        )
        mom_free = CrossSectionalMomentum(
            lookback_days=50, skip_days=5, n_long=2, rebalance_freq=21,
            commission_bps=0.0, slippage_bps=0.0, mode="classic",
        )

        eq_costly = mom_costly.backtest(prices, start=start, end=end)
        eq_free = mom_free.backtest(prices, start=start, end=end)

        assert eq_costly.iloc[-1] < eq_free.iloc[-1], (
            "Portfolio with costs should end with less equity than costless"
        )

    def test_empty_universe_returns_empty(self) -> None:
        """Backtest on empty universe should return empty Series."""
        mom = CrossSectionalMomentum(mode="classic", lookback_days=252)
        equity = mom.backtest({}, start="2020-01-01", end="2020-12-31")
        assert len(equity) == 0


class TestEarlyBreakout:
    """Tests for the early_breakout mode."""

    def test_acceleration_ranking(self) -> None:
        """Stocks accelerating recently should rank higher than steady climbers."""
        n = 300
        # Stock A: flat for 250 days, then surges +30% in last 50 days
        prices_a = [100.0] * 250 + _linear_prices(100, 135, 50)
        # Stock B: steady climb the entire time (higher long-term, lower accel)
        prices_b = _linear_prices(100, 140, n)
        prices = {
            "ACCEL": _make_price_df(prices_a),
            "STEADY": _make_price_df(prices_b),
        }
        mom = CrossSectionalMomentum(mode="early_breakout", n_long=1,
                                     lookback_days=126, short_lookback=21,
                                     min_short_return=0.05, max_12m_return=2.0)
        as_of = prices["ACCEL"].index[-1]
        weights = mom.rank(prices, as_of_date=as_of)
        assert weights["ACCEL"] > 0, "Accelerating stock should be selected"

    def test_moonshot_filter(self) -> None:
        """Stocks with >150% trailing return should be filtered."""
        n = 300
        # Stock A: 200% return (filtered out at max_12m_return=1.5)
        prices_a = _linear_prices(100, 300, n)
        # Stock B: 40% return with recent surge
        prices_b = [100.0] * 250 + _linear_prices(100, 130, 50)
        prices = {
            "MOONSHOT": _make_price_df(prices_a),
            "MODERATE": _make_price_df(prices_b),
        }
        mom = CrossSectionalMomentum(mode="early_breakout", n_long=2,
                                     lookback_days=126, short_lookback=21,
                                     min_short_return=0.05, max_12m_return=1.5)
        as_of = prices["MOONSHOT"].index[-1]
        weights = mom.rank(prices, as_of_date=as_of)
        assert weights["MOONSHOT"] == 0.0, "Moonshot should be filtered"
        assert weights["MODERATE"] > 0, "Moderate stock should be selected"

    def test_sector_cap(self) -> None:
        """Max per-sector cap should be enforced."""
        n = 300
        prices = {}
        # 5 tech stocks with varying acceleration (strong surges in last 50d)
        for i in range(5):
            flat = [100.0] * 250
            surge = _linear_prices(100, 125 + i * 10, 50)
            prices[f"TECH{i}"] = _make_price_df(flat + surge)

        sectors = {f"TECH{i}": "Information Technology" for i in range(5)}

        mom = CrossSectionalMomentum(mode="early_breakout", n_long=5,
                                     lookback_days=126, short_lookback=21,
                                     min_short_return=0.05, max_per_sector=2)
        as_of = prices["TECH0"].index[-1]
        weights = mom.rank(prices, as_of_date=as_of, sectors=sectors)

        selected = [t for t, w in weights.items() if w > 0]
        assert len(selected) == 2, f"Should select max 2 from same sector, got {len(selected)}"
