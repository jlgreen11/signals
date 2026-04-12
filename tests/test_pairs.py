"""Tests for the pairs trading strategy module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.model.pairs import PairsTrading


@pytest.fixture
def cointegrated_pair() -> dict[str, pd.Series]:
    """Two synthetic cointegrated series: B is a random walk, A = 2*B + noise."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")

    # B is a random walk
    b_returns = rng.normal(0.0003, 0.01, size=n)
    b_prices = 100 * np.exp(np.cumsum(b_returns))

    # A is cointegrated with B: A = 2*B + stationary noise
    noise = rng.normal(0, 1.0, size=n)
    # Make noise mean-reverting (AR(1) with phi < 1)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.5)
    a_prices = 2.0 * b_prices + noise + 50.0

    return {
        "A": pd.Series(a_prices, index=idx, name="A"),
        "B": pd.Series(b_prices, index=idx, name="B"),
    }


@pytest.fixture
def independent_walks() -> dict[str, pd.Series]:
    """Two independent random walks -- should NOT be cointegrated."""
    rng = np.random.default_rng(123)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")

    a_returns = rng.normal(0.0005, 0.015, size=n)
    b_returns = rng.normal(-0.0002, 0.012, size=n)
    a_prices = 100 * np.exp(np.cumsum(a_returns))
    b_prices = 50 * np.exp(np.cumsum(b_returns))

    return {
        "A": pd.Series(a_prices, index=idx, name="A"),
        "B": pd.Series(b_prices, index=idx, name="B"),
    }


class TestCointegrationDetection:
    def test_finds_cointegrated_pair(self, cointegrated_pair):
        """Should detect the cointegrated pair with p < 0.05."""
        pt = PairsTrading(coint_pvalue=0.05, lookback=400)
        pairs = pt.find_pairs(cointegrated_pair)
        assert len(pairs) == 1
        pair = pairs[0]
        assert {pair.stock_a, pair.stock_b} == {"A", "B"}
        assert pair.pvalue < 0.05

    def test_rejects_independent_walks(self, independent_walks):
        """Should NOT find cointegration between independent random walks."""
        pt = PairsTrading(coint_pvalue=0.05, lookback=400)
        pairs = pt.find_pairs(independent_walks)
        assert len(pairs) == 0

    def test_max_pairs_caps_output(self, cointegrated_pair):
        """max_pairs should limit the number of returned pairs."""
        pt = PairsTrading(coint_pvalue=0.99, lookback=400, max_pairs=1)
        pairs = pt.find_pairs(cointegrated_pair)
        assert len(pairs) <= 1


class TestHedgeRatio:
    def test_ols_hedge_ratio_on_simple_data(self):
        """If A = 2*B + 10 exactly, hedge ratio should be ~2.0, intercept ~10.0."""
        rng = np.random.default_rng(7)
        n = 200
        b = np.linspace(50, 150, n)
        a = 2.0 * b + 10.0 + rng.normal(0, 0.01, n)  # tiny noise

        beta, alpha = PairsTrading._ols_hedge_ratio(a, b)
        assert abs(beta - 2.0) < 0.01, f"hedge ratio {beta} too far from 2.0"
        assert abs(alpha - 10.0) < 0.5, f"intercept {alpha} too far from 10.0"


class TestZscoreSignals:
    def test_zscore_exceeds_entry_threshold(self):
        """Z-score should breach entry threshold when spread diverges."""
        rng = np.random.default_rng(42)
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")

        # Construct a spread that is stationary but has a spike at the end
        spread_values = rng.normal(0, 1, size=n)
        # Force a large positive spike at the end
        spread_values[-5:] = 4.0

        spread = pd.Series(spread_values, index=idx)

        pt = PairsTrading(entry_zscore=2.0, exit_zscore=0.5, zscore_window=60)
        zscore = pt.compute_zscore(spread)

        # The tail should have Z > entry threshold
        tail_z = zscore.iloc[-1]
        assert not pd.isna(tail_z)
        assert abs(tail_z) > pt.entry_zscore, (
            f"Z-score {tail_z} should exceed entry threshold {pt.entry_zscore}"
        )

    def test_zscore_within_exit_threshold(self):
        """Z-score should be near zero for a stationary zero-mean series."""
        rng = np.random.default_rng(99)
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")

        spread_values = rng.normal(0, 1, size=n)
        spread = pd.Series(spread_values, index=idx)

        pt = PairsTrading(zscore_window=60)
        zscore = pt.compute_zscore(spread)

        # For a truly random stationary series, most Z-scores should be < 2
        valid_z = zscore.dropna()
        frac_below_2 = (valid_z.abs() < 2.0).mean()
        assert frac_below_2 > 0.80, (
            f"Expected >80% of Z-scores to be < 2.0 for stationary noise, got {frac_below_2:.0%}"
        )


class TestBacktest:
    def test_backtest_produces_equity_curve(self, cointegrated_pair):
        """Backtest should return a non-empty equity curve."""
        pt = PairsTrading(lookback=200, zscore_window=40, max_pairs=3)
        start = cointegrated_pair["A"].index[0]
        end = cointegrated_pair["A"].index[-1]

        result = pt.backtest(cointegrated_pair, start=start, end=end, initial_cash=10_000)
        assert not result.equity_curve.empty
        assert result.equity_curve.iloc[0] > 0

    def test_market_neutral_low_beta(self):
        """Pairs strategy equity curve should have low correlation to the market."""
        rng = np.random.default_rng(42)
        n = 800
        idx = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC")

        # Market factor
        market_returns = rng.normal(0.0004, 0.012, size=n)
        market = 100 * np.exp(np.cumsum(market_returns))

        # Stock C follows market closely
        c_prices = market * (1 + rng.normal(0, 0.005, size=n))
        # Stock D = 1.5 * C + stationary noise (cointegrated with C)
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.3)
        d_prices = 1.5 * c_prices + noise + 20

        prices = {
            "C": pd.Series(c_prices, index=idx),
            "D": pd.Series(d_prices, index=idx),
        }

        pt = PairsTrading(
            lookback=200, zscore_window=40, max_pairs=3,
            entry_zscore=1.5, exit_zscore=0.3,
        )
        result = pt.backtest(
            prices,
            start=idx[0],
            end=idx[-1],
            initial_cash=10_000,
        )

        if result.equity_curve.empty or len(result.equity_curve) < 10:
            pytest.skip("Backtest produced too few equity points for correlation test")

        # Compute correlation between strategy returns and market returns
        strat_returns = result.equity_curve.pct_change().dropna()
        market_series = pd.Series(market, index=idx)
        market_rets = market_series.pct_change().dropna()

        # Align
        common = strat_returns.index.intersection(market_rets.index)
        if len(common) < 20:
            pytest.skip("Not enough overlapping dates for correlation")

        corr = strat_returns.loc[common].corr(market_rets.loc[common])
        # Market-neutral strategy should have |correlation| < 0.5
        assert abs(corr) < 0.5, (
            f"Strategy-market correlation {corr:.3f} is too high for a market-neutral strategy"
        )

    def test_parameter_validation(self):
        """Should reject invalid parameter combinations."""
        with pytest.raises(ValueError, match="entry_zscore must be > exit_zscore"):
            PairsTrading(entry_zscore=1.0, exit_zscore=2.0)
        with pytest.raises(ValueError, match="lookback must be >= 30"):
            PairsTrading(lookback=10)
