"""Tests for MultiFactor model and NewsFilter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signals.model.multifactor import MultiFactor, _percentile_rank
from signals.model.news_filter import NewsFilter

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_price_df(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Build a minimal DataFrame with a 'close' column and UTC DatetimeIndex."""
    dates = pd.bdate_range(start=start, periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame({"close": prices}, index=dates)


def _linear_prices(start: float, end: float, n: int) -> list[float]:
    """Generate n linearly spaced prices from start to end."""
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _make_fundamentals(data: list[dict]) -> pd.DataFrame:
    """Build a fundamentals DataFrame from a list of dicts."""
    return pd.DataFrame(data)


# ── MultiFactor tests ─────────────────────────────────────────────────────


class TestPercentileRank:
    """Tests for the _percentile_rank helper."""

    def test_basic_ranking(self) -> None:
        s = pd.Series([10, 20, 30, 40, 50])
        ranked = _percentile_rank(s)
        # Lowest should get ~20%, highest ~100%
        assert ranked.iloc[0] < ranked.iloc[-1]
        assert ranked.iloc[-1] == pytest.approx(100.0)
        assert ranked.iloc[0] == pytest.approx(20.0)

    def test_nan_handling(self) -> None:
        s = pd.Series([10, np.nan, 30])
        ranked = _percentile_rank(s)
        assert pd.isna(ranked.iloc[1])
        assert ranked.iloc[0] < ranked.iloc[2]


class TestFactorRanking:
    """Tests for the score() method factor rankings."""

    def test_momentum_rank_correct(self) -> None:
        """Higher-return stocks should have higher momentum rank."""
        n = 300
        prices = {
            "HIGH_MOM": _make_price_df(_linear_prices(100, 300, n)),  # +200%
            "MED_MOM": _make_price_df(_linear_prices(100, 150, n)),   # +50%
            "LOW_MOM": _make_price_df(_linear_prices(100, 105, n)),   # +5%
        }
        fund = _make_fundamentals([
            {"ticker": "HIGH_MOM", "pe_ratio": 20.0, "roe": 0.15},
            {"ticker": "MED_MOM", "pe_ratio": 20.0, "roe": 0.15},
            {"ticker": "LOW_MOM", "pe_ratio": 20.0, "roe": 0.15},
        ])
        mf = MultiFactor(vol_filter_quantile=1.0)  # Disable vol filter
        as_of = prices["HIGH_MOM"].index[-1]
        scored = mf.score(prices, fund, as_of)

        high_rank = scored.loc[scored["ticker"] == "HIGH_MOM", "momentum_rank"].iloc[0]
        low_rank = scored.loc[scored["ticker"] == "LOW_MOM", "momentum_rank"].iloc[0]
        assert high_rank > low_rank

    def test_value_rank_inversely_proportional_to_pe(self) -> None:
        """Lower P/E should produce higher value rank."""
        n = 300
        prices = {
            "LOW_PE": _make_price_df(_linear_prices(100, 150, n)),
            "HIGH_PE": _make_price_df(_linear_prices(100, 150, n)),
        }
        fund = _make_fundamentals([
            {"ticker": "LOW_PE", "pe_ratio": 10.0, "roe": 0.15},
            {"ticker": "HIGH_PE", "pe_ratio": 50.0, "roe": 0.15},
        ])
        mf = MultiFactor(vol_filter_quantile=1.0)
        as_of = prices["LOW_PE"].index[-1]
        scored = mf.score(prices, fund, as_of)

        low_pe_rank = scored.loc[scored["ticker"] == "LOW_PE", "value_rank"].iloc[0]
        high_pe_rank = scored.loc[scored["ticker"] == "HIGH_PE", "value_rank"].iloc[0]
        assert low_pe_rank > high_pe_rank, (
            f"Low P/E stock should have higher value rank: {low_pe_rank} vs {high_pe_rank}"
        )

    def test_quality_rank_proportional_to_roe(self) -> None:
        """Higher ROE should produce higher quality rank."""
        n = 300
        prices = {
            "HIGH_ROE": _make_price_df(_linear_prices(100, 150, n)),
            "LOW_ROE": _make_price_df(_linear_prices(100, 150, n)),
        }
        fund = _make_fundamentals([
            {"ticker": "HIGH_ROE", "pe_ratio": 20.0, "roe": 0.30},
            {"ticker": "LOW_ROE", "pe_ratio": 20.0, "roe": 0.05},
        ])
        mf = MultiFactor(vol_filter_quantile=1.0)
        as_of = prices["HIGH_ROE"].index[-1]
        scored = mf.score(prices, fund, as_of)

        high_roe_rank = scored.loc[scored["ticker"] == "HIGH_ROE", "quality_rank"].iloc[0]
        low_roe_rank = scored.loc[scored["ticker"] == "LOW_ROE", "quality_rank"].iloc[0]
        assert high_roe_rank > low_roe_rank


class TestCompositeScore:
    """Tests for composite score blending."""

    def test_weighted_blend_produces_expected_scores(self) -> None:
        """Composite should be weighted sum of factor ranks."""
        n = 300
        # Create stocks with different factor profiles
        prices = {
            "A": _make_price_df(_linear_prices(100, 200, n)),  # High momentum
            "B": _make_price_df(_linear_prices(100, 120, n)),  # Low momentum
        }
        fund = _make_fundamentals([
            {"ticker": "A", "pe_ratio": 50.0, "roe": 0.05},  # Bad value, bad quality
            {"ticker": "B", "pe_ratio": 10.0, "roe": 0.30},  # Good value, good quality
        ])
        # Momentum-heavy weights should favor A
        mf_mom = MultiFactor(
            momentum_weight=0.8, value_weight=0.1, quality_weight=0.1,
            vol_filter_quantile=1.0, n_long=1,
        )
        as_of = prices["A"].index[-1]
        weights_mom = mf_mom.rank(prices, fund, as_of)
        assert weights_mom["A"] > 0, "Momentum-heavy should pick the high-mom stock"

        # Value/quality-heavy weights should favor B
        mf_val = MultiFactor(
            momentum_weight=0.1, value_weight=0.45, quality_weight=0.45,
            vol_filter_quantile=1.0, n_long=1,
        )
        weights_val = mf_val.rank(prices, fund, as_of)
        assert weights_val["B"] > 0, "Value/quality-heavy should pick the value stock"


class TestVolFilter:
    """Tests for the volatility filter."""

    def test_high_vol_stocks_excluded(self) -> None:
        """Stocks in the top quartile of vol should be excluded."""
        n = 300
        rng = np.random.default_rng(42)

        # Calm stock: linear price path
        calm_prices = _linear_prices(100, 150, n)

        # Volatile stock: add large noise
        base = _linear_prices(100, 200, n)
        vol_prices = [p + rng.normal(0, 15) for p in base]
        vol_prices = [max(1.0, p) for p in vol_prices]  # Ensure positive

        prices = {
            "CALM": _make_price_df(calm_prices),
            "VOLATILE": _make_price_df(vol_prices),
        }
        fund = _make_fundamentals([
            {"ticker": "CALM", "pe_ratio": 20.0, "roe": 0.15},
            {"ticker": "VOLATILE", "pe_ratio": 20.0, "roe": 0.15},
        ])

        # With vol_filter_quantile=0.5, the more volatile stock should be excluded
        mf = MultiFactor(vol_filter_quantile=0.5, n_long=2)
        as_of = prices["CALM"].index[-1]
        scored = mf.score(prices, fund, as_of)

        calm_included = scored.loc[scored["ticker"] == "CALM", "included"].iloc[0]
        vol_included = scored.loc[scored["ticker"] == "VOLATILE", "included"].iloc[0]

        assert calm_included, "Calm stock should be included"
        assert not vol_included, "Volatile stock should be excluded"

    def test_vol_filter_disabled_at_1(self) -> None:
        """Setting vol_filter_quantile=1.0 should include all stocks."""
        n = 300
        rng = np.random.default_rng(42)
        base = _linear_prices(100, 200, n)
        vol_prices = [p + rng.normal(0, 20) for p in base]
        vol_prices = [max(1.0, p) for p in vol_prices]

        prices = {
            "A": _make_price_df(_linear_prices(100, 150, n)),
            "B": _make_price_df(vol_prices),
        }
        fund = _make_fundamentals([
            {"ticker": "A", "pe_ratio": 20.0, "roe": 0.15},
            {"ticker": "B", "pe_ratio": 20.0, "roe": 0.15},
        ])

        mf = MultiFactor(vol_filter_quantile=1.0)
        as_of = prices["A"].index[-1]
        scored = mf.score(prices, fund, as_of)
        assert scored["included"].all(), "All stocks should be included when filter=1.0"


class TestBacktest:
    """Tests for the backtest() method."""

    def test_produces_valid_equity_curve(self) -> None:
        """Backtest should return a non-empty, monotonic-index equity curve."""
        n = 400
        prices = {
            f"S{i}": _make_price_df(_linear_prices(100, 100 + i * 10, n))
            for i in range(5)
        }
        fund = _make_fundamentals([
            {"ticker": f"S{i}", "pe_ratio": 15.0 + i, "roe": 0.10 + i * 0.02}
            for i in range(5)
        ])
        mf = MultiFactor(
            lookback_days=50, skip_days=5, n_long=2, rebalance_freq=21,
            vol_filter_quantile=1.0,
        )
        start = str(prices["S0"].index[0].date())
        end = str(prices["S0"].index[-1].date())
        equity = mf.backtest(prices, fund, start=start, end=end, initial_cash=10000.0)

        assert len(equity) > 0, "Equity curve should not be empty"
        assert equity.iloc[0] == pytest.approx(10000.0, abs=5.0)
        assert equity.index.is_monotonic_increasing

    def test_empty_universe_returns_empty(self) -> None:
        """Backtest on empty universe should return empty Series."""
        mf = MultiFactor()
        fund = pd.DataFrame(columns=["ticker", "pe_ratio", "roe"])
        equity = mf.backtest({}, fund, start="2020-01-01", end="2020-12-31")
        assert len(equity) == 0


class TestFundamentalsCaching:
    """Tests for the fundamentals caching logic."""

    def test_saves_and_reloads_parquet(self) -> None:
        """Fundamentals should round-trip through parquet correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_fund.parquet"
            fund = pd.DataFrame({
                "ticker": ["AAPL", "MSFT"],
                "pe_ratio": [25.0, 30.0],
                "roe": [0.20, 0.35],
                "fetched_at": [pd.Timestamp.now().isoformat()] * 2,
            })
            fund.to_parquet(cache_path, index=False)

            loaded = pd.read_parquet(cache_path)
            assert len(loaded) == 2
            assert list(loaded.columns) == ["ticker", "pe_ratio", "roe", "fetched_at"]
            assert loaded.loc[0, "ticker"] == "AAPL"
            assert loaded.loc[1, "pe_ratio"] == pytest.approx(30.0)


class TestWeightsInterface:
    """Verify MultiFactor.rank() matches CrossSectionalMomentum.rank() interface."""

    def test_rank_returns_dict_with_all_tickers(self) -> None:
        """rank() should return a dict with every ticker in prices_dict."""
        n = 300
        prices = {
            f"S{i}": _make_price_df(_linear_prices(100, 100 + i * 10, n))
            for i in range(6)
        }
        fund = _make_fundamentals([
            {"ticker": f"S{i}", "pe_ratio": 15.0, "roe": 0.15}
            for i in range(6)
        ])
        mf = MultiFactor(n_long=2, vol_filter_quantile=1.0)
        as_of = prices["S0"].index[-1]
        weights = mf.rank(prices, fund, as_of)

        assert set(weights.keys()) == set(prices.keys())
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_weights_sum_to_one(self) -> None:
        """Positive weights should sum to 1.0."""
        n = 300
        prices = {
            f"S{i}": _make_price_df(_linear_prices(100, 100 + i * 10, n))
            for i in range(10)
        }
        fund = _make_fundamentals([
            {"ticker": f"S{i}", "pe_ratio": 15.0 + i, "roe": 0.10 + i * 0.01}
            for i in range(10)
        ])
        mf = MultiFactor(n_long=3, vol_filter_quantile=1.0)
        as_of = prices["S0"].index[-1]
        weights = mf.rank(prices, fund, as_of)

        positive = {t: w for t, w in weights.items() if w > 0}
        assert len(positive) == 3
        assert abs(sum(positive.values()) - 1.0) < 1e-9


# ── NewsFilter tests ──────────────────────────────────────────────────────


class TestNewsFilterKeywords:
    """Tests for keyword detection in the NewsFilter."""

    def test_geopolitical_detection(self) -> None:
        """Headlines with geopolitical keywords should be detected."""
        nf = NewsFilter()
        headlines = [
            "US imposes new sanctions on Russian oil exports",
            "Tariff fears weigh on semiconductor stocks",
        ]
        score, categories = nf._score_headlines(headlines)
        assert score >= 2
        assert "geopolitical" in categories

    def test_regulatory_detection(self) -> None:
        """Headlines with regulatory keywords should be detected."""
        nf = NewsFilter()
        headlines = [
            "SEC launches investigation into company accounting",
            "FDA rejects new drug application",
        ]
        score, categories = nf._score_headlines(headlines)
        assert score >= 2
        assert "regulatory" in categories

    def test_clean_headlines_score_zero(self) -> None:
        """Normal business headlines should produce zero risk."""
        nf = NewsFilter()
        headlines = [
            "Apple reports record quarterly revenue",
            "Tech stocks rally on strong earnings",
            "Market closes higher on jobs data",
        ]
        score, categories = nf._score_headlines(headlines)
        assert score == 0
        assert len(categories) == 0


class TestNewsFilterRecommendation:
    """Tests for the SKIP / CAUTION / PROCEED classification."""

    def test_proceed_for_low_risk(self) -> None:
        """Low risk score should produce PROCEED."""
        nf = NewsFilter(max_risk_score=3)
        result = nf.check_ticker_from_headlines("AAPL", [
            "Apple launches new iPhone model",
        ])
        assert result["recommendation"] == "PROCEED"
        assert not result["flagged"]

    def test_caution_for_moderate_risk(self) -> None:
        """Moderate risk score should produce CAUTION."""
        nf = NewsFilter(max_risk_score=3)
        result = nf.check_ticker_from_headlines("XOM", [
            "Oil sanctions threaten supply chain",
            "Tariff war escalates between major economies",
            "Embargo on key commodity exports",
        ])
        assert result["risk_score"] >= 3
        assert result["recommendation"] in ("CAUTION", "SKIP")
        assert result["flagged"]

    def test_skip_for_high_risk(self) -> None:
        """High risk score should produce SKIP."""
        nf = NewsFilter(max_risk_score=3)
        result = nf.check_ticker_from_headlines("DANGER", [
            "SEC investigation launched into company fraud",
            "Sanctions imposed on key supplier",
            "Merger deal collapses amid antitrust probe",
            "Company faces crisis as CEO indicted in lawsuit",
            "Tariff ban threatens business collapse",
        ])
        assert result["risk_score"] >= 5
        assert result["recommendation"] == "SKIP"


class TestNewsFilterIntegration:
    """Tests for filter_signals() integration."""

    def test_flagged_stocks_removed(self) -> None:
        """SKIP stocks should be removed from the signals dict."""
        nf = NewsFilter(max_risk_score=2)
        signals = {"SAFE": 0.5, "RISKY": 0.5}
        headlines = {
            "SAFE": ["Company reports strong earnings growth"],
            "RISKY": [
                "SEC investigation launched",
                "Sanctions hit company operations",
                "Merger collapses in crisis",
                "Company faces bankruptcy threat after lawsuit",
            ],
        }
        filtered = nf.filter_signals(signals, use_headlines=headlines)

        # RISKY should be removed or reduced
        assert "SAFE" in filtered
        # SAFE should get all the weight after redistribution
        assert filtered["SAFE"] == pytest.approx(1.0, abs=0.01) or (
            "RISKY" in filtered and filtered["RISKY"] < 0.5
        )

    def test_empty_signals_returns_empty(self) -> None:
        """Empty input should produce empty output."""
        nf = NewsFilter()
        result = nf.filter_signals({})
        assert result == {}

    def test_all_clean_unchanged(self) -> None:
        """When no stocks are flagged, weights should be unchanged."""
        nf = NewsFilter(max_risk_score=3)
        signals = {"A": 0.5, "B": 0.5}
        headlines = {
            "A": ["Stock rallies on good news"],
            "B": ["Company beats earnings estimates"],
        }
        filtered = nf.filter_signals(signals, use_headlines=headlines)
        assert filtered["A"] == pytest.approx(0.5)
        assert filtered["B"] == pytest.approx(0.5)

    def test_weight_redistribution(self) -> None:
        """After removing a stock, remaining weights should sum to 1.0."""
        nf = NewsFilter(max_risk_score=1)
        signals = {"A": 0.33, "B": 0.33, "C": 0.34}
        headlines = {
            "A": ["Normal earnings report"],
            "B": ["Normal growth story"],
            "C": [
                "SEC launches investigation into fraud",
                "Company faces sanctions and embargo",
                "Crisis deepens as merger collapses in lawsuit",
            ],
        }
        filtered = nf.filter_signals(signals, use_headlines=headlines)
        total = sum(filtered.values())
        assert total == pytest.approx(1.0, abs=0.01)
