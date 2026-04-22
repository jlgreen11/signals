"""Tests for the automation layer: SignalStore, CashOverlay, InsightsEngine, PaperTradeRunner."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from signals.automation.cash_overlay import CashOverlay
from signals.automation.insights_engine import InsightsEngine
from signals.automation.paper_runner import PaperTradeRunner
from signals.automation.signal_store import SignalStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db():
    """Create a temporary SQLite database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture()
def signal_store(tmp_db):
    return SignalStore(db_path=tmp_db)


@pytest.fixture()
def sample_prices():
    """Generate synthetic prices for testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=300, tz="UTC")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
               "META", "TSLA", "AVGO", "JPM", "UNH",
               "SPY", "EFA", "TLT", "GLD", "BTC-USD"]
    prices_dict: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        log_returns = rng.normal(0.0003, 0.02, size=300)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        df = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.005, 300)),
                "high": close * (1 + abs(rng.normal(0, 0.01, 300))),
                "low": close * (1 - abs(rng.normal(0, 0.01, 300))),
                "close": close,
                "volume": rng.integers(1_000_000, 50_000_000, size=300),
            },
            index=dates,
        )
        prices_dict[ticker] = df
    return prices_dict


class FakeDataStore:
    """Minimal stand-in for DataStore in tests."""

    def __init__(self, prices_dict: dict[str, pd.DataFrame]) -> None:
        self._prices = prices_dict
        self.db_path = ":memory:"

    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        return self._prices.get(symbol, pd.DataFrame())

    def last_timestamp(self, symbol: str, interval: str) -> pd.Timestamp | None:
        df = self._prices.get(symbol, pd.DataFrame())
        if df.empty:
            return None
        return df.index.max()


# ---------------------------------------------------------------------------
# SignalStore tests
# ---------------------------------------------------------------------------

class TestSignalStore:
    def test_record_and_retrieve_signal(self, signal_store):
        signal_store.record_signal(
            model="momentum", ticker="AAPL", signal="BUY",
            weight=0.2, confidence=0.8, metadata={"lookback": 252},
        )
        df = signal_store.get_latest_signals(model="momentum", n=10)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["signal"] == "BUY"
        assert df.iloc[0]["weight"] == pytest.approx(0.2)

    def test_record_multiple_signals(self, signal_store):
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            signal_store.record_signal(
                model="momentum", ticker=ticker, signal="BUY",
                weight=0.2, confidence=0.7,
            )
        df = signal_store.get_latest_signals(model="momentum", n=50)
        assert len(df) == 3

    def test_filter_by_model(self, signal_store):
        signal_store.record_signal(model="momentum", ticker="AAPL", signal="BUY", weight=0.2)
        signal_store.record_signal(model="tsmom", ticker="SPY", signal="BUY", weight=0.15)
        df_mom = signal_store.get_latest_signals(model="momentum")
        df_ts = signal_store.get_latest_signals(model="tsmom")
        assert len(df_mom) == 1
        assert len(df_ts) == 1

    def test_get_all_signals(self, signal_store):
        signal_store.record_signal(model="momentum", ticker="AAPL", signal="BUY", weight=0.2)
        signal_store.record_signal(model="tsmom", ticker="SPY", signal="BUY", weight=0.15)
        df = signal_store.get_latest_signals(n=50)
        assert len(df) == 2

    def test_record_and_retrieve_targets(self, signal_store):
        targets = {"AAPL": 0.2, "MSFT": 0.2, "GOOGL": 0.2}
        signal_store.record_portfolio_target("momentum", targets, cash_pct=0.4)
        result = signal_store.get_latest_targets(model="momentum")
        assert "momentum" in result
        assert result["momentum"]["targets"]["AAPL"] == pytest.approx(0.2)
        assert result["momentum"]["cash_pct"] == pytest.approx(0.4)

    def test_get_latest_targets_all_models(self, signal_store):
        signal_store.record_portfolio_target("momentum", {"AAPL": 0.2}, 0.8)
        signal_store.record_portfolio_target("tsmom", {"SPY": 0.3}, 0.7)
        result = signal_store.get_latest_targets()
        assert "momentum" in result
        assert "tsmom" in result

    def test_signal_history(self, signal_store):
        signal_store.record_signal(model="momentum", ticker="AAPL", signal="BUY", weight=0.2)
        signal_store.record_signal(model="momentum", ticker="AAPL", signal="SELL", weight=0.0)
        df = signal_store.get_signal_history("AAPL", model="momentum", days=90)
        assert len(df) == 2

    def test_signal_history_no_model_filter(self, signal_store):
        signal_store.record_signal(model="momentum", ticker="AAPL", signal="BUY", weight=0.2)
        signal_store.record_signal(model="tsmom", ticker="AAPL", signal="BUY", weight=0.1)
        df = signal_store.get_signal_history("AAPL", days=90)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# CashOverlay tests
# ---------------------------------------------------------------------------

class TestCashOverlay:
    def test_blend_basic(self):
        overlay = CashOverlay(total_capital=100_000)
        model_targets = {
            "momentum": {"AAPL": 0.20, "MSFT": 0.20, "GOOGL": 0.20, "AMZN": 0.20, "NVDA": 0.20},
            "tsmom": {"SPY": 0.25, "EFA": 0.25, "TLT": 0.25, "GLD": 0.25},
            "pead": {},
        }
        blended = overlay.blend(model_targets)
        assert "_CASH" in blended
        total = sum(blended.values())
        assert total == pytest.approx(100_000, rel=1e-6)

    def test_position_limit_enforced(self):
        overlay = CashOverlay(total_capital=100_000, max_position_pct=0.10)
        model_targets = {
            "momentum": {"AAPL": 1.0},  # Would be $50k without cap
        }
        blended = overlay.blend(model_targets)
        aapl = blended.get("AAPL", 0.0)
        assert aapl <= 100_000 * 0.10 + 1.0  # Allow $1 rounding

    def test_cash_reserve_maintained(self):
        overlay = CashOverlay(total_capital=100_000, cash_reserve_pct=0.10)
        model_targets = {
            "momentum": {"AAPL": 0.20, "MSFT": 0.20, "GOOGL": 0.20, "AMZN": 0.20, "NVDA": 0.20},
            "tsmom": {"SPY": 0.25, "EFA": 0.25, "TLT": 0.25, "GLD": 0.25},
            "pead": {"META": 0.50, "TSLA": 0.50},
        }
        blended = overlay.blend(model_targets)
        cash = blended.get("_CASH", 0.0)
        assert cash >= 100_000 * 0.10 - 1.0  # Allow $1 rounding

    def test_empty_targets(self):
        overlay = CashOverlay(total_capital=100_000)
        blended = overlay.blend({})
        assert blended["_CASH"] == pytest.approx(100_000)

    def test_rebalance_orders_buy(self):
        overlay = CashOverlay(total_capital=100_000)
        current = {}
        target = {"AAPL": 10_000}
        prices = {"AAPL": 150.0}
        orders = overlay.rebalance_orders(current, target, prices)
        assert len(orders) == 1
        assert orders[0]["action"] == "BUY"
        assert orders[0]["notional"] == pytest.approx(10_000)

    def test_rebalance_orders_sell(self):
        overlay = CashOverlay(total_capital=100_000)
        current = {"AAPL": 10_000}
        target = {"AAPL": 5_000}
        prices = {"AAPL": 150.0}
        orders = overlay.rebalance_orders(current, target, prices)
        assert len(orders) == 1
        assert orders[0]["action"] == "SELL"
        assert orders[0]["notional"] == pytest.approx(5_000)

    def test_summary_not_empty(self):
        overlay = CashOverlay(total_capital=100_000)
        text = overlay.summary()
        assert "Total capital" in text
        assert "momentum" in text


# ---------------------------------------------------------------------------
# InsightsEngine tests
# ---------------------------------------------------------------------------

class TestInsightsEngine:
    def test_run_daily_produces_all_model_signals(self, signal_store, sample_prices):
        data_store = FakeDataStore(sample_prices)
        overlay = CashOverlay(total_capital=100_000)
        engine = InsightsEngine(
            signal_store=signal_store,
            cash_overlay=overlay,
            data_store=data_store,
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                      "META", "TSLA", "AVGO", "JPM", "UNH"],
            tsmom_tickers=["SPY", "EFA", "TLT", "GLD", "BTC-USD"],
        )
        report = engine.run_daily()
        assert report["n_momentum_signals"] > 0
        # TSMOM may or may not have signals depending on synthetic data
        assert "blended_allocation" in report
        assert "_CASH" in report["blended_allocation"]

    def test_report_text_not_empty(self, signal_store, sample_prices):
        data_store = FakeDataStore(sample_prices)
        overlay = CashOverlay(total_capital=100_000)
        engine = InsightsEngine(
            signal_store=signal_store,
            cash_overlay=overlay,
            data_store=data_store,
            tickers=["AAPL", "MSFT", "GOOGL"],
            tsmom_tickers=["SPY", "EFA"],
        )
        report = engine.run_daily()
        assert len(report["report_text"]) > 50
        assert "DAILY INSIGHTS REPORT" in report["report_text"]

    def test_run_weekly(self, signal_store, sample_prices):
        data_store = FakeDataStore(sample_prices)
        overlay = CashOverlay(total_capital=100_000)
        engine = InsightsEngine(
            signal_store=signal_store,
            cash_overlay=overlay,
            data_store=data_store,
            tickers=["AAPL", "MSFT"],
            tsmom_tickers=["SPY"],
        )
        report = engine.run_weekly()
        assert "weekly_signal_count" in report
        assert report.get("report_type") == "weekly"


# ---------------------------------------------------------------------------
# PaperTradeRunner tests
# ---------------------------------------------------------------------------

class TestPaperTradeRunner:
    def test_execute_daily(self, tmp_path, signal_store, sample_prices):
        data_store = FakeDataStore(sample_prices)
        overlay = CashOverlay(total_capital=100_000)
        engine = InsightsEngine(
            signal_store=signal_store,
            cash_overlay=overlay,
            data_store=data_store,
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                      "META", "TSLA", "AVGO", "JPM", "UNH"],
            tsmom_tickers=["SPY", "EFA", "TLT", "GLD", "BTC-USD"],
        )
        runner = PaperTradeRunner(engine=engine, initial_capital=100_000, db_path=tmp_path / "paper_test.db")
        result = runner.execute_daily()
        assert "equity" in result
        assert result["equity"] > 0

    def test_positions_update_after_execution(self, tmp_path, signal_store, sample_prices):
        data_store = FakeDataStore(sample_prices)
        overlay = CashOverlay(total_capital=100_000)
        engine = InsightsEngine(
            signal_store=signal_store,
            cash_overlay=overlay,
            data_store=data_store,
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                      "META", "TSLA", "AVGO", "JPM", "UNH"],
            tsmom_tickers=["SPY", "EFA", "TLT", "GLD", "BTC-USD"],
        )
        runner = PaperTradeRunner(engine=engine, initial_capital=100_000, db_path=tmp_path / "paper_test.db")
        runner.execute_daily()
        positions = runner.get_positions()
        # Should have some positions after execution
        assert len(positions) > 0

    def test_performance_computed(self, tmp_path, signal_store, sample_prices):
        data_store = FakeDataStore(sample_prices)
        overlay = CashOverlay(total_capital=100_000)
        engine = InsightsEngine(
            signal_store=signal_store,
            cash_overlay=overlay,
            data_store=data_store,
            tickers=["AAPL", "MSFT"],
            tsmom_tickers=["SPY"],
        )
        runner = PaperTradeRunner(engine=engine, initial_capital=100_000, db_path=tmp_path / "paper_test.db")
        runner.execute_daily()
        perf = runner.get_performance()
        assert "total_return" in perf
        assert perf["n_days"] == 1

    def test_trade_log_populated(self, tmp_path, signal_store, sample_prices):
        data_store = FakeDataStore(sample_prices)
        overlay = CashOverlay(total_capital=100_000)
        engine = InsightsEngine(
            signal_store=signal_store,
            cash_overlay=overlay,
            data_store=data_store,
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                      "META", "TSLA", "AVGO", "JPM", "UNH"],
            tsmom_tickers=["SPY", "EFA", "TLT", "GLD", "BTC-USD"],
        )
        runner = PaperTradeRunner(engine=engine, initial_capital=100_000, db_path=tmp_path / "paper_test.db")
        runner.execute_daily()
        log = runner.get_trade_log()
        # Should have trades
        assert len(log) > 0

    def test_empty_performance_before_trading(self, tmp_path, signal_store, sample_prices):
        data_store = FakeDataStore(sample_prices)
        overlay = CashOverlay(total_capital=100_000)
        engine = InsightsEngine(
            signal_store=signal_store,
            cash_overlay=overlay,
            data_store=data_store,
            tickers=["AAPL"],
            tsmom_tickers=["SPY"],
        )
        runner = PaperTradeRunner(engine=engine, initial_capital=100_000, db_path=tmp_path / "paper_test.db")
        perf = runner.get_performance()
        assert perf["total_return"] == 0.0
        assert perf["n_days"] == 0
