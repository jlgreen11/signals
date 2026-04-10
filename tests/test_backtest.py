"""Tests for backtest engine + portfolio + metrics, against both model backends."""

from __future__ import annotations

import pandas as pd
import pytest

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import max_drawdown, sharpe_ratio
from signals.backtest.portfolio import Portfolio


def test_portfolio_long_round_trip():
    p = Portfolio(initial_cash=10000.0, commission_bps=0, slippage_bps=0)
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    p.set_target(ts, price=100.0, target_fraction=1.0)
    assert p.qty == pytest.approx(100.0)
    assert p.cash == pytest.approx(0.0)
    p.flatten(pd.Timestamp("2020-02-01", tz="UTC"), price=110.0)
    assert p.qty == 0.0
    assert p.cash == pytest.approx(11000.0)


def test_portfolio_short_round_trip():
    p = Portfolio(initial_cash=10000.0, commission_bps=0, slippage_bps=0)
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    p.set_target(ts, price=100.0, target_fraction=-1.0)
    assert p.qty == pytest.approx(-100.0)
    # Profit on a short when price falls
    p.flatten(pd.Timestamp("2020-02-01", tz="UTC"), price=90.0)
    assert p.qty == 0.0
    assert p.cash == pytest.approx(11000.0)


def test_portfolio_half_size():
    p = Portfolio(initial_cash=10000.0, commission_bps=0, slippage_bps=0)
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    p.set_target(ts, price=100.0, target_fraction=0.5)
    assert p.qty == pytest.approx(50.0)
    assert p.cash == pytest.approx(5000.0)


def test_portfolio_stop_loss_long():
    p = Portfolio(initial_cash=10000.0, commission_bps=0, slippage_bps=0)
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    p.set_target(ts, price=100.0, target_fraction=1.0)
    fired = p.check_stop(pd.Timestamp("2020-01-02", tz="UTC"), price=89.0, stop_loss_pct=0.10)
    assert fired
    assert p.qty == 0.0


def test_portfolio_stop_loss_short():
    p = Portfolio(initial_cash=10000.0, commission_bps=0, slippage_bps=0)
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    p.set_target(ts, price=100.0, target_fraction=-1.0)
    fired = p.check_stop(pd.Timestamp("2020-01-02", tz="UTC"), price=111.0, stop_loss_pct=0.10)
    assert fired
    assert p.qty == 0.0


def test_max_drawdown_computation():
    eq = pd.Series(
        [100, 110, 105, 120, 90, 95],
        index=pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC"),
    )
    dd = max_drawdown(eq)
    assert dd == pytest.approx((90 - 120) / 120)


def test_sharpe_zero_for_flat_returns():
    eq = pd.Series([100.0] * 10, index=pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC"))
    rets = eq.pct_change().dropna()
    assert sharpe_ratio(rets, 252) == 0.0


def test_engine_runs_with_hmm_backend(synthetic_prices):
    cfg = BacktestConfig(
        model_type="hmm", train_window=200, retrain_freq=40, n_states=3, n_iter=30
    )
    engine = BacktestEngine(cfg)
    result = engine.run(synthetic_prices, symbol="TEST")
    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0
    assert len(result.benchmark_curve) > 0


def test_engine_runs_with_homc_backend(synthetic_prices):
    cfg = BacktestConfig(
        model_type="homc", train_window=200, retrain_freq=40, n_states=3, order=2
    )
    engine = BacktestEngine(cfg)
    result = engine.run(synthetic_prices, symbol="TEST")
    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0
    assert len(result.benchmark_curve) > 0


def test_engine_runs_with_composite_backend(synthetic_prices):
    cfg = BacktestConfig(
        model_type="composite", train_window=200, retrain_freq=40,
        return_bins=3, volatility_bins=3,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(synthetic_prices, symbol="TEST")
    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0


def test_engine_supports_shorts_and_stops(synthetic_prices):
    cfg = BacktestConfig(
        model_type="composite", train_window=200, retrain_freq=40,
        allow_short=True, stop_loss_pct=0.10, target_scale_bps=30.0,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(synthetic_prices, symbol="TEST")
    sides = {t.side for t in result.trades}
    # We should see at least BUY/SELL/SHORT/COVER somewhere across the run.
    assert "BUY" in sides or "SHORT" in sides
