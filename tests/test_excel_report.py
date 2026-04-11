"""Smoke tests for the Excel daily-activity report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.excel_report import (
    build_daily_activity_frame,
    build_summary_frame,
    build_trade_frame,
    write_excel_report,
)


def _make_prices(n: int = 300, seed: int = 42) -> pd.DataFrame:
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


def _run_simple() -> tuple:
    prices = _make_prices()
    cfg = BacktestConfig(
        model_type="composite",
        train_window=100,
        retrain_freq=21,
        return_bins=3,
        volatility_bins=3,
        vol_window=10,
        laplace_alpha=0.01,
    )
    result = BacktestEngine(cfg).run(prices, symbol="TEST")
    return result, prices


def test_summary_frame_has_expected_fields() -> None:
    result, _ = _run_simple()
    df = build_summary_frame(result, symbol="TEST")
    assert {"metric", "value"} == set(df.columns)
    metrics = set(df["metric"])
    for required in ("Symbol", "Sharpe", "CAGR", "Max drawdown", "Equity at window end"):
        assert required in metrics


def test_activity_frame_has_one_row_per_bar() -> None:
    result, prices = _run_simple()
    df = build_daily_activity_frame(result, prices)
    assert "date" in df.columns
    assert "close" in df.columns
    assert "equity" in df.columns
    assert "cumulative_buys" in df.columns
    assert "cumulative_sells" in df.columns
    assert len(df) == len(result.equity_curve)


def test_activity_frame_cumulative_buys_is_monotonic() -> None:
    result, prices = _run_simple()
    df = build_daily_activity_frame(result, prices)
    cb = df["cumulative_buys"].values
    assert all(cb[i] <= cb[i + 1] for i in range(len(cb) - 1))


def test_trade_frame_matches_result_trades() -> None:
    result, _ = _run_simple()
    df = build_trade_frame(result)
    assert len(df) == len(result.trades)
    if len(df):
        assert set(df["side"]).issubset({"BUY", "SELL", "SHORT", "COVER", "STOP"})


def test_write_excel_report_end_to_end(tmp_path: Path) -> None:
    result, prices = _run_simple()
    out = tmp_path / "test_report.xlsx"
    paths = write_excel_report(result, prices, out, symbol="TEST")
    assert out.exists()
    assert paths.summary_rows > 0
    assert paths.activity_rows > 0
    # Spot-check: reopen and confirm sheet names
    import openpyxl
    wb = openpyxl.load_workbook(out)
    assert set(wb.sheetnames) >= {"Summary", "Daily Activity", "Trades"}


def test_activity_frame_drawdown_is_non_positive() -> None:
    result, prices = _run_simple()
    df = build_daily_activity_frame(result, prices)
    assert (df["drawdown_pct"] <= 1e-9).all()
