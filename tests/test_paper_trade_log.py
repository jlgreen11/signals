"""Tests for PaperTradeLog and PaperTradeEntry."""

from __future__ import annotations

import pytest

from signals.broker.paper_trade_log import PaperTradeEntry, PaperTradeLog


def test_empty_log_summary():
    log = PaperTradeLog(symbol="BTC-USD")
    summary = log.summary()
    assert summary["total_entries"] == 0
    assert summary["reconciled"] == 0
    assert summary["unreconciled"] == 0


def test_append_and_reconcile():
    log = PaperTradeLog(symbol="BTC-USD")
    entry = PaperTradeEntry(
        date="2026-04-11",
        signal="BUY",
        target_position=1.0,
        expected_fill_price=65000.0,
        signal_model="hybrid",
        signal_params={"vol_quantile": 0.70},
    )
    log.append(entry)
    assert len(log.entries) == 1
    assert len(log.unreconciled_entries()) == 1
    assert len(log.reconciled_entries()) == 0

    # Reconcile with an actual open slightly different from expected
    reconciled = log.reconcile(
        date="2026-04-11",
        actual_open=65100.0,
        actual_close=66000.0,
        commission_bps=5.0,
        slippage_bps=5.0,
    )
    assert reconciled is not None
    assert reconciled.reconciled
    assert reconciled.actual_open == 65100.0
    assert reconciled.actual_close == 66000.0
    assert reconciled.realized_return is not None
    assert reconciled.backtest_return is not None

    # Realized_return should be positive (bought at 65100*(1+0.0005)=65132.5,
    # sold/closed at 66000, long 1.0 position)
    assert reconciled.realized_return > 0


def test_reconcile_unknown_date_returns_none():
    log = PaperTradeLog(symbol="BTC-USD")
    log.append(
        PaperTradeEntry(
            date="2026-04-11",
            signal="BUY",
            target_position=1.0,
            expected_fill_price=65000.0,
            signal_model="hybrid",
            signal_params={},
        )
    )
    result = log.reconcile(date="2026-04-12", actual_open=1, actual_close=1)
    assert result is None


def test_reconcile_idempotent():
    """Reconciling the same date twice should return the already-reconciled entry."""
    log = PaperTradeLog(symbol="BTC-USD")
    log.append(
        PaperTradeEntry(
            date="2026-04-11",
            signal="BUY",
            target_position=1.0,
            expected_fill_price=100.0,
            signal_model="hybrid",
            signal_params={},
        )
    )
    r1 = log.reconcile(date="2026-04-11", actual_open=100, actual_close=110)
    assert r1 is not None
    first_realized = r1.realized_return
    r2 = log.reconcile(date="2026-04-11", actual_open=999, actual_close=999)
    assert r2 is not None
    assert r2.realized_return == first_realized  # unchanged


def test_summary_with_reconciled_entries():
    log = PaperTradeLog(symbol="BTC-USD")
    for date, price_open, price_close, target in [
        ("2026-04-11", 100.0, 105.0, 1.0),    # +5% while long
        ("2026-04-12", 105.0, 102.0, 0.0),    # flat, no return
        ("2026-04-13", 102.0, 104.0, 0.5),    # +2% at half size
    ]:
        log.append(
            PaperTradeEntry(
                date=date,
                signal="BUY" if target > 0 else "HOLD",
                target_position=target,
                expected_fill_price=price_open,
                signal_model="hybrid",
                signal_params={},
            )
        )
        log.reconcile(date=date, actual_open=price_open, actual_close=price_close)

    summary = log.summary()
    assert summary["total_entries"] == 3
    assert summary["reconciled"] == 3
    assert summary["unreconciled"] == 0
    assert summary["cumulative_realized_return"] > 0
    assert summary["cumulative_backtest_return"] > 0


def test_save_and_load_roundtrip(tmp_path):
    log = PaperTradeLog(symbol="BTC-USD")
    log.append(
        PaperTradeEntry(
            date="2026-04-11",
            signal="BUY",
            target_position=0.85,
            expected_fill_price=65000.0,
            signal_model="hybrid",
            signal_params={"vol_quantile": 0.70, "max_long": 1.5},
        )
    )
    path = log.save(log_dir=tmp_path)
    assert path.exists()

    loaded = PaperTradeLog.load(symbol="BTC-USD", log_dir=tmp_path)
    assert loaded.symbol == "BTC-USD"
    assert len(loaded.entries) == 1
    e = loaded.entries[0]
    assert e.date == "2026-04-11"
    assert e.signal == "BUY"
    assert e.target_position == pytest.approx(0.85)
    assert e.expected_fill_price == pytest.approx(65000.0)
    assert e.signal_params["vol_quantile"] == 0.70


def test_load_missing_file_returns_empty_log(tmp_path):
    log = PaperTradeLog.load(symbol="NONEXISTENT", log_dir=tmp_path)
    assert log.symbol == "NONEXISTENT"
    assert len(log.entries) == 0


def test_safe_symbol_strips_problematic_chars():
    log = PaperTradeLog(symbol="^GSPC")
    path = log.path(log_dir=None)
    # The filename should not contain '^' which is problematic in paths
    assert "^" not in path.name
    assert path.name == "GSPC.json"
