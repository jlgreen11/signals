"""Tests for AlpacaBroker — covers the dry-run path only.

Live-mode tests would require real credentials and alpaca-py installed.
Those are out of scope for the unit test suite. The critical thing to
verify is that WITHOUT `live=True`, the broker never touches the API —
it behaves as a pure logging stub.
"""

from __future__ import annotations

import os

import pytest

from signals.broker.alpaca import AlpacaBroker, AlpacaCredentials
from signals.broker.base import Order, OrderSide, OrderType


def test_dry_run_get_cash_is_synthetic():
    broker = AlpacaBroker(live=False)
    cash = broker.get_cash()
    assert cash == 100_000.0


def test_dry_run_get_positions_is_empty():
    broker = AlpacaBroker(live=False)
    positions = broker.get_positions()
    assert positions == []


def test_dry_run_submit_order_returns_synthetic_id():
    broker = AlpacaBroker(live=False)
    order = Order(symbol="BTCUSD", side=OrderSide.BUY, qty=0.01)
    result = broker.submit_order(order)
    assert result.id is not None
    assert result.id.startswith("dryrun-")
    assert result.symbol == "BTCUSD"
    assert result.side == OrderSide.BUY
    assert result.qty == 0.01


def test_dry_run_cancel_order_returns_true():
    broker = AlpacaBroker(live=False)
    assert broker.cancel_order("dryrun-12345678") is True


def test_dry_run_get_quote_returns_synthetic_quote():
    broker = AlpacaBroker(live=False)
    quote = broker.get_quote("BTCUSD")
    assert quote.symbol == "BTCUSD"
    # Synthetic quotes have zero bid/ask to signal "no data"
    assert quote.bid == 0.0
    assert quote.ask == 0.0


def test_live_mode_without_credentials_raises():
    """The constructor must reject live=True when no credentials exist."""
    # Save and clear any existing env vars
    saved_api = os.environ.pop("ALPACA_API_KEY", None)
    saved_secret = os.environ.pop("ALPACA_SECRET_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
            AlpacaBroker(live=True)
    finally:
        if saved_api:
            os.environ["ALPACA_API_KEY"] = saved_api
        if saved_secret:
            os.environ["ALPACA_SECRET_KEY"] = saved_secret


def test_live_mode_with_explicit_credentials_fails_if_sdk_missing():
    """Even with credentials, if the alpaca-py SDK is not installed,
    the constructor should fail at client initialization time — NOT
    silently fall through to dry-run mode."""
    creds = AlpacaCredentials(
        api_key="fake-key",
        secret_key="fake-secret",
        base_url="https://paper-api.alpaca.markets",
    )
    # The alpaca-py package is not a project dependency; the import
    # should fail inside _ensure_client().
    try:
        AlpacaBroker(live=True, credentials=creds)
        # If we got here, the SDK is installed — skip this negative test
        pytest.skip("alpaca-py is installed, skipping SDK-missing test")
    except RuntimeError as e:
        assert "alpaca-py" in str(e) or "not installed" in str(e)


def test_submit_order_in_dry_run_preserves_order_type():
    broker = AlpacaBroker(live=False)
    limit = Order(
        symbol="SPY",
        side=OrderSide.SELL,
        qty=10.0,
        order_type=OrderType.LIMIT,
        limit_price=450.0,
    )
    result = broker.submit_order(limit)
    assert result.order_type == OrderType.LIMIT
    assert result.limit_price == 450.0


def test_credentials_from_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    creds = AlpacaCredentials.from_env()
    assert creds is not None
    assert creds.api_key == "test-key"
    assert creds.secret_key == "test-secret"


def test_credentials_from_env_returns_none_when_missing(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    creds = AlpacaCredentials.from_env()
    assert creds is None
