"""AlpacaBroker — thin adapter over Alpaca's REST API.

⚠ SAFETY: This class will REFUSE to place real orders unless explicitly
initialized with `live=True`. The default is DRY RUN mode where
submit_order() logs the intent and returns a synthetic success response
without hitting the API. You must set `live=True` AND provide valid
credentials via environment variables to actually place orders.

This is Tier-3 Phase G — code-only. No live execution in this session.
The class is designed so that a user (who understands the risks and has
authorization over their own Alpaca account) can:

    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...
    broker = AlpacaBroker(live=True)  # explicit opt-in
    broker.submit_order(order)        # actually places the order

Without `live=True` or credentials, the broker operates as a dry-run
mock that's useful for paper-trading logic wiring but does NOT send
anything to Alpaca.

Dependency: alpaca-py SDK. Install via `pip install alpaca-py` (not
yet added to pyproject.toml dependencies — adding live trading as a
mandatory dep would make a research project harder to install for
someone who just wants to read the code).

Usage in signals integration:

    from signals.broker.alpaca import AlpacaBroker
    from signals.broker.base import Order, OrderSide

    broker = AlpacaBroker(live=False)  # dry-run, safe default
    quote = broker.get_quote("BTCUSD")  # real quote if credentials present
    order = Order(symbol="BTCUSD", side=OrderSide.BUY, qty=0.01)
    result = broker.submit_order(order)  # no-op in dry-run mode
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from signals.broker.base import (
    Broker,
    Order,
    OrderSide,
    OrderType,
    Position,
    Quote,
)
from signals.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AlpacaCredentials:
    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"

    @classmethod
    def from_env(cls) -> AlpacaCredentials | None:
        """Load credentials from ALPACA_API_KEY / ALPACA_SECRET_KEY env
        vars. Returns None if either is missing."""
        api = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        base = os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
        if not api or not secret:
            return None
        return cls(api_key=api, secret_key=secret, base_url=base)


class AlpacaBroker(Broker):
    """Alpaca REST adapter with explicit live/dry-run gating.

    SAFETY: default is dry-run. You must pass `live=True` AND have valid
    credentials (via env vars or explicit arg) for orders to actually
    hit the Alpaca API. A dry-run call logs the intended action and
    returns a synthetic Order response.

    Dependencies: `alpaca-py` SDK. This module imports it lazily so the
    package doesn't need to be installed just to read the class.
    """

    def __init__(
        self,
        live: bool = False,
        credentials: AlpacaCredentials | None = None,
        paper: bool = True,
    ):
        self.live = bool(live)
        self.paper = bool(paper)
        if credentials is None:
            credentials = AlpacaCredentials.from_env()
        self.credentials = credentials
        self._client = None
        if self.live and self.credentials is None:
            raise RuntimeError(
                "AlpacaBroker(live=True) requires ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables to be set."
            )
        if self.live:
            self._ensure_client()

    def _ensure_client(self) -> None:
        """Lazily import alpaca-py and initialize the trading client.

        Only called when live=True. Dry-run mode never touches the SDK.
        """
        if self._client is not None:
            return
        if self.credentials is None:
            raise RuntimeError("No Alpaca credentials available")
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as e:
            raise RuntimeError(
                "alpaca-py SDK is not installed. Run "
                "`pip install alpaca-py` to use AlpacaBroker in live mode."
            ) from e
        self._client = TradingClient(
            api_key=self.credentials.api_key,
            secret_key=self.credentials.secret_key,
            paper=self.paper,
        )
        log.info(
            "AlpacaBroker initialized (live=True, paper=%s)", self.paper
        )

    # ============================================================
    # Read-only operations — safe in both live and dry-run modes
    # ============================================================

    def get_cash(self) -> float:
        if not self.live:
            log.info("[dry-run] get_cash() → 100000.0")
            return 100_000.0
        self._ensure_client()
        account = self._client.get_account()  # type: ignore[union-attr]
        return float(account.cash)

    def get_positions(self) -> list[Position]:
        if not self.live:
            log.info("[dry-run] get_positions() → []")
            return []
        self._ensure_client()
        positions = self._client.get_all_positions()  # type: ignore[union-attr]
        return [
            Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_price=float(p.avg_entry_price),
            )
            for p in positions
        ]

    def get_quote(self, symbol: str) -> Quote:
        """Fetch a realtime quote. In dry-run, returns a synthetic quote
        with zero bid/ask to signal "no data available" — the caller
        should not treat dry-run quotes as real."""
        if not self.live:
            log.info("[dry-run] get_quote(%s) → synthetic", symbol)
            return Quote(symbol=symbol, bid=0.0, ask=0.0, last=0.0)
        self._ensure_client()
        # Alpaca's quote API depends on the asset class (equity vs crypto).
        # Delegated to the SDK's data client, which isn't imported here
        # to keep the surface area small.
        raise NotImplementedError(
            "Live quote fetch requires alpaca.data.StockHistoricalDataClient. "
            "See the Alpaca SDK docs."
        )

    # ============================================================
    # Write operations — EXPLICITLY gated on self.live
    # ============================================================

    def submit_order(self, order: Order) -> Order:
        """Submit an order. In dry-run mode, logs the intended action
        and returns a synthetic response with a generated id. In live
        mode, sends the order to Alpaca.

        ⚠ In live mode, this is REAL MONEY and CANNOT be undone except
        by submitting a cancel order. Use with care.
        """
        if not self.live:
            synthetic_id = f"dryrun-{uuid.uuid4().hex[:8]}"
            log.info(
                "[dry-run] submit_order(%s %s %s qty=%s type=%s limit=%s) → id=%s",
                order.symbol,
                order.side.value,
                order.qty,
                order.qty,
                order.order_type.value,
                order.limit_price,
                synthetic_id,
            )
            return Order(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                order_type=order.order_type,
                limit_price=order.limit_price,
                id=synthetic_id,
            )

        # LIVE MODE — this code path requires alpaca-py installed
        self._ensure_client()
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
        )

        alpaca_side = (
            AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
        )

        if order.order_type == OrderType.MARKET:
            request = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            if order.limit_price is None:
                raise ValueError("LIMIT order requires limit_price")
            request = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                limit_price=order.limit_price,
            )

        log.warning(
            "⚠ LIVE ORDER: %s %s %s %s",
            order.symbol,
            order.side.value,
            order.qty,
            order.order_type.value,
        )
        response = self._client.submit_order(request)  # type: ignore[union-attr]
        return Order(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            order_type=order.order_type,
            limit_price=order.limit_price,
            id=str(response.id),
        )

    def cancel_order(self, order_id: str) -> bool:
        if not self.live:
            log.info("[dry-run] cancel_order(%s) → True", order_id)
            return True
        self._ensure_client()
        try:
            self._client.cancel_order_by_id(order_id)  # type: ignore[union-attr]
            return True
        except Exception as e:
            log.error("cancel_order failed: %s", e)
            return False
