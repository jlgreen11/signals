"""In-memory paper broker for dry runs."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from signals.broker.base import (
    Broker,
    Order,
    OrderSide,
    OrderType,
    Position,
    Quote,
)


class PaperBroker(Broker):
    """A trivial broker that fills orders instantly at the latest quote.

    `quote_fn(symbol) -> float` provides the current price.
    """

    def __init__(self, initial_cash: float, quote_fn: Callable[[str], float]):
        self._cash = float(initial_cash)
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._quote_fn = quote_fn

    def get_cash(self) -> float:
        return self._cash

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def submit_order(self, order: Order) -> Order:
        order = Order(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            order_type=order.order_type,
            limit_price=order.limit_price,
            id=order.id or str(uuid.uuid4()),
        )
        price = self._quote_fn(order.symbol)
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == OrderSide.BUY and price > order.limit_price:
                self._orders[order.id] = order  # remains open
                return order
            if order.side == OrderSide.SELL and price < order.limit_price:
                self._orders[order.id] = order
                return order
            price = order.limit_price

        cost = price * order.qty
        if order.side == OrderSide.BUY:
            if cost > self._cash:
                raise ValueError("insufficient cash")
            self._cash -= cost
            pos = self._positions.get(order.symbol)
            if pos is None:
                self._positions[order.symbol] = Position(order.symbol, order.qty, price)
            else:
                new_qty = pos.qty + order.qty
                pos.avg_price = (pos.avg_price * pos.qty + price * order.qty) / new_qty
                pos.qty = new_qty
        else:
            pos = self._positions.get(order.symbol)
            if pos is None or pos.qty < order.qty:
                raise ValueError("insufficient position")
            self._cash += cost
            pos.qty -= order.qty
            if pos.qty == 0:
                del self._positions[order.symbol]
        return order

    def cancel_order(self, order_id: str) -> bool:
        return self._orders.pop(order_id, None) is not None

    def get_quote(self, symbol: str) -> Quote:
        price = self._quote_fn(symbol)
        return Quote(symbol=symbol, bid=price, ask=price, last=price)
