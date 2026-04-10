"""Cash + position tracking with commission, slippage, longs/shorts, and stops.

The portfolio is target-driven: each bar, the engine computes a desired position
fraction in the closed interval [-max_short, +max_long] (where 1.0 = 100% of
equity long, -1.0 = 100% short). `set_target(...)` reconciles by trading the
delta. Stops are checked separately each bar via `check_stop(...)`.

Conventions
-----------
- Positive `qty` = long, negative `qty` = short.
- Commissions and slippage are applied to the *traded notional* on every fill.
- Equity for short positions is mark-to-market: equity = cash + qty * price
  (when qty < 0, a price drop increases cash needed to cover, so equity falls).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    ts: pd.Timestamp
    side: str  # "BUY", "SELL", "SHORT", "COVER", "STOP"
    price: float
    qty: float            # positive number of units traded
    commission: float
    pnl: float = 0.0
    reason: str = ""      # "target", "stop", "flatten"


@dataclass
class Portfolio:
    """Long/short portfolio with target-fraction sizing and stop-loss."""

    initial_cash: float
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    cash: float = field(init=False)
    qty: float = field(init=False, default=0.0)
    avg_price: float = field(init=False, default=0.0)
    trades: list[Trade] = field(init=False, default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    # ----- State -----
    @property
    def position(self) -> float:
        return self.qty

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    def equity(self, price: float) -> float:
        return self.cash + self.qty * price

    def mark(self, ts: pd.Timestamp, price: float) -> None:
        self.equity_curve.append((ts, self.equity(price)))

    def position_fraction(self, price: float) -> float:
        eq = self.equity(price)
        if eq <= 0:
            return 0.0
        return (self.qty * price) / eq

    # ----- Trading -----
    def set_target(
        self,
        ts: pd.Timestamp,
        price: float,
        target_fraction: float,
        reason: str = "target",
        min_trade_fraction: float = 0.0,
    ) -> None:
        """Adjust position so that target_fraction of equity is invested.

        target_fraction in [-1, +1] (or wider if you allow leverage).
            +1.0 = 100% long
            -1.0 = 100% short
             0.0 = flat

        `min_trade_fraction` is a deadband: if the change in position-as-fraction-of-equity
        is smaller than this, the trade is skipped (avoids commission/slippage churn on
        tiny rebalances). Crossing zero (long↔short or in↔out) always trades regardless.
        """
        if price <= 0:
            return

        equity = self.equity(price)
        if equity <= 0:
            return

        current_frac = self.position_fraction(price)
        delta_frac = target_fraction - current_frac

        # Always allow trades that cross zero (open/close/flip).
        crosses_zero = (
            (current_frac == 0 and target_fraction != 0)
            or (target_fraction == 0 and current_frac != 0)
            or (np.sign(current_frac) != np.sign(target_fraction) and target_fraction != 0 and current_frac != 0)
        )
        if not crosses_zero and abs(delta_frac) < min_trade_fraction:
            return

        target_qty = (target_fraction * equity) / price
        delta = target_qty - self.qty
        if abs(delta * price) < 1e-6:
            return

        if delta > 0:
            self._buy(ts, price, delta, reason)
        else:
            self._sell(ts, price, -delta, reason)

    def check_stop(
        self,
        ts: pd.Timestamp,
        price: float,
        stop_loss_pct: float,
    ) -> bool:
        """If the open position has moved against entry by stop_loss_pct, flatten.

        Returns True if a stop fired.
        """
        if self.is_flat or stop_loss_pct <= 0 or self.avg_price <= 0:
            return False

        if self.is_long:
            adverse = (self.avg_price - price) / self.avg_price
        else:
            adverse = (price - self.avg_price) / self.avg_price

        if adverse >= stop_loss_pct:
            self.set_target(ts, price, 0.0, reason="stop")
            if self.trades:
                self.trades[-1].reason = "stop"
            return True
        return False

    def flatten(self, ts: pd.Timestamp, price: float) -> None:
        self.set_target(ts, price, 0.0, reason="flatten")

    # ----- Internal fills -----
    def _buy(self, ts: pd.Timestamp, price: float, qty: float, reason: str) -> None:
        """Buy `qty` units (positive). Closes any short before going long."""
        fill = price * (1 + self.slippage_bps * 1e-4)
        notional = qty * fill
        commission = notional * self.commission_bps * 1e-4

        if self.qty < 0:
            # Covering a short — realize PnL on covered units.
            cover = min(qty, -self.qty)
            cover_pnl = (self.avg_price - fill) * cover  # short profits when fill < avg
            self.cash -= cover * fill
            self.cash -= commission * (cover / qty)
            self.qty += cover
            if self.qty == 0:
                self.avg_price = 0.0
            self.trades.append(
                Trade(
                    ts=ts, side="COVER", price=fill, qty=cover,
                    commission=commission * (cover / qty), pnl=cover_pnl, reason=reason,
                )
            )
            remainder = qty - cover
            if remainder > 1e-12:
                rem_notional = remainder * fill
                rem_commission = rem_notional * self.commission_bps * 1e-4
                self.cash -= rem_notional + rem_commission
                self.qty = remainder
                self.avg_price = fill
                self.trades.append(
                    Trade(
                        ts=ts, side="BUY", price=fill, qty=remainder,
                        commission=rem_commission, reason=reason,
                    )
                )
            return

        # Pure long add (or open).
        self.cash -= notional + commission
        new_qty = self.qty + qty
        if new_qty > 0:
            self.avg_price = ((self.avg_price * self.qty) + (fill * qty)) / new_qty
        self.qty = new_qty
        self.trades.append(
            Trade(ts=ts, side="BUY", price=fill, qty=qty, commission=commission, reason=reason)
        )

    def _sell(self, ts: pd.Timestamp, price: float, qty: float, reason: str) -> None:
        """Sell `qty` units (positive). Closes any long before going short."""
        fill = price * (1 - self.slippage_bps * 1e-4)
        notional = qty * fill
        commission = notional * self.commission_bps * 1e-4

        if self.qty > 0:
            sell_qty = min(qty, self.qty)
            sell_pnl = (fill - self.avg_price) * sell_qty
            self.cash += sell_qty * fill
            self.cash -= commission * (sell_qty / qty)
            self.qty -= sell_qty
            if self.qty == 0:
                self.avg_price = 0.0
            self.trades.append(
                Trade(
                    ts=ts, side="SELL", price=fill, qty=sell_qty,
                    commission=commission * (sell_qty / qty), pnl=sell_pnl, reason=reason,
                )
            )
            remainder = qty - sell_qty
            if remainder > 1e-12:
                rem_notional = remainder * fill
                rem_commission = rem_notional * self.commission_bps * 1e-4
                # Open a short for the remainder.
                self.cash += rem_notional - rem_commission
                self.qty = -remainder
                self.avg_price = fill
                self.trades.append(
                    Trade(
                        ts=ts, side="SHORT", price=fill, qty=remainder,
                        commission=rem_commission, reason=reason,
                    )
                )
            return

        # Pure short add (or open).
        self.cash += notional - commission
        new_qty = self.qty - qty  # qty<0 grows more negative
        # Weighted-average entry for shorts (track |qty|).
        old_abs = -self.qty
        new_abs = old_abs + qty
        if new_abs > 0:
            self.avg_price = ((self.avg_price * old_abs) + (fill * qty)) / new_abs
        self.qty = new_qty
        self.trades.append(
            Trade(ts=ts, side="SHORT", price=fill, qty=qty, commission=commission, reason=reason)
        )

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        ts, eq = zip(*self.equity_curve, strict=True)
        return pd.Series(eq, index=pd.DatetimeIndex(ts), name="equity")
