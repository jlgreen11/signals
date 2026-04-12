"""Paper trade runner: connects InsightsEngine to the PaperBroker for execution."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from signals.automation.insights_engine import InsightsEngine
from signals.broker.base import Order, OrderSide, OrderType
from signals.broker.paper import PaperBroker
from signals.utils.logging import get_logger

log = get_logger(__name__)


class PaperTradeRunner:
    """Executes blended signals as paper trades and tracks P&L.

    Uses signals/broker/paper.py PaperBroker for in-memory execution.
    All trades are logged to an internal trade log for later analysis.
    """

    def __init__(
        self,
        engine: InsightsEngine,
        initial_capital: float = 100_000.0,
        broker: str = "paper",
    ) -> None:
        self.engine = engine
        self.initial_capital = initial_capital
        self._trade_log: list[dict] = []
        self._daily_equity: list[dict] = []
        self._prices: dict[str, float] = {}

        # Create the paper broker with a quote function that uses cached prices
        self._broker = PaperBroker(
            initial_cash=initial_capital,
            quote_fn=self._get_price,
        )

    def _get_price(self, symbol: str) -> float:
        """Quote function for PaperBroker. Uses cached prices."""
        return self._prices.get(symbol, 0.0)

    def _update_prices(self) -> None:
        """Refresh cached prices from the data store."""
        if self.engine.data_store is None:
            return
        all_tickers = list(
            set(self.engine.tickers + self.engine.tsmom_tickers)
        )
        for ticker in all_tickers:
            try:
                df = self.engine.data_store.load(ticker, "1d")
                if not df.empty:
                    self._prices[ticker] = float(df["close"].iloc[-1])
            except Exception as e:
                log.warning("Price load failed for %s: %s", ticker, e)

    def execute_daily(self, earnings_df: pd.DataFrame | None = None) -> dict:
        """Run the insights engine, compute rebalance trades, execute them.

        Steps:
        1. Run engine.run_daily()
        2. Get current positions from the paper broker
        3. Compute rebalance orders via CashOverlay
        4. Execute each order through the broker
        5. Log all trades
        6. Compute daily P&L
        7. Return execution report
        """
        now = datetime.now(tz=UTC)

        # Step 1: Run daily signals
        report = self.engine.run_daily(earnings_df)
        blended = report["blended_allocation"]

        # Step 2: Update prices
        self._update_prices()

        # Step 3: Get current positions
        current_positions = self._get_position_values()

        # Step 4: Compute rebalance orders
        target_positions = {
            k: v for k, v in blended.items() if k != "_CASH"
        }
        orders = self.engine.cash_overlay.rebalance_orders(
            current_positions=current_positions,
            target_positions=target_positions,
            prices=self._prices,
        )

        # Step 5: Execute orders through broker
        executed_orders: list[dict] = []
        for order_spec in orders:
            ticker = order_spec["ticker"]
            price = self._prices.get(ticker, 0.0)
            if price <= 0:
                log.warning("No price for %s, skipping order", ticker)
                continue

            shares = order_spec["shares"]
            if shares < 0.001:
                continue

            try:
                side = (
                    OrderSide.BUY
                    if order_spec["action"] == "BUY"
                    else OrderSide.SELL
                )
                order = Order(
                    symbol=ticker,
                    side=side,
                    qty=shares,
                    order_type=OrderType.MARKET,
                )
                self._broker.submit_order(order)
                executed_orders.append({
                    "timestamp": now.isoformat(),
                    "ticker": ticker,
                    "action": order_spec["action"],
                    "shares": shares,
                    "price": price,
                    "notional": order_spec["notional"],
                })
            except ValueError as e:
                log.warning("Order failed for %s: %s", ticker, e)

        # Step 6: Log trades
        self._trade_log.extend(executed_orders)

        # Step 7: Compute daily equity
        equity = self._compute_equity()
        self._daily_equity.append({
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "equity": equity,
            "cash": self._broker.get_cash(),
            "n_positions": len(self._broker.get_positions()),
            "n_trades": len(executed_orders),
        })

        return {
            "timestamp": now.isoformat(),
            "report": report,
            "executed_orders": executed_orders,
            "n_orders": len(executed_orders),
            "equity": equity,
            "cash": self._broker.get_cash(),
            "positions": self._get_position_values(),
        }

    def _get_position_values(self) -> dict[str, float]:
        """Get current positions as {ticker: dollar_value}."""
        positions: dict[str, float] = {}
        for pos in self._broker.get_positions():
            price = self._prices.get(pos.symbol, pos.avg_price)
            positions[pos.symbol] = pos.qty * price
        return positions

    def _compute_equity(self) -> float:
        """Compute total portfolio equity (cash + positions at market value)."""
        cash = self._broker.get_cash()
        position_value = sum(self._get_position_values().values())
        return cash + position_value

    def get_positions(self) -> pd.DataFrame:
        """Current paper portfolio positions with mark-to-market values."""
        rows = []
        for pos in self._broker.get_positions():
            price = self._prices.get(pos.symbol, pos.avg_price)
            market_value = pos.qty * price
            unrealized_pnl = (price - pos.avg_price) * pos.qty
            rows.append({
                "ticker": pos.symbol,
                "shares": pos.qty,
                "avg_price": pos.avg_price,
                "current_price": price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
            })
        if not rows:
            return pd.DataFrame(
                columns=["ticker", "shares", "avg_price", "current_price",
                         "market_value", "unrealized_pnl"]
            )
        return pd.DataFrame(rows)

    def get_performance(self) -> dict:
        """Performance since inception: total return, Sharpe, max DD, etc."""
        if not self._daily_equity:
            return {
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "n_days": 0,
                "current_equity": self.initial_capital,
            }

        equities = [e["equity"] for e in self._daily_equity]
        eq_series = pd.Series(equities)
        total_return = (equities[-1] / self.initial_capital) - 1.0

        # Simple Sharpe from daily returns
        if len(equities) >= 2:
            daily_returns = eq_series.pct_change().dropna()
            mean_r = daily_returns.mean()
            std_r = daily_returns.std()
            sharpe = (
                float(mean_r / std_r * (252 ** 0.5))
                if std_r > 0
                else 0.0
            )
        else:
            sharpe = 0.0

        # Max drawdown
        peak = eq_series.cummax()
        drawdown = (eq_series - peak) / peak
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "n_days": len(self._daily_equity),
            "current_equity": equities[-1],
            "initial_capital": self.initial_capital,
        }

    def get_trade_log(self) -> pd.DataFrame:
        """Full trade history."""
        if not self._trade_log:
            return pd.DataFrame(
                columns=["timestamp", "ticker", "action", "shares",
                         "price", "notional"]
            )
        return pd.DataFrame(self._trade_log)
