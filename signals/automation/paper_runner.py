"""Paper trade runner: connects InsightsEngine to the PaperBroker for execution.

Persistence: positions, cash, trade log, and daily equity are stored in
SQLite (same database as SignalStore) so they survive between CLI calls.
On init, the runner loads any previously-saved state; on every execute_daily
it saves the new state. This means `signals auto trade` → `signals auto
positions` works across separate invocations.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from signals.automation.insights_engine import InsightsEngine
from signals.broker.base import Order, OrderSide, OrderType
from signals.broker.paper import PaperBroker
from signals.utils.logging import get_logger

log = get_logger(__name__)


class PaperTradeRunner:
    """Executes blended signals as paper trades and tracks P&L.

    Supports two broker backends:
      - "paper" (default): in-memory PaperBroker with SQLite persistence
        for positions/cash/trades/equity between CLI calls.
      - "alpaca": Alpaca Trading API (paper or live). Positions, fills,
        and P&L are tracked by Alpaca's servers — no local persistence
        needed. Requires ALPACA_API_KEY + ALPACA_SECRET_KEY env vars.
    """

    def __init__(
        self,
        engine: InsightsEngine,
        initial_capital: float = 100_000.0,
        broker: str = "paper",
        db_path: str | Path | None = None,
    ) -> None:
        self.engine = engine
        self.initial_capital = initial_capital
        self.broker_type = broker
        self._trade_log: list[dict] = []
        self._daily_equity: list[dict] = []
        self._prices: dict[str, float] = {}

        # SQLite persistence path (default: same dir as SignalStore)
        if db_path is None:
            from signals.config import SETTINGS
            db_path = SETTINGS.data.dir / "signals.db"
        self._db_path = str(db_path)
        self._init_db()

        # Initialize the appropriate broker backend
        if broker == "alpaca":
            from signals.broker.alpaca import AlpacaBroker
            self._broker = AlpacaBroker(live=True, paper=True)
            self._use_alpaca = True
            log.info("Using Alpaca paper trading API")
        else:
            # Local PaperBroker with SQLite persistence
            self._broker = PaperBroker(
                initial_cash=initial_capital,
                quote_fn=self._get_price,
            )
            self._use_alpaca = False

        # Load previously saved state (if using local paper broker)
        if not self._use_alpaca:
            self._load_state()

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create persistence tables if they don't exist."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                ticker TEXT PRIMARY KEY,
                shares REAL NOT NULL,
                avg_price REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                notional REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                n_positions INTEGER NOT NULL,
                n_trades INTEGER NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _save_state(self) -> None:
        """Persist current positions and cash to SQLite."""
        conn = sqlite3.connect(self._db_path)
        # Clear and rewrite positions
        conn.execute("DELETE FROM paper_positions")
        for pos in self._broker.get_positions():
            conn.execute(
                "INSERT INTO paper_positions (ticker, shares, avg_price) VALUES (?, ?, ?)",
                (pos.symbol, pos.qty, pos.avg_price),
            )
        # Save cash
        conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value) VALUES (?, ?)",
            ("cash", str(self._broker.get_cash())),
        )
        conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value) VALUES (?, ?)",
            ("initial_capital", str(self.initial_capital)),
        )
        conn.commit()
        conn.close()

    def _load_state(self) -> None:
        """Load previously-saved positions and cash from SQLite."""
        conn = sqlite3.connect(self._db_path)
        try:
            # Load cash
            row = conn.execute(
                "SELECT value FROM paper_state WHERE key = 'cash'"
            ).fetchone()
            if row:
                self._broker._cash = float(row[0])

            # Load positions
            rows = conn.execute(
                "SELECT ticker, shares, avg_price FROM paper_positions"
            ).fetchall()
            from signals.broker.base import Position
            self._broker._positions = {}
            for ticker, shares, avg_price in rows:
                self._broker._positions[ticker] = Position(ticker, shares, avg_price)

            # Load trade log
            trade_rows = conn.execute(
                "SELECT timestamp, ticker, action, shares, price, notional "
                "FROM paper_trades ORDER BY id"
            ).fetchall()
            self._trade_log = [
                {"timestamp": r[0], "ticker": r[1], "action": r[2],
                 "shares": r[3], "price": r[4], "notional": r[5]}
                for r in trade_rows
            ]

            # Load equity history
            eq_rows = conn.execute(
                "SELECT timestamp, date, equity, cash, n_positions, n_trades "
                "FROM paper_equity ORDER BY id"
            ).fetchall()
            self._daily_equity = [
                {"timestamp": r[0], "date": r[1], "equity": r[2],
                 "cash": r[3], "n_positions": r[4], "n_trades": r[5]}
                for r in eq_rows
            ]
        except Exception as e:
            log.warning("Failed to load paper state: %s", e)
        finally:
            conn.close()

    def _persist_trade(self, trade: dict) -> None:
        """Append a single trade to the persistent log."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO paper_trades (timestamp, ticker, action, shares, price, notional) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (trade["timestamp"], trade["ticker"], trade["action"],
             trade["shares"], trade["price"], trade["notional"]),
        )
        conn.commit()
        conn.close()

    def _persist_equity(self, record: dict) -> None:
        """Append a daily equity record."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO paper_equity (timestamp, date, equity, cash, n_positions, n_trades) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (record["timestamp"], record["date"], record["equity"],
             record["cash"], record["n_positions"], record["n_trades"]),
        )
        conn.commit()
        conn.close()

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

        # Step 6: Log trades (in-memory + persist to SQLite)
        self._trade_log.extend(executed_orders)
        for trade in executed_orders:
            self._persist_trade(trade)

        # Step 7: Compute daily equity and persist
        equity = self._compute_equity()
        eq_record = {
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "equity": equity,
            "cash": self._broker.get_cash(),
            "n_positions": len(self._broker.get_positions()),
            "n_trades": len(executed_orders),
        }
        self._daily_equity.append(eq_record)
        self._persist_equity(eq_record)

        # Step 8: Save position state to SQLite (local paper broker only;
        # Alpaca tracks its own state server-side)
        if not self._use_alpaca:
            self._save_state()

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
