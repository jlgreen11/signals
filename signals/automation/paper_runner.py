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
        # Fixed-hold entry tracking. Matches canonical backtest behavior:
        # each position has a fixed entry_date and exits after hold_days.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_entries (
                ticker TEXT PRIMARY KEY,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sector TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Entry tracking (per-position entry_date for fixed-hold logic)
    # ------------------------------------------------------------------

    def _record_entry(self, ticker: str, entry_price: float, sector: str = "Unknown") -> None:
        """Record the entry date for a new position."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT OR REPLACE INTO paper_entries (ticker, entry_date, entry_price, sector) VALUES (?, ?, ?, ?)",
            (ticker, datetime.now(tz=UTC).isoformat(), entry_price, sector),
        )
        conn.commit()
        conn.close()

    def _delete_entry(self, ticker: str) -> None:
        """Remove the entry record when a position is closed."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("DELETE FROM paper_entries WHERE ticker = ?", (ticker,))
        conn.commit()
        conn.close()

    def _get_entries(self) -> dict[str, dict]:
        """Return {ticker: {entry_date, entry_price, sector}} for all tracked positions."""
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT ticker, entry_date, entry_price, sector FROM paper_entries"
            ).fetchall()
        finally:
            conn.close()
        return {
            r[0]: {"entry_date": r[1], "entry_price": r[2], "sector": r[3] or "Unknown"}
            for r in rows
        }

    def _trading_days_between(self, start_iso: str, end: datetime) -> int:
        """Approximate trading-day count between two dates (weekdays only)."""
        from datetime import timedelta
        start = datetime.fromisoformat(start_iso)
        days = 0
        d = start
        while d.date() < end.date():
            d += timedelta(days=1)
            if d.weekday() < 5:
                days += 1
        return days

    def _reconcile_entries(self) -> None:
        """Sync paper_entries with actual broker positions.

        - Positions in broker but not in entries -> insert with entry_date=now.
          (Used when an external process opened a position, or on first run
          with pre-existing positions.)
        - Entries in table but not in broker -> delete.
        """
        held = {p.symbol for p in self._broker.get_positions()}
        tracked = set(self._get_entries().keys())
        # Orphaned entries: position closed externally
        for ticker in tracked - held:
            self._delete_entry(ticker)
        # Untracked positions: record entry_date=now as best effort
        for pos in self._broker.get_positions():
            if pos.symbol not in tracked:
                self._record_entry(pos.symbol, pos.avg_price)

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

    def _get_trading_days_since_last_rebalance(self) -> int:
        """Count equity-calendar trading days since the last rebalance.

        Uses paper_state table key 'last_rebalance_date'. Returns a
        large number (999) if no rebalance has ever been recorded,
        which forces the first run to trade immediately.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT value FROM paper_state WHERE key = 'last_rebalance_date'"
            ).fetchone()
            if not row:
                return 999  # never rebalanced → trade immediately
            last_date = datetime.fromisoformat(row[0])
            # Count weekdays (rough equity calendar) between then and now
            now = datetime.now(tz=UTC)
            days = 0
            d = last_date
            from datetime import timedelta
            while d.date() < now.date():
                d += timedelta(days=1)
                if d.weekday() < 5:  # Mon-Fri
                    days += 1
            return days
        except Exception:
            return 999
        finally:
            conn.close()

    def _record_rebalance(self) -> None:
        """Mark today as the last rebalance date."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value) VALUES (?, ?)",
            ("last_rebalance_date", datetime.now(tz=UTC).isoformat()),
        )
        conn.commit()
        conn.close()

    def execute_daily(
        self,
        earnings_df: pd.DataFrame | None = None,
        rebalance_freq: int = 21,
        hold_days: int = 105,
        n_long: int = 15,
        max_per_sector: int = 2,
    ) -> dict:
        """Run the insights engine and execute the fixed-hold strategy.

        This mirrors the canonical backtest (`bias_free.run_bias_free_backtest`):

        1. Run engine.run_daily() for signal generation + reporting
        2. Reconcile entry tracking with broker positions
        3. FIXED-HOLD EXITS: sell any position held >= hold_days
        4. On rebalance day (every rebalance_freq trading days):
           - Get momentum top-N candidates from engine
           - BUY any candidate not currently held (subject to sector cap)
           - Equal-weight size new positions against total equity
        5. Record daily equity
        """
        now = datetime.now(tz=UTC)

        # Step 1: Signal generation
        report = self.engine.run_daily(earnings_df)
        momentum_targets = report.get("momentum_targets", {})

        # Step 2: Update prices and reconcile entries
        self._update_prices()
        self._reconcile_entries()
        entries = self._get_entries()

        executed_orders: list[dict] = []

        # Step 3: Fixed-hold exits — sell positions that have held >= hold_days
        exits: list[str] = []
        for pos in self._broker.get_positions():
            ticker = pos.symbol
            if ticker not in entries:
                continue
            days_held = self._trading_days_between(entries[ticker]["entry_date"], now)
            if days_held >= hold_days:
                exits.append(ticker)

        for ticker in exits:
            price = self._prices.get(ticker, 0.0)
            if price <= 0:
                log.warning("Cannot exit %s — no price", ticker)
                continue
            pos = next((p for p in self._broker.get_positions() if p.symbol == ticker), None)
            if pos is None:
                continue
            shares = pos.qty
            try:
                order = Order(
                    symbol=ticker, side=OrderSide.SELL, qty=shares,
                    order_type=OrderType.MARKET,
                )
                self._broker.submit_order(order)
                executed_orders.append({
                    "timestamp": now.isoformat(), "ticker": ticker,
                    "action": "SELL_HOLD_EXPIRED", "shares": shares,
                    "price": price, "notional": shares * price,
                })
                self._delete_entry(ticker)
            except ValueError as e:
                log.warning("Exit failed for %s: %s", ticker, e)

        # Step 4: Rebalance day — add new entries
        days_since = self._get_trading_days_since_last_rebalance()
        is_rebalance_day = days_since >= rebalance_freq

        if is_rebalance_day and momentum_targets:
            log.info("REBALANCE DAY (%d days since last)", days_since)

            # Current holdings after exits
            current_tickers = {p.symbol for p in self._broker.get_positions()}
            current_sectors: dict[str, int] = {}
            sectors_map = self.engine.sectors or {}
            for ticker in current_tickers:
                sec = sectors_map.get(ticker, "Unknown")
                current_sectors[sec] = current_sectors.get(sec, 0) + 1

            # Candidates: momentum picks ranked by weight, not already held
            candidates = sorted(
                ((t, w) for t, w in momentum_targets.items()
                 if t not in current_tickers and w > 0),
                key=lambda x: x[1], reverse=True,
            )

            # Fill to n_long with sector cap
            n_slots = n_long - len(current_tickers)
            selected: list[str] = []
            for ticker, _weight in candidates:
                if len(selected) >= n_slots:
                    break
                sec = sectors_map.get(ticker, "Unknown")
                if current_sectors.get(sec, 0) >= max_per_sector:
                    continue
                selected.append(ticker)
                current_sectors[sec] = current_sectors.get(sec, 0) + 1

            # Equal-weight sizing across target portfolio
            if selected:
                equity = self._compute_equity()
                target_n = max(len(current_tickers) + len(selected), n_long)
                per_pos = equity / target_n

                for ticker in selected:
                    price = self._prices.get(ticker, 0.0)
                    if price <= 0:
                        log.warning("No price for %s, skipping", ticker)
                        continue
                    shares = per_pos / price
                    if shares < 0.001:
                        continue
                    try:
                        order = Order(
                            symbol=ticker, side=OrderSide.BUY, qty=shares,
                            order_type=OrderType.MARKET,
                        )
                        self._broker.submit_order(order)
                        sec = sectors_map.get(ticker, "Unknown")
                        self._record_entry(ticker, price, sec)
                        executed_orders.append({
                            "timestamp": now.isoformat(), "ticker": ticker,
                            "action": "BUY", "shares": shares,
                            "price": price, "notional": per_pos,
                        })
                    except ValueError as e:
                        log.warning("Buy failed for %s: %s", ticker, e)

            self._record_rebalance()

        # Step 5: Persist trades + equity
        self._trade_log.extend(executed_orders)
        for trade in executed_orders:
            self._persist_trade(trade)

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
            "rebalance": is_rebalance_day,
            "days_since_rebalance": days_since,
            "n_exits": len(exits),
            "next_rebalance_in": max(0, rebalance_freq - days_since),
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
