"""SQLite-backed persistent signal store for multi-model signal recording."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

AUTOMATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS auto_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    ticker TEXT NOT NULL,
    signal TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    confidence REAL NOT NULL DEFAULT 0.0,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    targets TEXT NOT NULL,
    cash_pct REAL NOT NULL DEFAULT 0.0
);
"""


class SignalStore:
    """SQLite-backed store for model signals.

    Uses the same database as DataStore (data/signals.db) but with its own
    tables (auto_signals, portfolio_targets) to avoid collision with the
    existing signals table used by the Markov models.
    """

    def __init__(self, db_path: str = "data/signals.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(AUTOMATION_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record_signal(
        self,
        model: str,
        ticker: str,
        signal: str,
        weight: float = 0.0,
        confidence: float = 0.0,
        metadata: dict | None = None,
    ) -> int:
        """Record a signal from a model for a given ticker.

        Returns the row id of the inserted record.
        """
        now = datetime.now(tz=UTC).isoformat()
        meta_json = json.dumps(metadata) if metadata else None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO auto_signals
                    (timestamp, model, ticker, signal, weight, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now, model, ticker, signal, float(weight), float(confidence), meta_json),
            )
            return int(cur.lastrowid or 0)

    def record_portfolio_target(
        self,
        model: str,
        targets: dict[str, float],
        cash_pct: float = 0.0,
    ) -> int:
        """Record a portfolio target snapshot for a model.

        Returns the row id of the inserted record.
        """
        now = datetime.now(tz=UTC).isoformat()
        targets_json = json.dumps(targets)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO portfolio_targets
                    (timestamp, model, targets, cash_pct)
                VALUES (?, ?, ?, ?)
                """,
                (now, model, targets_json, float(cash_pct)),
            )
            return int(cur.lastrowid or 0)

    def get_latest_signals(
        self, model: str | None = None, n: int = 50
    ) -> pd.DataFrame:
        """Retrieve the most recent signals, optionally filtered by model."""
        with self._connect() as conn:
            if model:
                df = pd.read_sql_query(
                    """
                    SELECT id, timestamp, model, ticker, signal, weight, confidence, metadata
                    FROM auto_signals
                    WHERE model = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    conn,
                    params=(model, n),
                )
            else:
                df = pd.read_sql_query(
                    """
                    SELECT id, timestamp, model, ticker, signal, weight, confidence, metadata
                    FROM auto_signals
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    conn,
                    params=(n,),
                )
        return df

    def get_latest_targets(self, model: str | None = None) -> dict:
        """Get the most recent portfolio target for each model (or a specific one).

        Returns a dict like {'momentum': {'AAPL': 0.2, ...}, 'tsmom': {...}}.
        """
        with self._connect() as conn:
            if model:
                rows = conn.execute(
                    """
                    SELECT model, targets, cash_pct
                    FROM portfolio_targets
                    WHERE model = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (model,),
                ).fetchall()
            else:
                # Get the latest target for EACH model
                rows = conn.execute(
                    """
                    SELECT model, targets, cash_pct
                    FROM portfolio_targets
                    WHERE id IN (
                        SELECT MAX(id) FROM portfolio_targets GROUP BY model
                    )
                    ORDER BY model
                    """
                ).fetchall()

        result: dict[str, dict] = {}
        for model_name, targets_json, cash_pct in rows:
            result[model_name] = {
                "targets": json.loads(targets_json),
                "cash_pct": cash_pct,
            }
        return result

    def get_signal_history(
        self, ticker: str, model: str | None = None, days: int = 90
    ) -> pd.DataFrame:
        """Retrieve signal history for a ticker over the last N days."""
        cutoff = (
            datetime.now(tz=UTC)
            - pd.Timedelta(days=days)
        ).isoformat()

        with self._connect() as conn:
            if model:
                df = pd.read_sql_query(
                    """
                    SELECT id, timestamp, model, ticker, signal, weight, confidence, metadata
                    FROM auto_signals
                    WHERE ticker = ? AND model = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    """,
                    conn,
                    params=(ticker, model, cutoff),
                )
            else:
                df = pd.read_sql_query(
                    """
                    SELECT id, timestamp, model, ticker, signal, weight, confidence, metadata
                    FROM auto_signals
                    WHERE ticker = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    """,
                    conn,
                    params=(ticker, cutoff),
                )
        return df
