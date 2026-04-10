"""Parquet + SQLite storage for OHLCV bars and run metadata."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from signals.utils.logging import get_logger

log = get_logger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    source TEXT NOT NULL,
    rows_added INTEGER NOT NULL,
    first_ts TEXT,
    last_ts TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    signal TEXT NOT NULL,
    confidence REAL NOT NULL,
    state INTEGER NOT NULL,
    expected_return REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    n_states INTEGER,
    train_window INTEGER,
    retrain_freq INTEGER,
    sharpe REAL,
    cagr REAL,
    max_drawdown REAL,
    win_rate REAL,
    profit_factor REAL,
    calmar REAL,
    final_equity REAL,
    n_trades INTEGER,
    created_at TEXT NOT NULL
);
"""


class DataStore:
    """File-system + SQLite store for OHLCV bars and run metadata."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.db_path = self.base_dir / "signals.db"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ----- SQLite -----
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ----- Parquet bars -----
    def parquet_path(self, symbol: str, interval: str) -> Path:
        safe = symbol.replace("/", "_").replace("^", "")
        return self.raw_dir / f"{safe}_{interval}.parquet"

    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        path = self.parquet_path(symbol, interval)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df.sort_index()

    def save(self, symbol: str, interval: str, df: pd.DataFrame) -> Path:
        path = self.parquet_path(symbol, interval)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df.to_parquet(path)
        return path

    def append(self, symbol: str, interval: str, new_df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
        existing = self.load(symbol, interval)
        if existing.empty:
            merged = new_df
            added = len(new_df)
        else:
            merged = pd.concat([existing, new_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            added = len(merged) - len(existing)
        self.save(symbol, interval, merged)
        return added, merged

    def last_timestamp(self, symbol: str, interval: str) -> pd.Timestamp | None:
        df = self.load(symbol, interval)
        if df.empty:
            return None
        return df.index.max()

    # ----- Metadata writes -----
    def log_fetch(
        self,
        symbol: str,
        interval: str,
        source: str,
        rows_added: int,
        first_ts: pd.Timestamp | None,
        last_ts: pd.Timestamp | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fetch_log
                    (symbol, interval, source, rows_added, first_ts, last_ts, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    interval,
                    source,
                    rows_added,
                    first_ts.isoformat() if first_ts is not None else None,
                    last_ts.isoformat() if last_ts is not None else None,
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )

    def list_datasets(self) -> list[dict]:
        rows: list[dict] = []
        for path in sorted(self.raw_dir.glob("*.parquet")):
            stem = path.stem
            if "_" not in stem:
                continue
            symbol, interval = stem.rsplit("_", 1)
            df = pd.read_parquet(path)
            rows.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "rows": len(df),
                    "first": str(df.index.min()) if len(df) else None,
                    "last": str(df.index.max()) if len(df) else None,
                    "path": str(path),
                }
            )
        return rows

    def record_signal(
        self,
        symbol: str,
        ts: pd.Timestamp,
        signal: str,
        confidence: float,
        state: int,
        expected_return: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals
                    (symbol, ts, signal, confidence, state, expected_return, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    ts.isoformat(),
                    signal,
                    float(confidence),
                    int(state),
                    float(expected_return),
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )

    def recent_signals(self, symbol: str, days: int) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT ts, signal, confidence, state, expected_return, created_at
                FROM signals
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                conn,
                params=(symbol, days),
            )
        return df

    def record_backtest(self, row: dict) -> int:
        keys = list(row.keys())
        placeholders = ",".join(["?"] * len(keys))
        cols = ",".join(keys)
        with self._connect() as conn:
            cur = conn.execute(
                f"INSERT INTO backtest_runs ({cols}, created_at) VALUES ({placeholders}, ?)",
                (*row.values(), datetime.now(tz=timezone.utc).isoformat()),
            )
            return int(cur.lastrowid or 0)

    def list_backtests(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM backtest_runs ORDER BY id DESC", conn
            )

    def get_backtest(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM backtest_runs WHERE id = ?", conn, params=(run_id,)
            )
