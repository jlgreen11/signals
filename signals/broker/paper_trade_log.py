"""Paper-trading scaffolding — daily signal logging + fill reconciliation.

The goal of paper-trading is to answer the question that no amount of
backtesting can: **does the backtest's fee/slippage model match what
happens in the real world when I actually trade?**

The protocol is simple:

1. Each trading day, run `signals paper-trade record SYMBOL` before
   market open. It runs the production signal generator and appends
   the target position, expected execution price (next open), and
   metadata to a log.

2. After market close, run `signals paper-trade reconcile SYMBOL`.
   It reads the next bar's actual open and close from the data store,
   computes the "would-have-been-filled" price (at the open, with
   modeled slippage), and records the realized return for that day's
   signal.

3. At the end of the month, run `signals paper-trade report SYMBOL`
   to compare the backtest's projected PnL vs the realized PnL from
   the log. If the realized-vs-backtest PnL is within ±20%, the
   backtest is trustworthy. If realized is much worse, the execution
   model is hiding costs.

The log is a JSON file on disk so it's durable across sessions. One
file per symbol.

This module is the SCAFFOLDING. Actually running it for 30 days is the
user's responsibility — no amount of automation can compress calendar
time.

Schema of the log file (signals/paper_trade/SYMBOL.json):

    {
      "symbol": "BTC-USD",
      "started_at": "2026-04-11T00:00:00Z",
      "entries": [
        {
          "date": "2026-04-11",
          "signal": "BUY",
          "target_position": 0.85,
          "expected_fill_price": 65000.0,
          "signal_model": "hybrid",
          "signal_params": {"vol_quantile": 0.70, ...},
          "reconciled": false,
          "actual_open": null,
          "actual_close": null,
          "realized_return": null,
          "backtest_return": null
        },
        ...
      ]
    }

After reconciliation, each entry has `actual_open`, `actual_close`,
`realized_return`, and `backtest_return` populated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG_DIR = Path("paper_trade")


@dataclass
class PaperTradeEntry:
    date: str                       # ISO date of the signal (e.g. "2026-04-11")
    signal: str                     # "BUY", "SELL", "HOLD"
    target_position: float          # fraction of equity in [-1, 1]
    expected_fill_price: float      # price the signal assumed for execution
    signal_model: str               # model type that generated the signal
    signal_params: dict[str, Any]   # model hyperparameters for reproducibility
    reconciled: bool = False
    actual_open: float | None = None
    actual_close: float | None = None
    realized_return: float | None = None
    backtest_return: float | None = None

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "signal": self.signal,
            "target_position": self.target_position,
            "expected_fill_price": self.expected_fill_price,
            "signal_model": self.signal_model,
            "signal_params": self.signal_params,
            "reconciled": self.reconciled,
            "actual_open": self.actual_open,
            "actual_close": self.actual_close,
            "realized_return": self.realized_return,
            "backtest_return": self.backtest_return,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PaperTradeEntry:
        return cls(
            date=d["date"],
            signal=d["signal"],
            target_position=d["target_position"],
            expected_fill_price=d["expected_fill_price"],
            signal_model=d["signal_model"],
            signal_params=d.get("signal_params", {}),
            reconciled=d.get("reconciled", False),
            actual_open=d.get("actual_open"),
            actual_close=d.get("actual_close"),
            realized_return=d.get("realized_return"),
            backtest_return=d.get("backtest_return"),
        )


@dataclass
class PaperTradeLog:
    symbol: str
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    entries: list[PaperTradeEntry] = field(default_factory=list)

    def path(self, log_dir: Path | None = None) -> Path:
        base = log_dir or LOG_DIR
        return base / f"{self._safe_symbol()}.json"

    def _safe_symbol(self) -> str:
        return self.symbol.replace("^", "").replace("/", "_")

    def save(self, log_dir: Path | None = None) -> Path:
        path = self.path(log_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": self.symbol,
            "started_at": self.started_at,
            "entries": [e.to_dict() for e in self.entries],
        }
        path.write_text(json.dumps(payload, indent=2))
        path.chmod(0o600)
        return path

    @classmethod
    def load(cls, symbol: str, log_dir: Path | None = None) -> PaperTradeLog:
        """Load an existing log or return an empty one if the file doesn't exist."""
        instance = cls(symbol=symbol)
        path = instance.path(log_dir)
        if not path.exists():
            return instance
        data = json.loads(path.read_text())
        instance.symbol = data["symbol"]
        instance.started_at = data.get("started_at", instance.started_at)
        instance.entries = [
            PaperTradeEntry.from_dict(e) for e in data.get("entries", [])
        ]
        return instance

    def append(self, entry: PaperTradeEntry) -> None:
        self.entries.append(entry)

    def unreconciled_entries(self) -> list[PaperTradeEntry]:
        return [e for e in self.entries if not e.reconciled]

    def reconciled_entries(self) -> list[PaperTradeEntry]:
        return [e for e in self.entries if e.reconciled]

    def reconcile(
        self,
        date: str,
        actual_open: float,
        actual_close: float,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0,
    ) -> PaperTradeEntry | None:
        """Mark the entry for `date` as reconciled with the given prices.

        Computes realized_return assuming the signal was filled at the
        actual_open with the specified commission/slippage. Also computes
        backtest_return using the expected_fill_price (what the backtest
        would have assumed).
        """
        entry = next((e for e in self.entries if e.date == date), None)
        if entry is None:
            return None
        if entry.reconciled:
            return entry

        # Slippage: long pays extra on buy, gets less on sell
        sign = 1.0 if entry.target_position > 0 else (
            -1.0 if entry.target_position < 0 else 0.0
        )
        slippage_multiplier = 1.0 + sign * slippage_bps * 1e-4
        commission_factor = 1.0 - commission_bps * 1e-4

        # Realized: fill at actual_open with slippage + commission, held to close
        actual_fill = actual_open * slippage_multiplier
        realized_return = (
            (actual_close / actual_fill - 1.0)
            * entry.target_position
            * commission_factor
        )

        # Backtest: fill at expected_fill_price (what the signal assumed),
        # also held to close
        expected_fill = entry.expected_fill_price * slippage_multiplier
        backtest_return = (
            (actual_close / expected_fill - 1.0)
            * entry.target_position
            * commission_factor
        )

        entry.reconciled = True
        entry.actual_open = actual_open
        entry.actual_close = actual_close
        entry.realized_return = realized_return
        entry.backtest_return = backtest_return
        return entry

    def summary(self) -> dict:
        """Return a summary of the log's current state."""
        rec = self.reconciled_entries()
        if not rec:
            return {
                "symbol": self.symbol,
                "total_entries": len(self.entries),
                "reconciled": 0,
                "unreconciled": len(self.entries),
            }
        realized = sum(e.realized_return or 0.0 for e in rec)
        backtest = sum(e.backtest_return or 0.0 for e in rec)
        delta = realized - backtest
        return {
            "symbol": self.symbol,
            "total_entries": len(self.entries),
            "reconciled": len(rec),
            "unreconciled": len(self.entries) - len(rec),
            "cumulative_realized_return": realized,
            "cumulative_backtest_return": backtest,
            "delta_realized_minus_backtest": delta,
            "delta_pct": (
                delta / abs(backtest) * 100.0
                if backtest != 0
                else 0.0
            ),
        }
