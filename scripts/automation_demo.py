#!/usr/bin/env python3
"""Demonstration of the full automation pipeline.

Runs without any external data by using synthetic price data to exercise
all components end-to-end: SignalStore, CashOverlay, InsightsEngine,
PaperTradeRunner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.automation.cash_overlay import CashOverlay
from signals.automation.insights_engine import InsightsEngine
from signals.automation.paper_runner import PaperTradeRunner
from signals.automation.signal_store import SignalStore


def make_synthetic_prices(
    tickers: list[str],
    n_days: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, tz="UTC")
    prices_dict: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        # Random walk with drift
        log_returns = rng.normal(0.0003, 0.02, size=n_days)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        df = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.005, n_days)),
                "high": close * (1 + abs(rng.normal(0, 0.01, n_days))),
                "low": close * (1 - abs(rng.normal(0, 0.01, n_days))),
                "close": close,
                "volume": rng.integers(1_000_000, 50_000_000, size=n_days),
            },
            index=dates[:n_days],
        )
        prices_dict[ticker] = df
    return prices_dict


class SyntheticDataStore:
    """Minimal DataStore stand-in that serves synthetic prices from memory."""

    def __init__(self, prices_dict: dict[str, pd.DataFrame]) -> None:
        self._prices = prices_dict
        self.db_path = ":memory:"

    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        return self._prices.get(symbol, pd.DataFrame())


def main() -> None:
    print("=" * 60)
    print("  AUTOMATION PIPELINE DEMO")
    print("=" * 60)
    print()

    # --- Tickers ---
    momentum_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "AVGO", "JPM", "UNH",
        "V", "MA", "HD", "PG", "JNJ",
        "COST", "ABBV", "CRM", "AMD", "NFLX",
    ]
    tsmom_tickers = ["SPY", "EFA", "TLT", "IEF", "GLD", "USO", "DBA", "BTC-USD"]
    all_tickers = list(set(momentum_tickers + tsmom_tickers))

    # --- Synthetic data ---
    print("[1/5] Generating synthetic price data...")
    prices = make_synthetic_prices(all_tickers, n_days=500)
    data_store = SyntheticDataStore(prices)
    print(f"  Created {len(all_tickers)} tickers x 500 days of synthetic data")
    print()

    # --- Component 1: Signal Store ---
    print("[2/5] Initializing SignalStore (in-memory SQLite)...")
    import os
    import tempfile
    tmp_db = os.path.join(tempfile.mkdtemp(), "test_signals.db")
    store = SignalStore(db_path=tmp_db)
    print(f"  DB path: {tmp_db}")
    print()

    # --- Component 2: Cash Overlay ---
    print("[3/5] Configuring CashOverlay...")
    overlay = CashOverlay(
        total_capital=100_000.0,
        model_weights={"momentum": 0.50, "tsmom": 0.30, "pead": 0.20},
        max_position_pct=0.25,
        cash_reserve_pct=0.05,
    )
    print(overlay.summary())
    print()

    # --- Component 3: Insights Engine ---
    print("[4/5] Running InsightsEngine.run_daily()...")
    engine = InsightsEngine(
        signal_store=store,
        cash_overlay=overlay,
        data_store=data_store,
        tickers=momentum_tickers,
        tsmom_tickers=tsmom_tickers,
    )
    report = engine.run_daily()
    print()
    print(report["report_text"])
    print()

    # --- Component 4: Paper Trade Runner ---
    print("[5/5] Running PaperTradeRunner.execute_daily()...")
    runner = PaperTradeRunner(engine=engine, initial_capital=100_000.0)
    result = runner.execute_daily()
    print()

    # --- Results ---
    print("-" * 60)
    print("EXECUTION SUMMARY")
    print("-" * 60)
    print(f"  Orders executed:    {result['n_orders']}")
    print(f"  Portfolio equity:   ${result['equity']:,.2f}")
    print(f"  Cash remaining:     ${result['cash']:,.2f}")
    n_pos = len(runner.get_positions())
    print(f"  Open positions:     {n_pos}")
    print()

    # Show positions
    positions = runner.get_positions()
    if not positions.empty:
        print("POSITIONS:")
        print(
            positions[["ticker", "shares", "current_price", "market_value"]]
            .to_string(index=False, float_format="${:,.2f}".format)
        )
    print()

    # Show what trades would be executed
    trade_log = runner.get_trade_log()
    if not trade_log.empty:
        print(f"TRADE LOG ({len(trade_log)} trades):")
        print(
            trade_log[["ticker", "action", "shares", "price", "notional"]]
            .head(10)
            .to_string(index=False)
        )
    print()

    # Show performance
    perf = runner.get_performance()
    print("PERFORMANCE:")
    print(f"  Total return:   {perf['total_return'] * 100:+.2f}%")
    print(f"  Current equity: ${perf['current_equity']:,.2f}")
    print()

    # Verify signal store has data
    recent = store.get_latest_signals(n=10)
    print(f"SIGNAL STORE: {len(recent)} recent signals recorded")
    targets = store.get_latest_targets()
    print(f"PORTFOLIO TARGETS: {len(targets)} models with active targets")
    print()

    print("=" * 60)
    print("  DEMO COMPLETE - All components working")
    print("=" * 60)


if __name__ == "__main__":
    main()
