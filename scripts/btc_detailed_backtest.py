"""Detailed single-window BTC backtest + Excel export.

Picks ONE random start date and ONE random window length, runs the
production H-Vol @ q=0.70 hybrid baseline on it, and exports a full
daily-activity Excel.

This is a user-facing deliverable: one workbook showing every day's
price, signal, action, position, cash, equity, and running buy/sell
counts for the selected window.

Usage
-----
  python scripts/btc_detailed_backtest.py
  python scripts/btc_detailed_backtest.py --seed 2026 --min-len 150 --max-len 300
  python scripts/btc_detailed_backtest.py --seed 2026 --vol-target 0.30

Defaults pick a random seed driven by the wall clock so the "random"
window is actually random on each run. Pass `--seed` for reproducibility.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.excel_report import write_excel_report
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "BTC-USD"
DATA_START = pd.Timestamp("2015-01-01", tz="UTC")
DATA_END = pd.Timestamp("2024-12-31", tz="UTC")
TRAIN_WINDOW = 1000
VOL_WINDOW = 10
WARMUP_PAD = 5


def _build_config(vol_target_annual: float | None) -> BacktestConfig:
    cfg = BacktestConfig(
        model_type="hybrid",
        train_window=TRAIN_WINDOW,
        retrain_freq=21,
        n_states=5,
        order=5,
        return_bins=3,
        volatility_bins=3,
        vol_window=VOL_WINDOW,
        laplace_alpha=0.01,
        hybrid_routing_strategy="vol",
        hybrid_vol_quantile=0.70,
    )
    if vol_target_annual is not None:
        cfg.vol_target_enabled = True
        cfg.vol_target_annual = vol_target_annual
        cfg.vol_target_periods_per_year = 365
        cfg.vol_target_max_scale = 2.0
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (defaults to wall-clock nanoseconds).",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=120,
        help="Minimum window length in bars (default 120, ~6 months).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=360,
        help="Maximum window length in bars (default 360, ~18 months).",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=None,
        help="Optional annualized vol target for the overlay (e.g., 0.25). Off by default.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .xlsx path (defaults to scripts/data/btc_detailed_<timestamp>.xlsx).",
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time_ns() % (2**31))
    rng = random.Random(seed)

    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= DATA_START) & (prices.index <= DATA_END)]
    n = len(prices)
    print(f"{SYMBOL}: {n} bars loaded ({prices.index[0].date()} → {prices.index[-1].date()})")

    # Choose random window length and random start.
    length = rng.randint(args.min_len, args.max_len)
    min_start = TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = n - length - 1
    if max_start <= min_start:
        raise SystemExit(f"not enough data: need at least {min_start + length + 1} bars, have {n}")
    start_i = rng.randint(min_start, max_start)
    end_i = start_i + length

    print(f"Seed: {seed}")
    print(f"Random window: length={length} bars ({length/30:.1f} months)")
    print(f"Random start:  bar {start_i}  ({prices.index[start_i].date()})")
    print(f"Random end:    bar {end_i-1}  ({prices.index[end_i-1].date()})")

    cfg = _build_config(args.vol_target)
    if cfg.vol_target_enabled:
        print(f"Vol-target overlay: ON, annual={cfg.vol_target_annual:.0%}, max_scale={cfg.vol_target_max_scale}")
    else:
        print("Vol-target overlay: OFF (pure baseline)")

    # Engine input: include warmup history before the eval window
    slice_start = start_i - TRAIN_WINDOW - VOL_WINDOW - WARMUP_PAD
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i].copy()
    eval_start_ts = prices.index[start_i]

    print(f"Engine input: {len(engine_input)} bars (including warmup)")
    t0 = time.time()
    result = BacktestEngine(cfg).run(engine_input, symbol=SYMBOL)
    print(f"Backtest complete in {time.time()-t0:.1f}s")

    # Clip the result's equity curve and signals to the eval window only —
    # the Excel should show what happened during the random window the
    # script selected, not the warmup period.
    eq_mask = result.equity_curve.index >= eval_start_ts
    eval_equity = result.equity_curve.loc[eq_mask]
    eval_signals = (
        result.signals.loc[result.signals.index >= eval_start_ts]
        if not result.signals.empty
        else result.signals
    )
    eval_trades = [t for t in result.trades if t.ts >= eval_start_ts]

    # Do NOT rebase: the exporter replays the full trade history to
    # establish an inherited cash/position state at the first eval bar.
    # Rebasing the equity curve independently would break that invariant.
    # Sharpe / CAGR / MDD are invariant to scaling so metrics still reflect
    # the eval window correctly.
    rebased = eval_equity

    # Build a trimmed BacktestResult for the exporter.
    from signals.backtest.engine import BacktestResult
    from signals.backtest.metrics import compute_metrics

    # BTC trades 365 days/year — pass explicit periods_per_year to
    # avoid the legacy index-inference defaulting to 252 for daily bars.
    # See SKEPTIC_REVIEW.md § 8a / Round-5 audit.
    new_metrics = compute_metrics(
        rebased,
        eval_trades,
        risk_free_rate=cfg.risk_free_rate,
        periods_per_year=365.0,
    )
    # Benchmark = B&H on the eval window, same initial cash
    bench_window = engine_input.loc[engine_input.index >= eval_start_ts, "close"]
    if len(bench_window) > 0:
        bench = (bench_window / float(bench_window.iloc[0])) * cfg.initial_cash
    else:
        bench = pd.Series(dtype=float)
    new_bench_metrics = compute_metrics(
        bench,
        [],
        risk_free_rate=cfg.risk_free_rate,
        periods_per_year=365.0,
    )

    # Pass the FULL trade list (not eval_trades) to the trimmed result so
    # the Excel exporter can replay warmup trades and establish the
    # correct inherited position/cash state at the first eval bar. The
    # exporter clips the Trades sheet to the activity window automatically.
    trimmed = BacktestResult(
        config=cfg,
        symbol=SYMBOL,
        start=rebased.index[0] if len(rebased) > 0 else result.start,
        end=rebased.index[-1] if len(rebased) > 0 else result.end,
        equity_curve=rebased,
        benchmark_curve=bench,
        signals=eval_signals,
        metrics=new_metrics,
        benchmark_metrics=new_bench_metrics,
        trades=result.trades,
    )

    # Output path
    if args.out is None:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out = Path(__file__).parent / "data" / f"btc_detailed_{stamp}.xlsx"
    else:
        out = args.out

    extra = {
        "random_seed": seed,
        "window_length_bars": length,
        "window_start_bar": start_i,
        "eval_start_ts": str(eval_start_ts.date()),
        "vol_target_enabled": cfg.vol_target_enabled,
    }
    paths = write_excel_report(
        trimmed,
        prices=engine_input[["open", "high", "low", "close", "volume"]],
        out_path=out,
        symbol=SYMBOL,
        extra_summary=extra,
    )
    print()
    print(f"Excel written: {paths.xlsx_path}")
    print(f"  Summary rows : {paths.summary_rows}")
    print(f"  Activity rows: {paths.activity_rows}")
    print(f"  Trade rows   : {paths.trade_rows}")
    print()
    print("Key metrics:")
    print(f"  Sharpe           : {new_metrics.sharpe:+.3f}")
    print(f"  CAGR             : {new_metrics.cagr*100:+.2f}%")
    print(f"  Max drawdown     : {new_metrics.max_drawdown*100:+.2f}%")
    print(f"  Final equity     : ${float(rebased.iloc[-1]) if len(rebased) else 0:,.2f}")
    print(f"  Round-trip trades: {new_metrics.n_trades}")
    print(f"  Benchmark Sharpe : {new_bench_metrics.sharpe:+.3f}")


if __name__ == "__main__":
    main()
