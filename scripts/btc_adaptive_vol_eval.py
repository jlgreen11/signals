"""Evaluate the adaptive_vol hybrid routing strategy on BTC-USD.

Compares the adaptive strategy against the production H-Vol @ q=0.70
baseline on 16 random 6-month windows at 4 seeds for robustness.

Saves per-window raw results to
scripts/data/btc_adaptive_vol_eval.parquet.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "BTC-USD"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")
SEEDS = [42, 7, 100, 999]
N_WINDOWS = 16
SIX_MONTHS = 126
TRAIN_WINDOW = 1000
VOL_WINDOW = 10
WARMUP_PAD = 5


def _run_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
) -> Metrics:
    slice_start = start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]
    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=SYMBOL)
    except Exception as e:
        print(f"    engine error: {e}")
        return compute_metrics(pd.Series(dtype=float), [])
    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    eval_trades = [t for t in result.trades if t.ts >= eval_start_ts]
    return compute_metrics(eq_rebased, eval_trades)


def _baseline_cfg() -> BacktestConfig:
    return BacktestConfig(
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


def _adaptive_cfg(low: float, high: float, lookback: int) -> BacktestConfig:
    return BacktestConfig(
        model_type="hybrid",
        train_window=TRAIN_WINDOW,
        retrain_freq=21,
        n_states=5,
        order=5,
        return_bins=3,
        volatility_bins=3,
        vol_window=VOL_WINDOW,
        laplace_alpha=0.01,
        hybrid_routing_strategy="adaptive_vol",
        hybrid_adaptive_low=low,
        hybrid_adaptive_high=high,
        hybrid_adaptive_lookback=lookback,
    )


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars")

    # Test a few adaptive (low, high) pairs plus the baseline
    configs = {
        "baseline": _baseline_cfg(),
        "adapt_60_80_30": _adaptive_cfg(0.60, 0.80, 30),
        "adapt_50_90_30": _adaptive_cfg(0.50, 0.90, 30),
        "adapt_65_75_30": _adaptive_cfg(0.65, 0.75, 30),
        "adapt_60_80_60": _adaptive_cfg(0.60, 0.80, 60),
        "adapt_60_80_14": _adaptive_cfg(0.60, 0.80, 14),
    }

    min_start = TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = len(prices) - SIX_MONTHS - 1

    rows: list[dict] = []
    t0 = time.time()
    for name, cfg in configs.items():
        for seed in SEEDS:
            rng = random.Random(seed)
            starts = sorted(rng.sample(range(min_start, max_start), N_WINDOWS))
            sharpes: list[float] = []
            for win_idx, start_i in enumerate(starts):
                end_i = start_i + SIX_MONTHS
                m = _run_on_window(cfg, prices, start_i, end_i)
                rows.append({
                    "config_name": name,
                    "seed": seed,
                    "window_idx": win_idx,
                    "window_start": str(prices.index[start_i].date()),
                    "window_end": str(prices.index[end_i - 1].date()),
                    "cagr": float(m.cagr),
                    "sharpe": float(m.sharpe),
                    "max_dd": float(m.max_drawdown),
                    "n_trades": int(m.n_trades),
                })
                sharpes.append(float(m.sharpe))
            median_sh = float(np.median(sharpes))
            elapsed = time.time() - t0
            print(
                f"  {name:<18} seed={seed:>4d}  median Sh {median_sh:+5.2f}  "
                f"({elapsed:4.0f}s)"
            )

    df = pd.DataFrame(rows)
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "btc_adaptive_vol_eval.parquet"
    df.to_parquet(path, index=False)
    print(f"\nSaved: {path}")

    # Cross-seed summary
    print()
    print("=" * 100)
    print("Median Sharpe per config per seed")
    print("=" * 100)
    print(f"{'config':<20} {'seed 42':>8} {'seed 7':>8} {'seed 100':>9} {'seed 999':>9} {'avg':>6}")
    for name in configs:
        sub = df[df["config_name"] == name]
        medians = sub.groupby("seed")["sharpe"].median().to_dict()
        row_vals = [medians.get(s, float("nan")) for s in SEEDS]
        avg = np.mean(row_vals) if not any(np.isnan(row_vals)) else float("nan")
        print(
            f"{name:<20} "
            f"{row_vals[0]:>8.2f} {row_vals[1]:>8.2f} "
            f"{row_vals[2]:>9.2f} {row_vals[3]:>9.2f} {avg:>6.2f}"
        )

    # Identify any config that beats baseline at every seed
    baseline_medians = {
        int(s): float(v)
        for s, v in df[df["config_name"] == "baseline"]
        .groupby("seed")["sharpe"]
        .median()
        .items()
    }
    print()
    print("Robust winners (beat baseline at every seed):")
    found = False
    for name in configs:
        if name == "baseline":
            continue
        sub_med = {
            int(s): float(v)
            for s, v in df[df["config_name"] == name]
            .groupby("seed")["sharpe"]
            .median()
            .items()
        }
        if all(sub_med.get(s, -np.inf) >= baseline_medians.get(s, np.inf) for s in SEEDS):
            avg_lift = np.mean([sub_med[s] - baseline_medians[s] for s in SEEDS])
            print(f"  ✓ {name}  avg lift {avg_lift:+.3f}")
            found = True
    if not found:
        print("  ✗ None — all adaptive configs failed robustness")


if __name__ == "__main__":
    main()
