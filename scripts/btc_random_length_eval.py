"""Random-length random-start window evaluation for BTC.

Tier-4 methodology extension. Existing eval scripts use a fixed 126-bar
(6-month) evaluation window for every backtest. This hides an obvious
question: does the H-Vol hybrid's Sharpe depend on the window length?

If Sharpe is ~constant across 60d–400d windows, the strategy is scale-
invariant and the fixed-6mo eval is a fair proxy. If Sharpe varies
systematically with window length, the 6mo eval is hiding structure.

This script:
  1. Samples N random windows of random lengths from [min_len, max_len]
  2. Runs the H-Vol hybrid baseline + optional vol-target overlay on each
  3. Persists per-window metrics (including real trade counts — no bug) to
     scripts/data/btc_random_length_eval.parquet
  4. Prints a summary binned by window-length bucket

This also acts as a stress test for the new vol-targeting overlay
(signals/backtest/vol_target.py): if vol-targeting helps at ANY
window length, it should show up here.
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
N_WINDOWS_PER_SEED = 12
MIN_WIN = 60    # ~3 months
MAX_WIN = 400   # ~16 months
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


def _voltarget_cfg(annual: float = 0.20, max_scale: float = 2.0) -> BacktestConfig:
    cfg = _baseline_cfg()
    cfg.vol_target_enabled = True
    cfg.vol_target_annual = annual
    cfg.vol_target_periods_per_year = 365
    cfg.vol_target_max_scale = max_scale
    return cfg


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars")

    configs = {
        "baseline": _baseline_cfg(),
        "vt20_cap2": _voltarget_cfg(annual=0.20, max_scale=2.0),
        "vt30_cap2": _voltarget_cfg(annual=0.30, max_scale=2.0),
        "vt40_cap2": _voltarget_cfg(annual=0.40, max_scale=2.0),
    }

    min_start = TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_end = len(prices) - 1

    rows: list[dict] = []
    t0 = time.time()
    total_runs = len(configs) * len(SEEDS) * N_WINDOWS_PER_SEED
    run_i = 0
    for name, cfg in configs.items():
        for seed in SEEDS:
            rng = random.Random(seed)
            for win_idx in range(N_WINDOWS_PER_SEED):
                length = rng.randint(MIN_WIN, MAX_WIN)
                max_start_i = max_end - length
                if max_start_i <= min_start:
                    continue
                start_i = rng.randint(min_start, max_start_i)
                end_i = start_i + length
                m = _run_on_window(cfg, prices, start_i, end_i)
                rows.append({
                    "config_name": name,
                    "seed": seed,
                    "window_idx": win_idx,
                    "window_length": length,
                    "window_start": str(prices.index[start_i].date()),
                    "window_end": str(prices.index[end_i - 1].date()),
                    "cagr": float(m.cagr),
                    "sharpe": float(m.sharpe),
                    "max_dd": float(m.max_drawdown),
                    "n_trades": int(m.n_trades),
                    "win_rate": float(m.win_rate),
                    "profit_factor": float(m.profit_factor),
                })
                run_i += 1
            elapsed = time.time() - t0
            sub = [r for r in rows if r["config_name"] == name and r["seed"] == seed]
            med_sh = float(np.median([r["sharpe"] for r in sub]))
            med_trades = float(np.median([r["n_trades"] for r in sub]))
            print(
                f"  [{run_i:3d}/{total_runs}] {name:<12} seed={seed:>4d}  "
                f"median Sh {med_sh:+5.2f}  med trades {med_trades:>4.0f}  "
                f"({elapsed:4.0f}s)"
            )

    df = pd.DataFrame(rows)
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "btc_random_length_eval.parquet"
    df.to_parquet(path, index=False)
    print(f"\nSaved: {path}  ({len(df)} rows)")

    print()
    print("=" * 100)
    print("Per-config aggregate stats (across all seeds and window lengths)")
    print("=" * 100)
    agg = df.groupby("config_name").agg(
        median_sharpe=("sharpe", "median"),
        mean_sharpe=("sharpe", "mean"),
        median_cagr=("cagr", "median"),
        median_mdd=("max_dd", "median"),
        median_trades=("n_trades", "median"),
        n_runs=("sharpe", "size"),
    )
    print(agg.to_string())

    print()
    print("=" * 100)
    print("Sharpe by window-length bucket (baseline only)")
    print("=" * 100)
    base = df[df["config_name"] == "baseline"].copy()
    base["bucket"] = pd.cut(
        base["window_length"],
        bins=[0, 100, 150, 200, 300, 500],
        labels=["60-100", "100-150", "150-200", "200-300", "300-400"],
    )
    bucket_agg = base.groupby("bucket", observed=True).agg(
        median_sharpe=("sharpe", "median"),
        mean_sharpe=("sharpe", "mean"),
        n_runs=("sharpe", "size"),
    )
    print(bucket_agg.to_string())


if __name__ == "__main__":
    main()
