"""Evaluate EnsembleModel on BTC 16 random 6-month windows at 4 seeds.

The ensemble combines composite + HOMC + boost with equal weights.
Compares against the H-Vol @ q=0.70 production baseline.
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
        train_window=1000,
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


def _ensemble_cfg() -> BacktestConfig:
    return BacktestConfig(
        model_type="ensemble",
        train_window=1000,
        retrain_freq=21,
        n_states=5,
        order=5,
        return_bins=3,
        volatility_bins=3,
        vol_window=VOL_WINDOW,
        laplace_alpha=0.01,
    )


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars")

    configs = {
        "baseline": _baseline_cfg(),
        "ensemble": _ensemble_cfg(),
    }

    min_start = 1000 + VOL_WINDOW + WARMUP_PAD
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
            elapsed = time.time() - t0
            print(f"  {name:<10} seed={seed:>4d}  median Sh {float(np.median(sharpes)):+5.2f}  ({elapsed:4.0f}s)")

    df = pd.DataFrame(rows)
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "btc_ensemble_eval.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved: {path}")

    print()
    print(f"{'config':<12}{'seed 42':>10} {'seed 7':>10} {'seed 100':>10} {'seed 999':>10} {'avg':>8}")
    for name in configs:
        sub = df[df["config_name"] == name]
        medians = sub.groupby("seed")["sharpe"].median().to_dict()
        row_vals = [medians.get(s, float("nan")) for s in SEEDS]
        avg = float(np.mean(row_vals))
        print(
            f"{name:<12}"
            f"{row_vals[0]:>10.2f} {row_vals[1]:>10.2f} "
            f"{row_vals[2]:>10.2f} {row_vals[3]:>10.2f} {avg:>8.2f}"
        )


if __name__ == "__main__":
    main()
