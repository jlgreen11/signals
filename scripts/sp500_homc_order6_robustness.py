"""Robustness check for HOMC@order=6 on S&P 500.

The main HOMC order sweep (scripts/sp500_homc_order_sweep.py) showed
order=6 with median Sharpe 0.90, beating buy & hold's 0.77 by +0.13
Sharpe. Before promoting this as a real S&P default, I need to check
whether the result is robust or a data-mining artifact.

This script re-runs HOMC@order=6 on S&P with 4 different random seeds
{42, 7, 100, 999}. If order=6 genuinely captures S&P structure, the
median Sharpe should stay above B&H's 0.77 across all seeds. If the
Tier-0g result was a lucky seed, at least one other seed should show
a collapse.

Also includes order=5 and order=4 as controls at each seed — if order=6
is uniquely good while 5 and 4 are mediocre across seeds, the order=6
edge is more credible. If the "best" order changes with the seed, all
three are noise-equivalent.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "^GSPC"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")
SEEDS = [42, 7, 100, 999]
ORDERS = [4, 5, 6]
N_WINDOWS = 16
TRAIN_WINDOW = 1000
VOL_WINDOW = 10
SIX_MONTHS = 126


def _run_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
) -> Metrics:
    warmup_pad = 5
    slice_start = start_i - cfg.train_window - cfg.vol_window - warmup_pad
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]
    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=SYMBOL)
    except Exception:
        return compute_metrics(pd.Series(dtype=float), [])
    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(eq_rebased, [])


def _eval_at_seed(
    prices: pd.DataFrame,
    seed: int,
    order: int,
) -> tuple[float, float, float, int]:
    """Return (mean_sharpe, median_sharpe, median_cagr, positive_count)."""
    warmup_pad = 5
    min_start = TRAIN_WINDOW + VOL_WINDOW + warmup_pad
    max_start = len(prices) - SIX_MONTHS - 1
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(min_start, max_start), N_WINDOWS))

    cfg = BacktestConfig(
        model_type="homc",
        train_window=TRAIN_WINDOW,
        retrain_freq=21,
        n_states=5,
        order=order,
        vol_window=VOL_WINDOW,
        laplace_alpha=1.0,
    )
    sharpes: list[float] = []
    cagrs: list[float] = []
    for start_i in starts:
        end_i = start_i + SIX_MONTHS
        m = _run_on_window(cfg, prices, start_i, end_i)
        sharpes.append(m.sharpe)
        cagrs.append(m.cagr)

    return (
        float(np.mean(sharpes)),
        float(np.median(sharpes)),
        float(np.median(cagrs)),
        int(sum(1 for c in cagrs if c > 0)),
    )


def _bh_at_seed(prices: pd.DataFrame, seed: int) -> tuple[float, float]:
    warmup_pad = 5
    min_start = TRAIN_WINDOW + VOL_WINDOW + warmup_pad
    max_start = len(prices) - SIX_MONTHS - 1
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(min_start, max_start), N_WINDOWS))

    sharpes: list[float] = []
    cagrs: list[float] = []
    for start_i in starts:
        end_i = start_i + SIX_MONTHS
        eval_window = prices.iloc[start_i:end_i]
        bh_eq = (eval_window["close"] / eval_window["close"].iloc[0]) * 10_000.0
        m = compute_metrics(bh_eq, [])
        sharpes.append(m.sharpe)
        cagrs.append(m.cagr)
    return float(np.median(sharpes)), float(np.median(cagrs))


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars")
    print()
    print("Robustness check: HOMC orders 4/5/6 across 4 seeds")
    print("=" * 100)
    print(f"{'seed':>6} {'order':>6} {'mean Sh':>9} {'median Sh':>11} {'median CAGR':>13} {'pos/N':>8}  {'B&H median Sh':>15}")
    print("-" * 100)

    for seed in SEEDS:
        bh_median_sh, bh_median_cagr = _bh_at_seed(prices, seed)
        for order in ORDERS:
            mean_sh, median_sh, median_cagr, pos = _eval_at_seed(prices, seed, order)
            beats = "✓" if median_sh > bh_median_sh else " "
            print(
                f"{seed:>6d} {order:>6d} {mean_sh:>9.2f} {median_sh:>11.2f} "
                f"{median_cagr * 100:>12.1f}% {pos:>3d}/{N_WINDOWS}  "
                f"{bh_median_sh:>13.2f} {beats}"
            )
        print()

    print("=" * 100)
    print("Verdict")
    print("=" * 100)
    print(
        "A genuine order=6 edge should show median Sharpe > B&H at every seed.\n"
        "If order=6 is inconsistent (some seeds above, some below), it's noise.\n"
        "If order=6 is consistently above while 4/5 are mixed, order=6 is real."
    )


if __name__ == "__main__":
    main()
