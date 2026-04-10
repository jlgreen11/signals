"""2D sweep of (blend_low, blend_high) for H-Blend on BTC-USD.

Motivation (IMPROVEMENTS.md #16a): The H-Blend hybrid uses a linear ramp
between blend_low_quantile and blend_high_quantile of training vol. The
ad-hoc default is (0.50, 0.85) which produced median Sharpe 2.06 on the
16-window BTC random evaluation. The retuned H-Vol hard-switch at
q=0.70 scores 2.15. The question is whether a better (low, high) pair
beats that 2.15 number.

Grid: low in {0.30, 0.40, 0.50, 0.60} × high in {0.70, 0.80, 0.85, 0.90}.
Only pairs with low < high are evaluated. Total = 16 pairs, minus
invalids = ~14 valid configs × 16 windows = ~224 backtests. Cost ~5 min.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.config import SETTINGS
from signals.data.storage import DataStore


SYMBOL = "BTC-USD"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")

LOW_GRID = [0.30, 0.40, 0.50, 0.60]
HIGH_GRID = [0.70, 0.80, 0.85, 0.90]


@dataclass
class SweepRow:
    low: float
    high: float
    mean_sharpe: float
    median_sharpe: float
    mean_cagr: float
    median_cagr: float
    mean_max_dd: float
    positive_windows: int
    n_windows: int


def _run_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
) -> Metrics:
    warmup_pad = 5
    slice_start = start_i - cfg.train_window - cfg.vol_window - warmup_pad
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]

    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=symbol)
    except Exception as e:
        print(f"  engine error: {e}")
        return compute_metrics(pd.Series(dtype=float), [])

    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(eq_rebased, [])


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars ({prices.index[0].date()} → {prices.index[-1].date()})")

    vol_window = 10
    homc_train_window = 1000
    warmup_pad = 5
    six_months = 126
    n_windows = 16
    seed = 42

    min_start = homc_train_window + vol_window + warmup_pad
    max_start = len(prices) - six_months - 1
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(min_start, max_start), n_windows))

    def _cfg(low: float, high: float) -> BacktestConfig:
        return BacktestConfig(
            model_type="hybrid",
            train_window=homc_train_window,
            retrain_freq=21,
            n_states=5,
            order=5,
            return_bins=3,
            volatility_bins=3,
            vol_window=vol_window,
            laplace_alpha=0.01,
            hybrid_routing_strategy="blend",
            hybrid_blend_low=low,
            hybrid_blend_high=high,
        )

    valid_pairs = [(lo, hi) for lo in LOW_GRID for hi in HIGH_GRID if lo < hi]
    total = len(valid_pairs)
    results: list[SweepRow] = []
    for idx, (low, high) in enumerate(valid_pairs, start=1):
        cfg = _cfg(low, high)
        sharpes: list[float] = []
        cagrs: list[float] = []
        mdds: list[float] = []
        for start_i in starts:
            end_i = start_i + six_months
            m = _run_on_window(cfg, prices, start_i, end_i, SYMBOL)
            sharpes.append(m.sharpe)
            cagrs.append(m.cagr)
            mdds.append(m.max_drawdown)
        row = SweepRow(
            low=low,
            high=high,
            mean_sharpe=float(np.mean(sharpes)),
            median_sharpe=float(np.median(sharpes)),
            mean_cagr=float(np.mean(cagrs)),
            median_cagr=float(np.median(cagrs)),
            mean_max_dd=float(np.mean(mdds)),
            positive_windows=int(sum(1 for c in cagrs if c > 0)),
            n_windows=n_windows,
        )
        results.append(row)
        print(
            f"  [{idx:2d}/{total}] low={low:.2f} high={high:.2f}  "
            f"mean_Sh={row.mean_sharpe:5.2f}  "
            f"median_Sh={row.median_sharpe:5.2f}  "
            f"median_CAGR={row.median_cagr * 100:+7.1f}%  "
            f"mean_MDD={row.mean_max_dd * 100:+6.1f}%  "
            f"pos={row.positive_windows}/{n_windows}"
        )

    print()
    print("=" * 100)
    print(f"H-Blend ramp sweep — {SYMBOL} — ranked by median Sharpe")
    print("=" * 100)
    results_sorted = sorted(results, key=lambda r: r.median_sharpe, reverse=True)
    print(
        f"{'low':>6} {'high':>6} "
        f"{'mean Sh':>8} {'median Sh':>10} "
        f"{'median CAGR':>13} {'mean MDD':>10} {'pos/N':>8}"
    )
    for r in results_sorted:
        print(
            f"{r.low:>6.2f} {r.high:>6.2f} "
            f"{r.mean_sharpe:>8.2f} {r.median_sharpe:>10.2f} "
            f"{r.median_cagr * 100:>12.1f}% "
            f"{r.mean_max_dd * 100:>9.1f}% "
            f"{r.positive_windows:>3}/{r.n_windows}"
        )

    # Reference points
    HVOL_070_MEDIAN = 2.15
    HBLEND_DEFAULT_MEDIAN = 2.06
    best = results_sorted[0]
    print()
    print(f"Reference: H-Vol @ q=0.70 median Sharpe = {HVOL_070_MEDIAN:.2f}")
    print(f"Reference: H-Blend default (0.50, 0.85) median Sharpe = {HBLEND_DEFAULT_MEDIAN:.2f}")
    print(
        f"Best blend pair: low={best.low:.2f}, high={best.high:.2f}, "
        f"median Sharpe = {best.median_sharpe:.2f}"
    )
    if best.median_sharpe > HVOL_070_MEDIAN:
        print(
            f"  → Blend BEATS H-Vol @ q=0.70 by "
            f"{best.median_sharpe - HVOL_070_MEDIAN:+.2f} Sharpe. "
            f"Consider making H-Blend (low={best.low:.2f}, high={best.high:.2f}) "
            f"the new default."
        )
    else:
        print(
            f"  → Blend does NOT beat H-Vol @ q=0.70 "
            f"({best.median_sharpe:.2f} vs {HVOL_070_MEDIAN:.2f}). "
            f"Hard-switch at q=0.70 remains the BTC default."
        )


if __name__ == "__main__":
    main()
