"""Sweep (target_scale_bps, max_long) pairs for H-Vol @ q=0.70 on BTC-USD.

Motivation (Tier 1 #5): the 16-window random evaluation showed the
strategy gets the *direction* right in winning windows but under-sizes.
2019-01-11 → 2019-05-16 is the canonical example — composite's Sharpe
3.54 vs B&H's 3.50, but CAGR +332% vs +822%: same direction, half the
size. The fix is more aggressive sizing — lower target_scale_bps
(bigger magnitude per bp of expected return) and/or higher max_long
(allow leveraged long positions).

This sweep tests a grid of (scale, max_long) combinations against the
BTC random-window eval and ranks by median Sharpe. The winner goes to
IMPROVEMENTS.md for consideration as a new default.

NOTE: The SignalGenerator formula was fixed in this same session to let
max_long values above 1.0 actually take effect. Before the fix, the
magnitude was capped at 1.0 before clipping, so max_long > 1.0 had no
effect. Now magnitude = |expected| / scale (uncapped) and max_long is
the real leverage ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "BTC-USD"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")

# Grid: scale × max_long. Baseline is scale=20, max_long=1.0.
SCALE_GRID = [5, 10, 15, 20]
MAX_LONG_GRID = [1.0, 1.25, 1.5, 2.0]


@dataclass
class SweepRow:
    target_scale_bps: float
    max_long: float
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
    return compute_metrics(
        eq_rebased,
        [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=365.0,
    )


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
    starts = draw_non_overlapping_starts(
        seed=seed,
        min_start=min_start,
        max_start=max_start,
        window_len=six_months,
        n_windows=n_windows,
    )

    def _cfg(scale: float, max_long: float) -> BacktestConfig:
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
            target_scale_bps=scale,
            max_long=max_long,
            hybrid_routing_strategy="vol",
            hybrid_vol_quantile=0.70,
        )

    results: list[SweepRow] = []
    total = len(SCALE_GRID) * len(MAX_LONG_GRID)
    idx = 0
    for scale in SCALE_GRID:
        for max_long in MAX_LONG_GRID:
            idx += 1
            cfg = _cfg(scale, max_long)
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
                target_scale_bps=scale,
                max_long=max_long,
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
                f"  [{idx:2d}/{total}] scale={scale:2d} max_long={max_long:.2f}  "
                f"mean_Sh={row.mean_sharpe:5.2f}  "
                f"median_Sh={row.median_sharpe:5.2f}  "
                f"median_CAGR={row.median_cagr * 100:+7.1f}%  "
                f"mean_MDD={row.mean_max_dd * 100:+6.1f}%  "
                f"pos={row.positive_windows}/{n_windows}"
            )

    print()
    print("=" * 100)
    print(f"Sizing sweep — {SYMBOL} — H-Vol @ q=0.70 — ranked by median Sharpe")
    print("=" * 100)
    results_sorted = sorted(results, key=lambda r: r.median_sharpe, reverse=True)
    print(
        f"{'scale':>6} {'max_long':>9} "
        f"{'mean Sh':>8} {'median Sh':>10} "
        f"{'median CAGR':>13} {'mean MDD':>10} {'pos/N':>8}"
    )
    for r in results_sorted:
        print(
            f"{r.target_scale_bps:>6.0f} {r.max_long:>9.2f} "
            f"{r.mean_sharpe:>8.2f} {r.median_sharpe:>10.2f} "
            f"{r.median_cagr * 100:>12.1f}% "
            f"{r.mean_max_dd * 100:>9.1f}% "
            f"{r.positive_windows:>3}/{r.n_windows}"
        )

    baseline = [r for r in results if r.target_scale_bps == 20 and r.max_long == 1.0][0]
    best = results_sorted[0]
    print()
    print(f"Baseline (scale=20, max_long=1.0): median Sharpe {baseline.median_sharpe:.2f}, "
          f"CAGR {baseline.median_cagr * 100:+.1f}%, MDD {baseline.mean_max_dd * 100:+.1f}%")
    print(f"Best (scale={best.target_scale_bps:.0f}, max_long={best.max_long:.2f}): "
          f"median Sharpe {best.median_sharpe:.2f}, CAGR {best.median_cagr * 100:+.1f}%, "
          f"MDD {best.mean_max_dd * 100:+.1f}%")
    print(
        f"Improvement over baseline: "
        f"{best.median_sharpe - baseline.median_sharpe:+.2f} Sharpe, "
        f"{(best.median_cagr - baseline.median_cagr) * 100:+.1f}pp CAGR, "
        f"{(best.mean_max_dd - baseline.mean_max_dd) * 100:+.1f}pp MDD"
    )


if __name__ == "__main__":
    main()
