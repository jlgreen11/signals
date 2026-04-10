"""Sweep the H-Vol hybrid's vol_quantile_threshold across a grid of values
for both BTC-USD and ^GSPC, and report the per-asset optimum.

The current H-Vol default is 0.75 (route the top 25% of training-vol days
to composite, everything else to HOMC). That default is ad hoc — it was
picked for "obvious economic interpretation" rather than empirical
optimization. This sweep runs the full 16-window random evaluation at
each candidate threshold on each asset and ranks the results by median
Sharpe.

The random-window evaluation (as opposed to a single holdout) is the
right validation methodology here because:

  - Single-holdout results are period-cherry-picked (proven by the
    Tier-0b experiment where BTC 20% vs 30% holdout flipped HOMC's
    verdict)
  - Random windows sample across bull, bear, and chop regimes
  - Median (not mean) Sharpe is the right summary because mean is
    dominated by the 2020-11 bull outlier and misrepresents the
    typical-window behavior
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


SYMBOLS = [
    ("BTC-USD", pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")),
    ("^GSPC",   pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")),
]

QUANTILE_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]


@dataclass
class SweepRow:
    symbol: str
    quantile: float
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
        print(f"  engine error at window {eval_start_ts.date()}: {e}")
        return compute_metrics(pd.Series(dtype=float), [])

    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(eq_rebased, [])


def _sweep_symbol(
    store: DataStore,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[SweepRow]:
    prices = store.load(symbol, "1d").sort_index()
    prices = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)]
    if prices.empty:
        raise ValueError(f"No data for {symbol}")

    print(f"\n{symbol}: {len(prices)} bars ({prices.index[0].date()} → {prices.index[-1].date()})")

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

    rows: list[SweepRow] = []
    for q in QUANTILE_GRID:
        cfg = BacktestConfig(
            model_type="hybrid",
            train_window=homc_train_window,
            retrain_freq=21,
            n_states=5,
            order=5,
            return_bins=3,
            volatility_bins=3,
            vol_window=vol_window,
            laplace_alpha=0.01,
            hybrid_routing_strategy="vol",
            hybrid_vol_quantile=q,
        )
        sharpes: list[float] = []
        cagrs: list[float] = []
        mdds: list[float] = []
        for start_i in starts:
            end_i = start_i + six_months
            m = _run_on_window(cfg, prices, start_i, end_i, symbol)
            sharpes.append(m.sharpe)
            cagrs.append(m.cagr)
            mdds.append(m.max_drawdown)

        rows.append(
            SweepRow(
                symbol=symbol,
                quantile=q,
                mean_sharpe=float(np.mean(sharpes)),
                median_sharpe=float(np.median(sharpes)),
                mean_cagr=float(np.mean(cagrs)),
                median_cagr=float(np.median(cagrs)),
                mean_max_dd=float(np.mean(mdds)),
                positive_windows=int(sum(1 for c in cagrs if c > 0)),
                n_windows=n_windows,
            )
        )
        print(
            f"  q={q:.2f}  mean_Sh={rows[-1].mean_sharpe:5.2f}  "
            f"median_Sh={rows[-1].median_sharpe:5.2f}  "
            f"median_CAGR={rows[-1].median_cagr * 100:+6.1f}%  "
            f"mean_MDD={rows[-1].mean_max_dd * 100:+6.1f}%  "
            f"pos={rows[-1].positive_windows}/{n_windows}"
        )

    return rows


def main() -> None:
    store = DataStore(SETTINGS.data.dir)

    all_rows: list[SweepRow] = []
    for symbol, start_ts, end_ts in SYMBOLS:
        rows = _sweep_symbol(store, symbol, start_ts, end_ts)
        all_rows.extend(rows)

    print()
    print("=" * 100)
    print("Vol quantile sweep — per-asset results")
    print("=" * 100)
    print(
        f"{'symbol':<10} {'q':>5} {'mean Sh':>8} "
        f"{'median Sh':>10} {'median CAGR':>12} {'mean MDD':>10} {'pos/N':>8}"
    )
    for r in all_rows:
        print(
            f"{r.symbol:<10} {r.quantile:>5.2f} {r.mean_sharpe:>8.2f} "
            f"{r.median_sharpe:>10.2f} "
            f"{r.median_cagr * 100:>11.1f}% "
            f"{r.mean_max_dd * 100:>9.1f}% "
            f"{r.positive_windows:>3}/{r.n_windows}"
        )

    print()
    print("=" * 100)
    print("Best quantile per asset (by median Sharpe)")
    print("=" * 100)
    for symbol, _, _ in SYMBOLS:
        sym_rows = [r for r in all_rows if r.symbol == symbol]
        best = max(sym_rows, key=lambda r: r.median_sharpe)
        current_default = [r for r in sym_rows if abs(r.quantile - 0.75) < 1e-9][0]
        print(
            f"{symbol:<10} best q={best.quantile:.2f}  "
            f"median Sh={best.median_sharpe:.2f}  "
            f"(current default q=0.75: median Sh={current_default.median_sharpe:.2f})"
        )
        delta = best.median_sharpe - current_default.median_sharpe
        if best.quantile != 0.75:
            print(
                f"           → improvement of {delta:+.2f} Sharpe by "
                f"{'increasing' if best.quantile > 0.75 else 'decreasing'} threshold"
            )


if __name__ == "__main__":
    main()
