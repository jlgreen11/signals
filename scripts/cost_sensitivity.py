"""Tier B2/B3 of SKEPTIC_REVIEW.md — transaction-cost and deadband sensitivity.

The headline BTC result (H-Vol @ q=0.70, Sharpe ~2.15 median on 16 random
6-month windows) was generated at commission_bps=5.0 + slippage_bps=5.0
(10 bps round-trip) with min_trade_fraction=0.20. The critique is two-fold:

  B2. No sensitivity analysis for transaction costs. A strategy whose edge
      evaporates at 15 bps commission is a very different beast from one
      that tolerates 25 bps.

  B3. The min_trade_fraction=0.20 deadband is entangled with the cost
      assumption. A wider deadband cushions a strategy against higher costs
      (fewer rebalances), so you can't interpret one without sweeping the
      other.

This script addresses both by running two grids against the existing
16-random-window BTC evaluation at seed 42, with H-Vol @ q=0.70 as the
sole strategy (the Tier-0e default):

  Grid 1 (2-D, 5×5 = 25 configs):
    commission_bps    ∈ {2.5, 5.0, 10.0, 15.0, 25.0}
    min_trade_fraction ∈ {0.05, 0.10, 0.15, 0.20, 0.30}

  Grid 2 (1-D, 5 configs):
    slippage_bps ∈ {2.5, 5.0, 10.0, 15.0, 25.0}
    (with commission_bps=5.0, min_trade_fraction=0.20 held fixed)

Output:
  - 5×5 median-Sharpe table for Grid 1
  - 1×5 median-Sharpe table for Grid 2
  - "Cost ridge" identification: which configs drop Sharpe by more than 0.5
    from the baseline (commission=5, deadband=0.20, slippage=5)
  - Parquet dump at scripts/data/cost_sensitivity.parquet with all per-window
    results so follow-up analyses don't need to re-run the compute.

Runnable as:  python scripts/cost_sensitivity.py

NOTE: this is compute-heavy — 25 grid cells × 16 windows + 5 slip cells × 16
windows = 480 H-Vol backtests. Only seed 42 is used, because the goal is
cost sensitivity, not seed robustness (covered elsewhere).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.config import SETTINGS
from signals.data.storage import DataStore

# Match random_window_eval.py exactly so results are directly comparable.
SYMBOL = "BTC-USD"
START_TS = pd.Timestamp("2015-01-01", tz="UTC")
END_TS = pd.Timestamp("2024-12-31", tz="UTC")
SEED = 42
N_WINDOWS = 16
SIX_MONTHS = 126
VOL_WINDOW = 10
HOMC_TRAIN_WINDOW = 1000
WARMUP_PAD = 5

# Grid 1: 2-D sweep over commission_bps × min_trade_fraction.
COMMISSION_GRID = [2.5, 5.0, 10.0, 15.0, 25.0]
DEADBAND_GRID = [0.05, 0.10, 0.15, 0.20, 0.30]

# Grid 2: 1-D sweep over slippage_bps (other costs held at baseline).
SLIPPAGE_GRID = [2.5, 5.0, 10.0, 15.0, 25.0]

# Baseline config for the "drop by more than 0.5 Sharpe" cost-ridge check.
BASELINE_COMMISSION = 5.0
BASELINE_DEADBAND = 0.20
BASELINE_SLIPPAGE = 5.0
COST_RIDGE_DROP = 0.5


@dataclass
class WindowRow:
    grid: str                     # "commission_deadband" or "slippage"
    commission_bps: float
    slippage_bps: float
    min_trade_fraction: float
    window_idx: int
    start: pd.Timestamp
    end: pd.Timestamp
    sharpe: float
    cagr: float
    max_drawdown: float


def _hvol_config(
    *,
    commission_bps: float,
    slippage_bps: float,
    min_trade_fraction: float,
) -> BacktestConfig:
    """H-Vol @ q=0.70 with the requested cost/deadband knobs.

    Everything else matches the Tier-0e BTC default exactly (see
    random_window_eval.py `_build_strategies`).
    """
    return BacktestConfig(
        model_type="hybrid",
        train_window=HOMC_TRAIN_WINDOW,
        retrain_freq=21,
        n_states=5,
        order=5,
        return_bins=3,
        volatility_bins=3,
        vol_window=VOL_WINDOW,
        laplace_alpha=0.01,
        hybrid_routing_strategy="vol",
        hybrid_vol_quantile=0.70,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        min_trade_fraction=min_trade_fraction,
    )


def _run_one_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
) -> Metrics:
    """Mirror of random_window_eval._run_strategy_on_window — same slicing
    logic so the numbers are directly comparable to the published results."""
    slice_start = start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]

    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=symbol)
    except Exception as exc:
        print(f"    engine error: {exc}")
        return compute_metrics(pd.Series(dtype=float), [])

    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(eq_rebased, [])


def _load_prices() -> pd.DataFrame:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START_TS) & (prices.index <= END_TS)]
    if prices.empty:
        raise ValueError(f"No data for {SYMBOL} in the requested date range")
    print(
        f"{SYMBOL}: {len(prices)} bars  "
        f"({prices.index[0].date()} → {prices.index[-1].date()})"
    )
    return prices


def _pick_window_starts(prices: pd.DataFrame) -> list[int]:
    min_start = HOMC_TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = len(prices) - SIX_MONTHS - 1
    if max_start - min_start < N_WINDOWS:
        raise ValueError(
            f"{SYMBOL} has too few bars for {N_WINDOWS} {SIX_MONTHS}-bar windows "
            f"with a {HOMC_TRAIN_WINDOW}-bar warmup"
        )
    from _window_sampler import draw_non_overlapping_starts
    return draw_non_overlapping_starts(
        seed=SEED,
        min_start=min_start,
        max_start=max_start,
        window_len=SIX_MONTHS,
        n_windows=N_WINDOWS,
    )


def _run_grid_commission_deadband(
    prices: pd.DataFrame, starts: list[int]
) -> list[WindowRow]:
    rows: list[WindowRow] = []
    total = len(COMMISSION_GRID) * len(DEADBAND_GRID)
    cell_idx = 0
    for commission in COMMISSION_GRID:
        for deadband in DEADBAND_GRID:
            cell_idx += 1
            print(
                f"[commission×deadband {cell_idx}/{total}] "
                f"commission_bps={commission}  min_trade_fraction={deadband}"
            )
            cfg = _hvol_config(
                commission_bps=commission,
                slippage_bps=BASELINE_SLIPPAGE,
                min_trade_fraction=deadband,
            )
            for w_idx, start_i in enumerate(starts, start=1):
                end_i = start_i + SIX_MONTHS
                eval_window = prices.iloc[start_i:end_i]
                m = _run_one_window(cfg, prices, start_i, end_i, SYMBOL)
                rows.append(
                    WindowRow(
                        grid="commission_deadband",
                        commission_bps=commission,
                        slippage_bps=BASELINE_SLIPPAGE,
                        min_trade_fraction=deadband,
                        window_idx=w_idx,
                        start=eval_window.index[0],
                        end=eval_window.index[-1],
                        sharpe=m.sharpe,
                        cagr=m.cagr,
                        max_drawdown=m.max_drawdown,
                    )
                )
    return rows


def _run_grid_slippage(
    prices: pd.DataFrame, starts: list[int]
) -> list[WindowRow]:
    rows: list[WindowRow] = []
    for i, slippage in enumerate(SLIPPAGE_GRID, start=1):
        print(
            f"[slippage {i}/{len(SLIPPAGE_GRID)}] "
            f"slippage_bps={slippage}  (commission={BASELINE_COMMISSION}, "
            f"deadband={BASELINE_DEADBAND})"
        )
        cfg = _hvol_config(
            commission_bps=BASELINE_COMMISSION,
            slippage_bps=slippage,
            min_trade_fraction=BASELINE_DEADBAND,
        )
        for w_idx, start_i in enumerate(starts, start=1):
            end_i = start_i + SIX_MONTHS
            eval_window = prices.iloc[start_i:end_i]
            m = _run_one_window(cfg, prices, start_i, end_i, SYMBOL)
            rows.append(
                WindowRow(
                    grid="slippage",
                    commission_bps=BASELINE_COMMISSION,
                    slippage_bps=slippage,
                    min_trade_fraction=BASELINE_DEADBAND,
                    window_idx=w_idx,
                    start=eval_window.index[0],
                    end=eval_window.index[-1],
                    sharpe=m.sharpe,
                    cagr=m.cagr,
                    max_drawdown=m.max_drawdown,
                )
            )
    return rows


def _to_frame(rows: list[WindowRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "grid": r.grid,
                "commission_bps": r.commission_bps,
                "slippage_bps": r.slippage_bps,
                "min_trade_fraction": r.min_trade_fraction,
                "window_idx": r.window_idx,
                "start": r.start,
                "end": r.end,
                "sharpe": r.sharpe,
                "cagr": r.cagr,
                "max_drawdown": r.max_drawdown,
            }
            for r in rows
        ]
    )


def _print_commission_deadband_table(df: pd.DataFrame) -> pd.DataFrame:
    cd = df[df["grid"] == "commission_deadband"]
    pivot = (
        cd.groupby(["commission_bps", "min_trade_fraction"])["sharpe"]
        .median()
        .unstack("min_trade_fraction")
        .reindex(index=COMMISSION_GRID, columns=DEADBAND_GRID)
    )

    print()
    print("=" * 90)
    print(
        "Grid 1 — median H-Vol Sharpe on 16 random BTC 6-month windows (seed 42)"
    )
    print("Rows: commission_bps   Columns: min_trade_fraction   (slippage fixed at 5.0)")
    print("=" * 90)
    header = "commission\\deadband | " + " ".join(f"{d:>7.2f}" for d in DEADBAND_GRID)
    print(header)
    print("-" * len(header))
    for commission in COMMISSION_GRID:
        cells = " ".join(
            f"{pivot.loc[commission, d]:>7.2f}" for d in DEADBAND_GRID
        )
        print(f"{commission:>18.2f} | {cells}")
    return pivot


def _print_slippage_table(df: pd.DataFrame) -> pd.Series:
    slp = df[df["grid"] == "slippage"]
    row = (
        slp.groupby("slippage_bps")["sharpe"]
        .median()
        .reindex(SLIPPAGE_GRID)
    )

    print()
    print("=" * 90)
    print(
        "Grid 2 — median H-Vol Sharpe vs slippage_bps "
        "(commission=5.0, deadband=0.20)"
    )
    print("=" * 90)
    print("slippage_bps | " + " ".join(f"{s:>7.2f}" for s in SLIPPAGE_GRID))
    print("-" * 90)
    print("median Sharpe| " + " ".join(f"{row.loc[s]:>7.2f}" for s in SLIPPAGE_GRID))
    return row


def _print_cost_ridge(
    pivot: pd.DataFrame, slippage_row: pd.Series
) -> None:
    """Configs whose median Sharpe drops by more than COST_RIDGE_DROP from
    the baseline. Baseline is taken from Grid 1 at (5 bps, 0.20), which is
    exactly the config the headline was generated at."""
    try:
        baseline = pivot.loc[BASELINE_COMMISSION, BASELINE_DEADBAND]
    except KeyError:
        baseline = float("nan")

    print()
    print("=" * 90)
    print(
        f"Cost ridge — configs where median Sharpe drops by > {COST_RIDGE_DROP:.2f} "
        f"from baseline"
    )
    print(
        f"Baseline: commission={BASELINE_COMMISSION}, deadband={BASELINE_DEADBAND}, "
        f"slippage={BASELINE_SLIPPAGE}  -->  median Sharpe = {baseline:.2f}"
    )
    print("=" * 90)

    threshold = baseline - COST_RIDGE_DROP
    offenders: list[tuple[str, float]] = []
    for commission in COMMISSION_GRID:
        for deadband in DEADBAND_GRID:
            val = pivot.loc[commission, deadband]
            if not np.isnan(val) and val < threshold:
                label = (
                    f"commission={commission:>5.2f}  deadband={deadband:>5.2f}  "
                    f"slippage={BASELINE_SLIPPAGE:>5.2f}"
                )
                offenders.append((label, val))

    for slippage in SLIPPAGE_GRID:
        val = slippage_row.loc[slippage]
        if not np.isnan(val) and val < threshold:
            label = (
                f"commission={BASELINE_COMMISSION:>5.2f}  "
                f"deadband={BASELINE_DEADBAND:>5.2f}  slippage={slippage:>5.2f}"
            )
            offenders.append((label, val))

    if not offenders:
        print(
            "No config in either grid drops Sharpe by more than "
            f"{COST_RIDGE_DROP:.2f} from baseline."
        )
        return

    offenders.sort(key=lambda x: x[1])
    for label, val in offenders:
        drop = baseline - val
        print(f"  {label}  -->  Sharpe={val:>5.2f}  (drop={drop:+.2f})")


def main() -> None:
    prices = _load_prices()
    starts = _pick_window_starts(prices)
    print(f"Selected {len(starts)} window starts (seed={SEED}).")

    rows: list[WindowRow] = []
    rows.extend(_run_grid_commission_deadband(prices, starts))
    rows.extend(_run_grid_slippage(prices, starts))

    df = _to_frame(rows)

    pivot = _print_commission_deadband_table(df)
    slippage_row = _print_slippage_table(df)
    _print_cost_ridge(pivot, slippage_row)

    out_path = Path(__file__).parent / "data" / "cost_sensitivity.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print()
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
