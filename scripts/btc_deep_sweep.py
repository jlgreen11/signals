"""Comprehensive BTC backtest sweep across the full hyperparameter surface.

Runs five independent experiment dimensions on the 16-window random BTC
evaluation (seed 42). Saves every per-window result to parquet so the
raw data is inspectable/analyzable offline without re-running.

Dimensions:

  Dim 1 — HOMC order × n_states
          orders {1..9} × n_states {3, 5, 7}
          27 configs, 432 backtests
          Question: does BTC's HOMC profile match the S&P one (sparsity
          wall around order=7) or is there a different sweet spot?

  Dim 2 — Composite grid × train_window × laplace_alpha
          {2×2, 3×3, 4×4, 5×5} × {252, 504, 1000} × {0.01, 1.0}
          24 configs, 384 backtests
          Question: is the current 3×3 / 252 / 0.01 default actually
          optimal for composite on random windows?

  Dim 3 — Hybrid vol_quantile × max_long
          quantile {0.50, 0.60, 0.70, 0.75, 0.80} × max_long {1.0, 1.25, 1.5, 2.0}
          20 configs, 320 backtests
          Question: does the Tier-0e q=0.70 hold up in the cross-section,
          and is the Tier-0f leverage finding stable?

  Dim 4 — Hybrid retrain frequency
          retrain_freq {7, 14, 21, 42, 63}
          5 configs, 80 backtests
          Question: does the production 21-bar retrain matter vs more/less?

  Dim 5 — Hybrid buy/sell threshold grid
          buy_bps {10, 15, 20, 25, 30} × sell_bps {-10, -15, -20, -25, -30}
          25 configs, 400 backtests
          Question: where does the threshold parameterization sit on the
          Sharpe surface, and is there a better (buy, sell) pair?

Total: 1616 backtests, seed 42 single-pass. Phase 3 (robustness) re-runs
top configs per dim at seeds {7, 100, 999} separately — see
scripts/btc_deep_sweep_robustness.py.

Outputs:
  scripts/data/btc_deep_sweep_results.parquet  — long-format: one row per
      (dimension, config, window) with all metrics
  scripts/data/btc_deep_sweep_summary.parquet  — one row per config with
      aggregate stats across the 16 windows
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "BTC-USD"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")
SEED = 42
N_WINDOWS = 16
SIX_MONTHS = 126
VOL_WINDOW = 10
WARMUP_PAD = 5


@dataclass
class SweepConfig:
    """One row in the sweep — a BacktestConfig plus descriptive metadata."""
    dim: str                    # "homc_order_states", "composite_grid", etc.
    config_id: int              # unique within dim
    cfg: BacktestConfig
    params: dict[str, Any] = field(default_factory=dict)


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
    return compute_metrics(eq_rebased, [])


# ============================================================================
# Dimension builders — each returns a list of SweepConfig
# ============================================================================


def _dim1_homc_order_states() -> list[SweepConfig]:
    """HOMC order × n_states matrix."""
    configs: list[SweepConfig] = []
    for config_id, (order, n_states) in enumerate(
        product(range(1, 10), [3, 5, 7])
    ):
        configs.append(
            SweepConfig(
                dim="homc_order_states",
                config_id=config_id,
                cfg=BacktestConfig(
                    model_type="homc",
                    train_window=1000,
                    retrain_freq=21,
                    n_states=n_states,
                    order=order,
                    vol_window=VOL_WINDOW,
                    laplace_alpha=1.0,
                ),
                params={"order": order, "n_states": n_states},
            )
        )
    return configs


def _dim2_composite_grid() -> list[SweepConfig]:
    """Composite return_bins × vol_bins × train_window × laplace_alpha."""
    configs: list[SweepConfig] = []
    for config_id, (rbins, train_window, alpha) in enumerate(
        product([2, 3, 4, 5], [252, 504, 1000], [0.01, 1.0])
    ):
        configs.append(
            SweepConfig(
                dim="composite_grid",
                config_id=config_id,
                cfg=BacktestConfig(
                    model_type="composite",
                    train_window=train_window,
                    retrain_freq=21,
                    return_bins=rbins,
                    volatility_bins=rbins,  # square grid
                    vol_window=VOL_WINDOW,
                    laplace_alpha=alpha,
                ),
                params={
                    "return_bins": rbins,
                    "vol_bins": rbins,
                    "train_window": train_window,
                    "laplace_alpha": alpha,
                },
            )
        )
    return configs


def _dim3_hybrid_quantile_leverage() -> list[SweepConfig]:
    """Hybrid vol_quantile × max_long matrix."""
    configs: list[SweepConfig] = []
    for config_id, (quantile, max_long) in enumerate(
        product([0.50, 0.60, 0.70, 0.75, 0.80], [1.0, 1.25, 1.5, 2.0])
    ):
        configs.append(
            SweepConfig(
                dim="hybrid_quantile_leverage",
                config_id=config_id,
                cfg=BacktestConfig(
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
                    hybrid_vol_quantile=quantile,
                    max_long=max_long,
                ),
                params={"vol_quantile": quantile, "max_long": max_long},
            )
        )
    return configs


def _dim4_hybrid_retrain_freq() -> list[SweepConfig]:
    """Retrain frequency impact on the production H-Vol default."""
    configs: list[SweepConfig] = []
    for config_id, retrain_freq in enumerate([7, 14, 21, 42, 63]):
        configs.append(
            SweepConfig(
                dim="hybrid_retrain_freq",
                config_id=config_id,
                cfg=BacktestConfig(
                    model_type="hybrid",
                    train_window=1000,
                    retrain_freq=retrain_freq,
                    n_states=5,
                    order=5,
                    return_bins=3,
                    volatility_bins=3,
                    vol_window=VOL_WINDOW,
                    laplace_alpha=0.01,
                    hybrid_routing_strategy="vol",
                    hybrid_vol_quantile=0.70,
                ),
                params={"retrain_freq": retrain_freq},
            )
        )
    return configs


def _dim5_hybrid_thresholds() -> list[SweepConfig]:
    """Buy/sell threshold grid for the H-Vol default."""
    configs: list[SweepConfig] = []
    for config_id, (buy_bps, sell_bps) in enumerate(
        product([10, 15, 20, 25, 30], [-10, -15, -20, -25, -30])
    ):
        configs.append(
            SweepConfig(
                dim="hybrid_thresholds",
                config_id=config_id,
                cfg=BacktestConfig(
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
                    buy_threshold_bps=float(buy_bps),
                    sell_threshold_bps=float(sell_bps),
                ),
                params={"buy_bps": buy_bps, "sell_bps": sell_bps},
            )
        )
    return configs


# ============================================================================
# Main sweep runner
# ============================================================================


def _run_dim(
    dim_name: str,
    configs: list[SweepConfig],
    prices: pd.DataFrame,
    starts: list[int],
) -> list[dict]:
    """Run all configs in a dimension across all random windows. Returns
    a list of long-format row dicts, one per (config, window)."""
    rows: list[dict] = []
    total = len(configs)
    t0 = time.time()
    print(f"\n{'=' * 80}")
    print(f"Dimension: {dim_name} — {total} configs × {len(starts)} windows = {total * len(starts)} backtests")
    print(f"{'=' * 80}")
    for idx, sc in enumerate(configs, start=1):
        sharpes: list[float] = []
        for win_idx, start_i in enumerate(starts):
            end_i = start_i + SIX_MONTHS
            m = _run_on_window(sc.cfg, prices, start_i, end_i)
            rows.append({
                "dim": dim_name,
                "config_id": sc.config_id,
                "window_idx": win_idx,
                "window_start": str(prices.index[start_i].date()),
                "window_end": str(prices.index[end_i - 1].date()),
                "cagr": float(m.cagr),
                "sharpe": float(m.sharpe),
                "max_dd": float(m.max_drawdown),
                "n_trades": int(m.n_trades),
                "final_equity": float(m.final_equity),
                **sc.params,
            })
            sharpes.append(float(m.sharpe))
        median_sh = float(np.median(sharpes))
        mean_sh = float(np.mean(sharpes))
        pos = int(sum(1 for s in sharpes if s > 0))
        elapsed = time.time() - t0
        print(
            f"  [{idx:3d}/{total}] {sc.params} → "
            f"mean Sh {mean_sh:+5.2f}  median Sh {median_sh:+5.2f}  "
            f"pos {pos}/{len(starts)}  "
            f"({elapsed:4.0f}s elapsed)"
        )
    return rows


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars ({prices.index[0].date()} → {prices.index[-1].date()})")

    # Window selection — binding constraint is the 1000-bar HOMC/hybrid
    # training window. All dimensions use the same 16 random windows so
    # results are directly comparable.
    min_start = 1000 + VOL_WINDOW + WARMUP_PAD
    max_start = len(prices) - SIX_MONTHS - 1
    rng = random.Random(SEED)
    starts = sorted(rng.sample(range(min_start, max_start), N_WINDOWS))

    # Build all dimensions
    dimensions = [
        ("homc_order_states", _dim1_homc_order_states()),
        ("composite_grid", _dim2_composite_grid()),
        ("hybrid_quantile_leverage", _dim3_hybrid_quantile_leverage()),
        ("hybrid_retrain_freq", _dim4_hybrid_retrain_freq()),
        ("hybrid_thresholds", _dim5_hybrid_thresholds()),
    ]
    total_configs = sum(len(c) for _, c in dimensions)
    total_backtests = total_configs * N_WINDOWS
    print(f"Total: {total_configs} configs × {N_WINDOWS} windows = {total_backtests} backtests")

    overall_t0 = time.time()
    all_rows: list[dict] = []
    for dim_name, configs in dimensions:
        rows = _run_dim(dim_name, configs, prices, starts)
        all_rows.extend(rows)

    elapsed = time.time() - overall_t0
    print(f"\n{'=' * 80}")
    print(f"Full sweep complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Collected {len(all_rows)} rows")
    print(f"{'=' * 80}")

    # Long-format raw data
    df = pd.DataFrame(all_rows)
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / "btc_deep_sweep_results.parquet"
    df.to_parquet(raw_path, index=False)
    print(f"Raw data: {raw_path}")

    # Config-level summary
    summary_cols = ["dim", "config_id"]
    param_cols = [c for c in df.columns if c not in {
        "dim", "config_id", "window_idx", "window_start", "window_end",
        "cagr", "sharpe", "max_dd", "n_trades", "final_equity",
    }]
    grouped = df.groupby(summary_cols + param_cols, dropna=False).agg(
        mean_sharpe=("sharpe", "mean"),
        median_sharpe=("sharpe", "median"),
        std_sharpe=("sharpe", "std"),
        mean_cagr=("cagr", "mean"),
        median_cagr=("cagr", "median"),
        mean_max_dd=("max_dd", "mean"),
        mean_trades=("n_trades", "mean"),
        positive_windows=("cagr", lambda s: int((s > 0).sum())),
        n_windows=("sharpe", "count"),
    ).reset_index()

    summary_path = data_dir / "btc_deep_sweep_summary.parquet"
    grouped.to_parquet(summary_path, index=False)
    print(f"Summary  : {summary_path}")

    # Print a quick top-10 per dimension
    print()
    print("=" * 80)
    print("Top-3 per dimension by median Sharpe")
    print("=" * 80)
    for dim_name, _ in dimensions:
        sub = grouped[grouped["dim"] == dim_name].sort_values("median_sharpe", ascending=False)
        print(f"\n  {dim_name}:")
        for _, r in sub.head(3).iterrows():
            param_str = " | ".join(f"{k}={r[k]}" for k in param_cols if not pd.isna(r[k]))
            print(
                f"    median Sh {r['median_sharpe']:+5.2f}  mean Sh {r['mean_sharpe']:+5.2f}  "
                f"median CAGR {r['median_cagr'] * 100:+6.1f}%  "
                f"MDD {r['mean_max_dd'] * 100:+6.1f}%  "
                f"pos {int(r['positive_windows'])}/{int(r['n_windows'])}  |  {param_str}"
            )


if __name__ == "__main__":
    main()
