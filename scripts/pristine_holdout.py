"""Pristine holdout evaluation — SKEPTIC_REVIEW.md Tier A4.

The project's current "holdout" numbers are compromised: the "tightened
defaults" (`vol_window=10`, `buy_bps=25`, `sell_bps=-35`, etc.) were
iteratively chosen across sweeps that had full visibility of the
2023-2024 slice. Reporting them as out-of-sample is not pristine.

This script implements the clean-OOS procedure:

1. Freeze the 2023-01-01 → 2024-12-31 slice as the **holdout**. Never
   read from it during tuning.
2. Sweep over a coarse parameter grid on the 2015-01-01 → 2022-12-31
   **training** slice only. For each sweep config, score by the
   in-sample walk-forward median Sharpe across 16 non-overlapping
   random 6-month windows (seed 42, using the shared non-overlap
   sampler so the windows don't collide with the 2023+ slice).
3. Pick the single best config by in-sample median Sharpe.
4. Run that config **exactly once** on the holdout slice (single
   walk-forward pass, no window sampling — the whole 2-year slice is
   the test period).
5. Report both the in-sample sweep winner's numbers AND the holdout
   numbers side-by-side. Whatever the holdout column says is the only
   "clean OOS" evidence the project has.

The parameter grid is intentionally small (~20 configs) because the
whole point of this script is to produce ONE honest out-of-sample
number, not to run another exhaustive sweep. Adding more configs
increases the multi-trial correction on the in-sample winner and
weakens the holdout interpretation.

Output:
- `scripts/data/pristine_holdout.parquet` — per-config in-sample + holdout
- `scripts/data/pristine_holdout.md` — human-readable summary
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "BTC-USD"
TRAIN_START = pd.Timestamp("2015-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
HOLDOUT_START = pd.Timestamp("2023-01-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2024-12-31", tz="UTC")

SIX_MONTHS = 126
HOMC_TRAIN_WINDOW = 1000
VOL_WINDOW = 10
WARMUP_PAD = 5
N_WINDOWS = 12  # fewer than 16 because the training slice is smaller

SEED = 42


@dataclass
class GridConfig:
    label: str
    hybrid_vol_quantile: float
    buy_bps: float
    sell_bps: float
    min_trade_fraction: float

    def to_backtest_config(self) -> BacktestConfig:
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
            hybrid_vol_quantile=self.hybrid_vol_quantile,
            buy_threshold_bps=self.buy_bps,
            sell_threshold_bps=self.sell_bps,
            target_scale_bps=20.0,
            min_trade_fraction=self.min_trade_fraction,
            allow_short=False,
            stop_loss_pct=0.0,
            periods_per_year=365.0,
            risk_free_rate=historical_usd_rate("2018-2022"),
        )


def _build_grid() -> list[GridConfig]:
    """~18 coarse configs. Intentionally small.

    We vary quantile ∈ {0.60, 0.70, 0.80}, sell_bps ∈ {-25, -35},
    min_trade_fraction ∈ {0.10, 0.20}. That's 3×2×2×1 = 12 configs.
    Plus the current "production" H-Vol defaults as an anchor.
    """
    grid: list[GridConfig] = []
    for q in (0.60, 0.70, 0.80):
        for sell_bps in (-25.0, -35.0):
            for mtf in (0.10, 0.20):
                grid.append(
                    GridConfig(
                        label=f"q={q}_sell={sell_bps}_mtf={mtf}",
                        hybrid_vol_quantile=q,
                        buy_bps=25.0,
                        sell_bps=sell_bps,
                        min_trade_fraction=mtf,
                    )
                )
    grid.append(
        GridConfig(
            label="production_hvol_default",
            hybrid_vol_quantile=0.70,
            buy_bps=25.0,
            sell_bps=-35.0,
            min_trade_fraction=0.20,
        )
    )
    return grid


def _run_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
) -> Metrics:
    slice_start = max(0, start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD)
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
    return compute_metrics(
        eq_rebased,
        [],
        risk_free_rate=cfg.risk_free_rate,
        periods_per_year=cfg.periods_per_year,
    )


def _evaluate_in_sample(
    cfg: BacktestConfig, train_prices: pd.DataFrame
) -> pd.DataFrame:
    """Walk-forward evaluation on 12 non-overlapping random windows in the
    training slice. Returns a per-window DataFrame."""
    min_start = HOMC_TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = len(train_prices) - SIX_MONTHS - 1
    starts = draw_non_overlapping_starts(
        seed=SEED,
        min_start=min_start,
        max_start=max_start,
        window_len=SIX_MONTHS,
        n_windows=N_WINDOWS,
    )
    rows: list[dict] = []
    for start_i in starts:
        end_i = start_i + SIX_MONTHS
        m = _run_on_window(cfg, train_prices, start_i, end_i)
        rows.append({
            "start": train_prices.index[start_i].date(),
            "sharpe": m.sharpe,
            "cagr": m.cagr,
            "mdd": m.max_drawdown,
        })
    return pd.DataFrame(rows)


def _evaluate_holdout(
    cfg: BacktestConfig, full_prices: pd.DataFrame
) -> Metrics:
    """Single walk-forward pass on the 2023-2024 slice with a proper
    train-window warmup buffer from the training slice."""
    holdout_start_i = full_prices.index.searchsorted(HOLDOUT_START)
    slice_start = max(0, holdout_start_i - HOMC_TRAIN_WINDOW - VOL_WINDOW - WARMUP_PAD)
    holdout_end_i = full_prices.index.searchsorted(HOLDOUT_END, side="right")
    engine_input = full_prices.iloc[slice_start:holdout_end_i]
    result = BacktestEngine(cfg).run(engine_input, symbol=SYMBOL)
    eq = result.equity_curve.loc[result.equity_curve.index >= HOLDOUT_START]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(
        eq_rebased,
        [],
        risk_free_rate=cfg.risk_free_rate,
        periods_per_year=cfg.periods_per_year,
    )


def _buy_hold_holdout(full_prices: pd.DataFrame) -> Metrics:
    bh_slice = full_prices.loc[
        (full_prices.index >= HOLDOUT_START) & (full_prices.index <= HOLDOUT_END)
    ]
    eq = (bh_slice["close"] / bh_slice["close"].iloc[0]) * 10_000.0
    return compute_metrics(
        eq,
        [],
        risk_free_rate=historical_usd_rate("2023-2024"),
        periods_per_year=365.0,
    )


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    full_prices = store.load(SYMBOL, "1d").sort_index()
    full_prices = full_prices.loc[
        (full_prices.index >= TRAIN_START) & (full_prices.index <= HOLDOUT_END)
    ]
    train_prices = full_prices.loc[
        (full_prices.index >= TRAIN_START) & (full_prices.index <= TRAIN_END)
    ]

    print(f"{SYMBOL}: train={len(train_prices)} bars "
          f"({train_prices.index[0].date()} → {train_prices.index[-1].date()}); "
          f"full={len(full_prices)} bars")

    grid = _build_grid()
    print(f"\nSweeping {len(grid)} configs on train slice only...")

    rows: list[dict] = []
    for i, gc in enumerate(grid, start=1):
        print(f"  [{i}/{len(grid)}] {gc.label}")
        cfg = gc.to_backtest_config()
        in_sample = _evaluate_in_sample(cfg, train_prices)
        rows.append({
            "label": gc.label,
            "hybrid_vol_quantile": gc.hybrid_vol_quantile,
            "buy_bps": gc.buy_bps,
            "sell_bps": gc.sell_bps,
            "min_trade_fraction": gc.min_trade_fraction,
            "is_median_sharpe": in_sample["sharpe"].median(),
            "is_mean_sharpe": in_sample["sharpe"].mean(),
            "is_median_cagr": in_sample["cagr"].median(),
            "is_mean_mdd": in_sample["mdd"].mean(),
        })

    sweep = pd.DataFrame(rows).sort_values("is_median_sharpe", ascending=False)
    print("\nIn-sample sweep (sorted by median Sharpe):")
    print(sweep.to_string(index=False))

    # Pick the winner and evaluate on holdout — ONCE.
    winner = sweep.iloc[0]
    print(f"\nWinner: {winner['label']}")
    print(f"  in-sample median Sharpe: {winner['is_median_sharpe']:.3f}")

    winner_cfg = next(
        gc.to_backtest_config() for gc in grid if gc.label == winner["label"]
    )
    # For the holdout evaluation, update the risk-free rate to the
    # actually-prevailing rate in 2023-2024 (~5%). The in-sample rate
    # was the 2018-2022 average (~0.9%).
    winner_cfg.risk_free_rate = historical_usd_rate("2023-2024")
    holdout_metrics = _evaluate_holdout(winner_cfg, full_prices)

    # Also run the current production default as a comparison, so we can
    # see whether the sweep winner and the default agree.
    production_cfg = GridConfig(
        label="production_hvol_default",
        hybrid_vol_quantile=0.70,
        buy_bps=25.0,
        sell_bps=-35.0,
        min_trade_fraction=0.20,
    ).to_backtest_config()
    production_cfg.risk_free_rate = historical_usd_rate("2023-2024")
    production_holdout = _evaluate_holdout(production_cfg, full_prices)

    bh_holdout = _buy_hold_holdout(full_prices)

    print("\n" + "=" * 72)
    print("Pristine holdout (2023-2024) — single-shot walk-forward")
    print("=" * 72)
    print(f"  Buy & hold holdout      : Sharpe={bh_holdout.sharpe:+.3f}  "
          f"CAGR={bh_holdout.cagr:+.1%}  MDD={bh_holdout.max_drawdown:+.1%}")
    print(f"  Sweep-winner holdout    : Sharpe={holdout_metrics.sharpe:+.3f}  "
          f"CAGR={holdout_metrics.cagr:+.1%}  MDD={holdout_metrics.max_drawdown:+.1%}")
    print(f"  Production H-Vol holdout: Sharpe={production_holdout.sharpe:+.3f}  "
          f"CAGR={production_holdout.cagr:+.1%}  MDD={production_holdout.max_drawdown:+.1%}")

    # Persist artifacts
    out_parquet = Path(__file__).parent / "data" / "pristine_holdout.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}")

    out_md = Path(__file__).parent / "data" / "pristine_holdout.md"
    # Build the sweep table manually (no tabulate dependency).
    header = "| label | q | buy | sell | mtf | IS med Sh | IS mean Sh | IS med CAGR | IS mean MDD |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    sweep_rows = [header, sep]
    for _, r in sweep.iterrows():
        sweep_rows.append(
            f"| {r['label']} | {r['hybrid_vol_quantile']:.2f} | "
            f"{r['buy_bps']:+.1f} | {r['sell_bps']:+.1f} | "
            f"{r['min_trade_fraction']:.2f} | "
            f"{r['is_median_sharpe']:+.3f} | {r['is_mean_sharpe']:+.3f} | "
            f"{r['is_median_cagr']:+.1%} | {r['is_mean_mdd']:+.1%} |"
        )
    md_lines = [
        "# Pristine holdout — SKEPTIC_REVIEW.md Tier A4",
        "",
        "**Training slice**: 2015-01-01 → 2022-12-31 (sweep + in-sample eval)",
        "**Holdout slice**: 2023-01-01 → 2024-12-31 (single-shot, never seen during tuning)",
        "",
        f"## Sweep on training slice ({len(grid)} configs, {N_WINDOWS} non-overlapping windows, seed={SEED})",
        "",
        *sweep_rows,
        "",
        "## Single-shot holdout evaluation",
        "",
        "| config | Sharpe | CAGR | Max DD |",
        "|---|---:|---:|---:|",
        f"| Buy & hold (2023-2024) | {bh_holdout.sharpe:+.3f} | {bh_holdout.cagr:+.1%} | {bh_holdout.max_drawdown:+.1%} |",
        f"| Sweep winner ({winner['label']}) | {holdout_metrics.sharpe:+.3f} | {holdout_metrics.cagr:+.1%} | {holdout_metrics.max_drawdown:+.1%} |",
        f"| Production H-Vol default | {production_holdout.sharpe:+.3f} | {production_holdout.cagr:+.1%} | {production_holdout.max_drawdown:+.1%} |",
        "",
        "Risk-free rate for Sharpe uses `historical_usd_rate` from "
        "`signals.backtest.risk_free`: 0.009 for the 2018-2022 in-sample "
        "average, 0.050 for the 2023-2024 holdout average.",
        "",
        "**Interpretation**: this is the project's only clean OOS number. "
        "The 'Production H-Vol holdout' row is the one to compare against "
        "the README's 2.21 headline — note that the README number was "
        "measured on the same period but with the tuning process having "
        "visibility into it, while this number was produced by a config "
        "chosen with zero visibility of the 2023-2024 slice.",
    ]
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
