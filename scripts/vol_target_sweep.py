"""Round-4 improvement #1 — enable vol_target on top of BTC_HYBRID_PRODUCTION.

The `vol_target_enabled` flag in `BacktestConfig` is a pre-built,
well-documented overlay (`signals/backtest/vol_target.py`) that scales
position size inversely to realized volatility. It has never been
enabled in production. On a vol-clustering asset like BTC this should
reduce variance without sacrificing mean return — free Sharpe in the
Moskowitz et al. (2012) sense.

Pre-registered grid (DO NOT EXPAND per guardrail D1; the hypothesis
was pre-stated in the vol_target.py docstring):

    vol_target_annual ∈ {0.15, 0.20, 0.25}
    seeds             ∈ 10 pre-registered seeds (same set as prior
                          sweeps for consistency)

Total: 3 configs × 10 seeds × 16 non-overlapping 6-month windows = 480
backtests on top of the Round-3 BTC_HYBRID_PRODUCTION bundle (q=0.50,
rf=14, tw=750, vol_window=10).

Evaluation:
  - 10-seed random-window eval on 2015-01-01 → 2024-12-31 (same universe
    as explore_improvements.py Tier 4 so the comparison is apples to
    apples).
  - Production-baseline control (vol_target_enabled=False) included as
    the "no-overlay" reference.

Success criterion: multi-seed avg median Sharpe ≥ production baseline
+ 0.10 (≥ 1.651) AND dominates baseline on min-seed.

Failure: if nothing beats baseline, write negative result and move on.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.engine import (
    BTC_HYBRID_PRODUCTION,
    BacktestConfig,
    BacktestEngine,
)
from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "BTC-USD"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")

SIX_MONTHS = 126
WARMUP_PAD = 5
N_WINDOWS = 16
SEEDS = [42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23]

# Pre-registered grid
VOL_TARGET_ANNUALS = (0.15, 0.20, 0.25)


@dataclass
class Config:
    label: str
    vol_target_enabled: bool
    vol_target_annual: float


def _build_grid() -> list[Config]:
    grid: list[Config] = [
        # Baseline: production bundle with vol_target OFF.
        Config(
            label="baseline_no_vol_target",
            vol_target_enabled=False,
            vol_target_annual=0.0,
        ),
    ]
    for target in VOL_TARGET_ANNUALS:
        grid.append(Config(
            label=f"vt_enabled_annual{target:.2f}",
            vol_target_enabled=True,
            vol_target_annual=target,
        ))
    return grid


def _make_cfg(c: Config) -> BacktestConfig:
    """Build a BacktestConfig from the production bundle + the overlay
    knobs. The production bundle (BTC_HYBRID_PRODUCTION) holds the
    Round-3 winner: q=0.50, rf=14, tw=750, vol_window=10."""
    base = dict(BTC_HYBRID_PRODUCTION)
    base["risk_free_rate"] = historical_usd_rate("2018-2024")
    base["vol_target_enabled"] = c.vol_target_enabled
    base["vol_target_annual"] = c.vol_target_annual
    base["vol_target_periods_per_year"] = 365  # BTC is crypto
    base["vol_target_max_scale"] = 2.0
    base["vol_target_min_scale"] = 0.0
    return BacktestConfig(**base)


def _run_one_window(
    cfg: BacktestConfig, prices: pd.DataFrame, start_i: int, end_i: int
) -> tuple[float, float, float, int]:
    slice_start = max(0, start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD)
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]
    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=SYMBOL)
    except Exception as e:
        print(f"    engine error: {e}")
        return 0.0, 0.0, 0.0, 0
    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return 0.0, 0.0, 0.0, 0
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    m = compute_metrics(
        eq_rebased,
        [],
        risk_free_rate=cfg.risk_free_rate,
        periods_per_year=cfg.periods_per_year,
    )
    return m.sharpe, m.cagr, m.max_drawdown, m.n_trades


def _evaluate_config(
    c: Config, prices: pd.DataFrame
) -> pd.DataFrame:
    cfg = _make_cfg(c)
    min_start = cfg.train_window + cfg.vol_window + WARMUP_PAD
    max_start = len(prices) - SIX_MONTHS - 1
    rows: list[dict] = []
    for seed in SEEDS:
        starts = draw_non_overlapping_starts(
            seed=seed,
            min_start=min_start,
            max_start=max_start,
            window_len=SIX_MONTHS,
            n_windows=N_WINDOWS,
        )
        for w, start_i in enumerate(starts):
            end_i = start_i + SIX_MONTHS
            sharpe, cagr, mdd, n_trades = _run_one_window(
                cfg, prices, start_i, end_i
            )
            rows.append({
                "label": c.label,
                "vol_target_enabled": c.vol_target_enabled,
                "vol_target_annual": c.vol_target_annual,
                "seed": seed,
                "window_idx": w,
                "start": prices.index[start_i],
                "end": prices.index[end_i - 1],
                "sharpe": sharpe,
                "cagr": cagr,
                "max_dd": mdd,
                "n_trades": n_trades,
            })
    return pd.DataFrame(rows)


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars")

    grid = _build_grid()
    print(f"Pre-registered grid: {len(grid)} configs, {len(SEEDS)} seeds")

    t0 = time.time()
    all_rows: list[pd.DataFrame] = []
    for i, c in enumerate(grid, start=1):
        print(f"\n[{i}/{len(grid)}] {c.label}")
        df = _evaluate_config(c, prices)
        all_rows.append(df)
        per_seed_median = df.groupby("seed")["sharpe"].median()
        avg = per_seed_median.mean()
        stderr = per_seed_median.sem()
        mins = per_seed_median.min()
        maxs = per_seed_median.max()
        elapsed = time.time() - t0
        print(
            f"  avg median Sharpe: {avg:+.3f} ± {stderr:.3f}  "
            f"(min {mins:+.3f}, max {maxs:+.3f})  elapsed {elapsed:.0f}s"
        )

    full_df = pd.concat(all_rows, ignore_index=True)
    out_parquet = Path(__file__).parent / "data" / "vol_target_sweep.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}")

    per_seed = (
        full_df.groupby(["label", "vol_target_annual", "seed"])["sharpe"]
        .median()
        .reset_index()
    )
    agg = (
        per_seed.groupby(["label", "vol_target_annual"])["sharpe"]
        .agg(["mean", "sem", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print("\n" + "=" * 80)
    print("Multi-seed ranking — production bundle + vol_target overlay")
    print("=" * 80)
    print(agg.to_string(index=False))

    winner = agg.iloc[0]
    baseline_row = agg[agg["label"] == "baseline_no_vol_target"].iloc[0]
    winner_sharpe = winner["mean"]
    baseline_sharpe = baseline_row["mean"]
    baseline_min = baseline_row["min"]
    winner_min = winner["min"]

    delta = winner_sharpe - baseline_sharpe
    print(f"\nBaseline (vol_target OFF):   {baseline_sharpe:+.3f} ± "
          f"{baseline_row['sem']:.3f}  (min seed {baseline_min:+.3f})")
    print(f"Winner ({winner['label']}): {winner_sharpe:+.3f} ± "
          f"{winner['sem']:.3f}  (min seed {winner_min:+.3f})")
    print(f"Delta: {delta:+.3f}")

    # Success criterion: winner must beat baseline by ≥ 0.10 AND
    # dominate on min-seed.
    materiality_ok = delta >= 0.10
    min_seed_ok = winner_min > baseline_min
    success = materiality_ok and min_seed_ok and winner["label"] != "baseline_no_vol_target"
    print(f"\nMateriality (Δ ≥ 0.10):      "
          f"{'PASS' if materiality_ok else 'FAIL'}")
    print(f"Min-seed dominance:          "
          f"{'PASS' if min_seed_ok else 'FAIL'}")
    print(f"Overall: {'SUCCESS — enable vol_target' if success else 'NO CHANGE — keep production'}")

    out_md = Path(__file__).parent / "VOL_TARGET_SWEEP_RESULTS.md"
    lines = [
        "# Round-4 #1 — vol_target overlay sweep",
        "",
        "**Run date**: 2026-04-11",
        "**Script**: `scripts/vol_target_sweep.py`",
        "**Test parameters**:",
        "",
        "- Base config: `BTC_HYBRID_PRODUCTION` (q=0.50, rf=14, tw=750, vw=10)",
        "- Overlay: `vol_target_enabled` ∈ {False, True} with",
        f"  `vol_target_annual` ∈ {list(VOL_TARGET_ANNUALS)} when enabled",
        f"- Seeds: {SEEDS}",
        f"- Windows: {N_WINDOWS} non-overlapping 6-month per seed",
        "- Annualization: 365/yr, rf = historical_usd_rate('2018-2024')",
        "",
        "## Multi-seed ranking",
        "",
        "| label | vt_annual | avg Sharpe | stderr | min seed | max seed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| `{r['label']}` | {r['vol_target_annual']:.2f} | "
            f"{r['mean']:+.3f} | {r['sem']:.3f} | "
            f"{r['min']:+.3f} | {r['max']:+.3f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- Materiality (winner − baseline ≥ 0.10): "
        f"**{'PASS' if materiality_ok else 'FAIL'}** ({delta:+.3f})",
        f"- Min-seed dominance: **{'PASS' if min_seed_ok else 'FAIL'}** "
        f"({winner_min:+.3f} vs baseline {baseline_min:+.3f})",
        f"- **{'SUCCESS' if success else 'NO CHANGE'}**",
        "",
    ]
    if success:
        lines += [
            f"Recommend enabling `vol_target_enabled=True` with "
            f"`vol_target_annual={winner['vol_target_annual']:.2f}` as the "
            "new BTC production default.",
        ]
    else:
        lines += [
            "Vol target overlay did not materially improve the production "
            "bundle. Baseline (no overlay) retained.",
        ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
