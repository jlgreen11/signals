"""Large-grid improvement search — materially beat the 0.78 baseline.

Addresses the user's Round-3 request: "explore many many more permutations
of variables to materially improve the algo". The current production H-Vol
@ q=0.70 baseline has a multi-seed avg median Sharpe of 0.78 ± 0.08 on 10
seeds × 16 non-overlapping 6-month BTC windows. This script casts a much
wider net across parameters and structural alternatives to see if any
configuration clears 1.0 Sharpe robustly — the "material improvement"
threshold.

Multi-testing discipline:
  - The search uses 5 seeds during exploration and 10 seeds during final
    candidate comparison. All seeds are pre-registered.
  - Any config that emerges as a "winner" gets re-scored with the
    project-level DSR correction using n_trials >= <size of this sweep>
    plus the legacy project-level count ~1900.
  - Nothing gets promoted until it beats the baseline at mean Sharpe AND
    at every seed's min, NOT on a single seed.

Families explored (four tiers, short-to-long compute budget):

  Tier 1 — Pure vol filter (no Markov) sweep. Fast.
    vol_window ∈ {5, 10, 14, 20, 30}
    quantile   ∈ {0.30, 0.40, 0.50, 0.60, 0.70}
    retrain_freq ∈ {7, 21, 42}
    75 configs × 5 seeds × 16 windows = 6000 backtests

  Tier 2 — Vol filter + vol-target overlay on top of the Tier-1 winner.
    target_annual ∈ {0.15, 0.20, 0.25, 0.30, 0.40}
    max_scale    ∈ {1.5, 2.0, 3.0}
    15 configs × 5 seeds × 16 windows = 1200 backtests

  Tier 3 — Hybrid with wider axes than the existing multi_seed_eval sweep.
    vol_window ∈ {5, 10, 14}
    quantile   ∈ {0.40, 0.50, 0.55}
    retrain_freq ∈ {14, 21}
    train_window ∈ {750, 1000, 1500}
    54 configs × 5 seeds × 16 windows = 4320 backtests

  Tier 4 — 10-seed confirmation for the top 5 of Tier 1-3 combined.
    5 configs × 10 seeds × 16 windows = 800 backtests
    Includes the current production baseline as a control.

Total compute budget: ~12,320 backtests. Pure vol filter is fast (~0.5s
each); hybrid variants are slower (~3s each). Estimated wall time
2-4 hours depending on hardware.

Outputs:
  scripts/data/explore_improvements.parquet   — raw per-window, all tiers
  scripts/data/explore_improvements.md        — tier-by-tier summary with
                                                DSR-corrected verdicts
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import (
    Metrics,
    compute_metrics,
    deflated_sharpe_ratio,
)
from signals.backtest.portfolio import Portfolio
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore
from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility

SYMBOL = "BTC-USD"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")

SIX_MONTHS = 126
HOMC_TRAIN_WINDOW = 1000
VOL_WINDOW_BASE = 10
WARMUP_PAD = 5
N_WINDOWS = 16

EXPLORE_SEEDS = [42, 7, 100, 999, 1337]         # 5 seeds for Tier 1-3
CONFIRM_SEEDS = [42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23]  # 10 for Tier 4

BASELINE_LABEL = "hybrid_prod_q0.70_w10_r21_tw1000"

# Project-level trial count — the sum of every sweep the project has
# previously run, per SKEPTIC_REVIEW § B5.
LEGACY_PROJECT_TRIALS = 1_900


# --------------------------------------------------------------------------
# Pure vol filter strategy (self-contained, mirrors trivial_baselines_btc.py)
# --------------------------------------------------------------------------


def _run_vol_filter(
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    *,
    vol_window: int,
    quantile: float,
    retrain_freq: int,
    train_window: int = HOMC_TRAIN_WINDOW,
    vol_target_annual: float | None = None,
    vol_target_max_scale: float = 2.0,
) -> Metrics:
    """Pure vol filter: target = +1 if current_vol < q-th percentile of train vol, else 0.

    Retrains the quantile threshold every `retrain_freq` bars on the
    trailing `train_window` bars. Optionally overlays a vol-target
    position sizer: target = base_target * min(max_scale, annual_target / realized_annualized_vol).
    """
    slice_start = max(0, start_i - train_window - vol_window - WARMUP_PAD)
    engine_input = prices.iloc[slice_start:end_i].copy()
    engine_input["return_1d"] = log_returns(engine_input["close"])
    engine_input["vol"] = rolling_volatility(
        engine_input["return_1d"], window=vol_window
    )
    engine_input = engine_input.dropna(subset=["return_1d", "vol"])

    eval_start_ts = prices.index[start_i]
    p = Portfolio(initial_cash=10_000.0, commission_bps=5.0, slippage_bps=5.0)

    bars_since_retrain = retrain_freq
    vol_threshold: float | None = None

    for i in range(train_window, len(engine_input) - 1):
        ts = engine_input.index[i]
        next_ts = engine_input.index[i + 1]
        close_price = float(engine_input.iloc[i]["close"])
        next_open = float(engine_input.iloc[i + 1]["open"])

        if bars_since_retrain >= retrain_freq:
            window_vols = engine_input.iloc[i - train_window : i]["vol"].dropna()
            if len(window_vols) >= 10:
                vol_threshold = float(np.quantile(window_vols, quantile))
            bars_since_retrain = 0

        current_vol = float(engine_input.iloc[i]["vol"])
        if vol_threshold is None:
            target = 0.0
        else:
            target = 1.0 if current_vol < vol_threshold else 0.0

        if vol_target_annual is not None and target != 0.0:
            # Scale base target by annual_target / realized_annualized
            realized_annual = current_vol * np.sqrt(365.0)
            if realized_annual > 1e-6:
                scale = min(vol_target_max_scale, vol_target_annual / realized_annual)
                scale = max(0.0, scale)
                target = target * scale

        p.set_target(next_ts, next_open, target, min_trade_fraction=0.10)
        p.mark(ts, close_price)
        bars_since_retrain += 1

    last_ts = engine_input.index[-1]
    last_close = float(engine_input.iloc[-1]["close"])
    p.flatten(last_ts, last_close)
    p.mark(last_ts, last_close)

    eq = p.equity_series().loc[lambda s: s.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * 10_000.0
    return compute_metrics(
        eq_rebased,
        [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=365.0,
    )


# --------------------------------------------------------------------------
# Hybrid strategy runner
# --------------------------------------------------------------------------


def _run_hybrid(
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    *,
    vol_window: int,
    quantile: float,
    retrain_freq: int,
    train_window: int,
) -> Metrics:
    cfg = BacktestConfig(
        model_type="hybrid",
        train_window=train_window,
        retrain_freq=retrain_freq,
        n_states=5,
        order=5,
        return_bins=3,
        volatility_bins=3,
        vol_window=vol_window,
        laplace_alpha=0.01,
        hybrid_routing_strategy="vol",
        hybrid_vol_quantile=quantile,
        periods_per_year=365.0,
        risk_free_rate=historical_usd_rate("2018-2024"),
    )
    slice_start = max(0, start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD)
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]
    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=SYMBOL)
    except Exception as e:
        print(f"    hybrid error: {e}")
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


# --------------------------------------------------------------------------
# Seed evaluation
# --------------------------------------------------------------------------


@dataclass
class EvalConfig:
    label: str
    family: str            # "vol_filter" | "vol_filter_vt" | "hybrid"
    vol_window: int
    quantile: float
    retrain_freq: int
    train_window: int = HOMC_TRAIN_WINDOW
    vol_target_annual: float | None = None
    vol_target_max_scale: float = 2.0


def _evaluate_config(
    ec: EvalConfig, prices: pd.DataFrame, seeds: list[int]
) -> pd.DataFrame:
    """For each seed, draw 16 non-overlapping windows and run the config.

    Returns a DataFrame with columns [seed, window_idx, start, end,
    sharpe, cagr, max_dd] plus the config metadata on every row.
    """
    min_start = HOMC_TRAIN_WINDOW + ec.vol_window + WARMUP_PAD
    max_start = len(prices) - SIX_MONTHS - 1

    rows: list[dict] = []
    for seed in seeds:
        starts = draw_non_overlapping_starts(
            seed=seed,
            min_start=min_start,
            max_start=max_start,
            window_len=SIX_MONTHS,
            n_windows=N_WINDOWS,
        )
        for w, start_i in enumerate(starts):
            end_i = start_i + SIX_MONTHS
            if ec.family.startswith("vol_filter"):
                m = _run_vol_filter(
                    prices,
                    start_i,
                    end_i,
                    vol_window=ec.vol_window,
                    quantile=ec.quantile,
                    retrain_freq=ec.retrain_freq,
                    train_window=ec.train_window,
                    vol_target_annual=ec.vol_target_annual,
                    vol_target_max_scale=ec.vol_target_max_scale,
                )
            elif ec.family == "hybrid":
                m = _run_hybrid(
                    prices,
                    start_i,
                    end_i,
                    vol_window=ec.vol_window,
                    quantile=ec.quantile,
                    retrain_freq=ec.retrain_freq,
                    train_window=ec.train_window,
                )
            else:
                raise ValueError(f"unknown family {ec.family!r}")
            rows.append({
                "label": ec.label,
                "family": ec.family,
                "vol_window": ec.vol_window,
                "quantile": ec.quantile,
                "retrain_freq": ec.retrain_freq,
                "train_window": ec.train_window,
                "vt_annual": ec.vol_target_annual,
                "vt_max_scale": ec.vol_target_max_scale,
                "seed": seed,
                "window_idx": w,
                "start": prices.index[start_i],
                "end": prices.index[end_i - 1],
                "sharpe": m.sharpe,
                "cagr": m.cagr,
                "max_dd": m.max_drawdown,
                "n_trades": m.n_trades,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Grid definitions
# --------------------------------------------------------------------------


def _tier1_vol_filter_grid() -> list[EvalConfig]:
    configs = []
    for vw in (5, 10, 14, 20, 30):
        for q in (0.30, 0.40, 0.50, 0.60, 0.70):
            for rf in (7, 21, 42):
                configs.append(
                    EvalConfig(
                        label=f"vf_vw{vw}_q{q:.2f}_rf{rf}",
                        family="vol_filter",
                        vol_window=vw,
                        quantile=q,
                        retrain_freq=rf,
                    )
                )
    return configs


def _tier2_vol_filter_vt_grid(best_t1: EvalConfig) -> list[EvalConfig]:
    configs = []
    for vt in (0.15, 0.20, 0.25, 0.30, 0.40):
        for mx in (1.5, 2.0, 3.0):
            configs.append(
                EvalConfig(
                    label=f"vf_vt_vw{best_t1.vol_window}_q{best_t1.quantile:.2f}_rf{best_t1.retrain_freq}_vt{vt:.2f}_mx{mx:.1f}",
                    family="vol_filter_vt",
                    vol_window=best_t1.vol_window,
                    quantile=best_t1.quantile,
                    retrain_freq=best_t1.retrain_freq,
                    vol_target_annual=vt,
                    vol_target_max_scale=mx,
                )
            )
    return configs


def _tier3_hybrid_grid() -> list[EvalConfig]:
    configs = []
    for vw in (5, 10, 14):
        for q in (0.40, 0.50, 0.55):
            for rf in (14, 21):
                for tw in (750, 1000, 1500):
                    configs.append(
                        EvalConfig(
                            label=f"hyb_vw{vw}_q{q:.2f}_rf{rf}_tw{tw}",
                            family="hybrid",
                            vol_window=vw,
                            quantile=q,
                            retrain_freq=rf,
                            train_window=tw,
                        )
                    )
    return configs


def _baseline_config() -> EvalConfig:
    return EvalConfig(
        label=BASELINE_LABEL,
        family="hybrid",
        vol_window=10,
        quantile=0.70,
        retrain_freq=21,
        train_window=1000,
    )


# --------------------------------------------------------------------------
# Summary / scoring
# --------------------------------------------------------------------------


def _summarize_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """One row per config: multi-seed avg median Sharpe etc."""
    per_seed = (
        df.groupby(["label", "family", "seed"])
        .agg(
            median_sharpe=("sharpe", "median"),
            mean_sharpe=("sharpe", "mean"),
            median_cagr=("cagr", "median"),
            mean_mdd=("max_dd", "mean"),
            pos_count=("cagr", lambda s: (s > 0).sum()),
        )
        .reset_index()
    )
    agg = (
        per_seed.groupby(["label", "family"])
        .agg(
            avg_median_sharpe=("median_sharpe", "mean"),
            stderr_median_sharpe=("median_sharpe", "sem"),
            min_median_sharpe=("median_sharpe", "min"),
            max_median_sharpe=("median_sharpe", "max"),
            avg_mean_sharpe=("mean_sharpe", "mean"),
            avg_median_cagr=("median_cagr", "mean"),
            avg_mean_mdd=("mean_mdd", "mean"),
            avg_pos_count=("pos_count", "mean"),
        )
        .reset_index()
        .sort_values("avg_median_sharpe", ascending=False)
    )
    # Merge back the per-config metadata (vol_window, quantile, etc.)
    meta = df.groupby("label").first().reset_index()[
        ["label", "vol_window", "quantile", "retrain_freq", "train_window", "vt_annual", "vt_max_scale"]
    ]
    agg = agg.merge(meta, on="label", how="left")
    return agg


def _dsr_verdict(sharpe: float, n_trials: int, n_obs: int) -> tuple[float, str]:
    dsr = deflated_sharpe_ratio(sharpe, n_trials, n_obs)
    if dsr >= 0.95:
        verdict = f"survives DSR at n_trials={n_trials} (DSR={dsr:.4f})"
    else:
        verdict = f"FAILS DSR at n_trials={n_trials} (DSR={dsr:.4f})"
    return dsr, verdict


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]

    print(f"{SYMBOL}: {len(prices)} bars")

    all_rows: list[pd.DataFrame] = []

    # ==================================================================
    # TIER 1 — Pure vol filter, 75 configs × 5 seeds
    # ==================================================================
    print("\n" + "=" * 72)
    print("TIER 1 — Pure vol filter sweep (75 × 5 seeds)")
    print("=" * 72)
    t1_grid = _tier1_vol_filter_grid()
    t1_rows: list[pd.DataFrame] = []
    t0 = time.time()
    for i, ec in enumerate(t1_grid, start=1):
        print(f"  [{i}/{len(t1_grid)}] {ec.label}  (elapsed {time.time()-t0:.0f}s)")
        t1_rows.append(_evaluate_config(ec, prices, EXPLORE_SEEDS))
    t1_df = pd.concat(t1_rows, ignore_index=True)
    t1_df["tier"] = 1
    all_rows.append(t1_df)

    t1_summary = _summarize_by_config(t1_df)
    print("\nTier 1 top 5:")
    print(t1_summary.head(5)[
        ["label", "avg_median_sharpe", "stderr_median_sharpe",
         "min_median_sharpe", "max_median_sharpe"]
    ].to_string(index=False))

    best_t1_label = t1_summary.iloc[0]["label"]
    best_t1_cfg = next(c for c in t1_grid if c.label == best_t1_label)
    print(f"\nTier 1 winner: {best_t1_label} "
          f"(avg median Sharpe = {t1_summary.iloc[0]['avg_median_sharpe']:.3f})")

    # ==================================================================
    # TIER 2 — Vol filter + vol target overlay on Tier 1 winner
    # ==================================================================
    print("\n" + "=" * 72)
    print("TIER 2 — Vol filter + vol target overlay (15 × 5 seeds)")
    print("=" * 72)
    t2_grid = _tier2_vol_filter_vt_grid(best_t1_cfg)
    t2_rows: list[pd.DataFrame] = []
    t0 = time.time()
    for i, ec in enumerate(t2_grid, start=1):
        print(f"  [{i}/{len(t2_grid)}] {ec.label}  (elapsed {time.time()-t0:.0f}s)")
        t2_rows.append(_evaluate_config(ec, prices, EXPLORE_SEEDS))
    t2_df = pd.concat(t2_rows, ignore_index=True)
    t2_df["tier"] = 2
    all_rows.append(t2_df)

    t2_summary = _summarize_by_config(t2_df)
    print("\nTier 2 top 5:")
    print(t2_summary.head(5)[
        ["label", "avg_median_sharpe", "stderr_median_sharpe"]
    ].to_string(index=False))

    # ==================================================================
    # TIER 3 — Hybrid with wider parameter axes
    # ==================================================================
    print("\n" + "=" * 72)
    print("TIER 3 — Hybrid wide grid (54 × 5 seeds)")
    print("=" * 72)
    t3_grid = _tier3_hybrid_grid()
    t3_rows: list[pd.DataFrame] = []
    t0 = time.time()
    for i, ec in enumerate(t3_grid, start=1):
        print(f"  [{i}/{len(t3_grid)}] {ec.label}  (elapsed {time.time()-t0:.0f}s)")
        t3_rows.append(_evaluate_config(ec, prices, EXPLORE_SEEDS))
    t3_df = pd.concat(t3_rows, ignore_index=True)
    t3_df["tier"] = 3
    all_rows.append(t3_df)

    t3_summary = _summarize_by_config(t3_df)
    print("\nTier 3 top 5:")
    print(t3_summary.head(5)[
        ["label", "avg_median_sharpe", "stderr_median_sharpe"]
    ].to_string(index=False))

    # ==================================================================
    # TIER 4 — 10-seed confirmation for top candidates + baseline control
    # ==================================================================
    print("\n" + "=" * 72)
    print("TIER 4 — 10-seed confirmation (top-5 combined + baseline)")
    print("=" * 72)

    # Combine top-1 from each tier + a couple of runners-up + baseline
    combined_summary = pd.concat(
        [t1_summary, t2_summary, t3_summary], ignore_index=True
    ).sort_values("avg_median_sharpe", ascending=False)
    top5_labels = list(combined_summary.head(5)["label"])
    print("\nTop 5 candidates across all tiers:")
    for label in top5_labels:
        row = combined_summary[combined_summary["label"] == label].iloc[0]
        print(f"  {label}: avg={row['avg_median_sharpe']:.3f}  "
              f"min_seed={row['min_median_sharpe']:.3f}")

    # Look up the EvalConfig objects for the top 5
    all_grids = t1_grid + t2_grid + t3_grid
    t4_configs = [next(c for c in all_grids if c.label == lbl) for lbl in top5_labels]
    t4_configs.append(_baseline_config())

    t4_rows: list[pd.DataFrame] = []
    t0 = time.time()
    for i, ec in enumerate(t4_configs, start=1):
        print(f"  [{i}/{len(t4_configs)}] CONFIRM {ec.label} at 10 seeds  "
              f"(elapsed {time.time()-t0:.0f}s)")
        t4_rows.append(_evaluate_config(ec, prices, CONFIRM_SEEDS))
    t4_df = pd.concat(t4_rows, ignore_index=True)
    t4_df["tier"] = 4
    all_rows.append(t4_df)

    t4_summary = _summarize_by_config(t4_df)
    print("\n" + "=" * 72)
    print("TIER 4 — 10-seed final ranking")
    print("=" * 72)
    print(t4_summary[
        ["label", "avg_median_sharpe", "stderr_median_sharpe",
         "min_median_sharpe", "max_median_sharpe", "avg_mean_mdd"]
    ].to_string(index=False))

    # DSR correction — number of trials is the total across all tiers
    sweep_n_trials = len(t1_grid) + len(t2_grid) + len(t3_grid)  # exploration tiers
    project_n_trials = LEGACY_PROJECT_TRIALS + sweep_n_trials
    n_obs_per_window = 126  # approx bars per window

    winner_row = t4_summary.iloc[0]
    winner_label = winner_row["label"]
    winner_sharpe = winner_row["avg_median_sharpe"]
    baseline_row = t4_summary[t4_summary["label"] == BASELINE_LABEL].iloc[0]
    baseline_sharpe = baseline_row["avg_median_sharpe"]

    print("\n" + "=" * 72)
    print("DSR-corrected verdict")
    print("=" * 72)
    print(f"  Sweep trial count: {sweep_n_trials}")
    print(f"  Project-level trial count: {project_n_trials}")
    print(f"  Baseline avg median Sharpe: {baseline_sharpe:.3f}")
    print(f"  Winner ({winner_label}) avg median Sharpe: {winner_sharpe:.3f}")
    print(f"  Delta: {winner_sharpe - baseline_sharpe:+.3f}")
    dsr_sweep, v_sweep = _dsr_verdict(winner_sharpe, sweep_n_trials, n_obs_per_window * N_WINDOWS)
    dsr_project, v_project = _dsr_verdict(winner_sharpe, project_n_trials, n_obs_per_window * N_WINDOWS)
    print(f"  {v_sweep}")
    print(f"  {v_project}")

    if (winner_sharpe > baseline_sharpe + 0.10
            and winner_row["min_median_sharpe"] > baseline_row["min_median_sharpe"]):
        print("\n  >>> WINNER CLEARS MATERIALITY THRESHOLD (+0.10 avg AND dominates on min seed)")
    else:
        print("\n  >>> No config materially beats baseline. "
              "Parameter plateau confirmed.")

    # ==================================================================
    # Persist
    # ==================================================================
    full_df = pd.concat(all_rows, ignore_index=True)
    out_parquet = Path(__file__).parent / "data" / "explore_improvements.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}  ({len(full_df)} rows)")

    out_md = Path(__file__).parent / "data" / "explore_improvements.md"
    lines = [
        "# explore_improvements.py — large-grid search for material improvements",
        "",
        "**Objective**: find a configuration that materially beats the H-Vol "
        "@ q=0.70 production baseline (multi-seed avg Sharpe ≈ 0.78 ± 0.08).",
        "",
        "**Discipline**: 5 seeds during exploration, 10 seeds during final "
        "candidate confirmation, all seeds pre-registered. Multi-testing "
        "correction applied via project-level DSR.",
        "",
        f"**Sweep trial count**: {sweep_n_trials}  (75 Tier-1 + 15 Tier-2 + 54 Tier-3)",
        f"**Project-level trial count (including legacy)**: {project_n_trials}",
        "",
        "## Tier 1 — Pure vol filter (5 seeds, 75 configs)",
        "",
        "Top 10:",
        "",
        "| label | vol_window | q | rf | avg Sh | stderr | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t1_summary.head(10).iterrows():
        lines.append(
            f"| {r['label']} | {int(r['vol_window'])} | "
            f"{r['quantile']:.2f} | {int(r['retrain_freq'])} | "
            f"{r['avg_median_sharpe']:+.3f} | {r['stderr_median_sharpe']:.3f} | "
            f"{r['min_median_sharpe']:+.3f} | {r['max_median_sharpe']:+.3f} |"
        )
    lines += [
        "",
        "## Tier 2 — Vol filter + vol target overlay (5 seeds, 15 configs)",
        "",
        "| label | vt_annual | vt_max | avg Sh | stderr |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in t2_summary.head(10).iterrows():
        lines.append(
            f"| {r['label']} | {r['vt_annual']:.2f} | "
            f"{r['vt_max_scale']:.1f} | "
            f"{r['avg_median_sharpe']:+.3f} | {r['stderr_median_sharpe']:.3f} |"
        )
    lines += [
        "",
        "## Tier 3 — Hybrid wider grid (5 seeds, 54 configs)",
        "",
        "Top 10:",
        "",
        "| label | vol_window | q | rf | tw | avg Sh | stderr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t3_summary.head(10).iterrows():
        lines.append(
            f"| {r['label']} | {int(r['vol_window'])} | "
            f"{r['quantile']:.2f} | {int(r['retrain_freq'])} | "
            f"{int(r['train_window'])} | "
            f"{r['avg_median_sharpe']:+.3f} | {r['stderr_median_sharpe']:.3f} |"
        )
    lines += [
        "",
        "## Tier 4 — 10-seed confirmation (top candidates + production baseline)",
        "",
        "| label | avg Sh | stderr | min seed | max seed | mean MDD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in t4_summary.iterrows():
        lines.append(
            f"| {r['label']} | {r['avg_median_sharpe']:+.3f} | "
            f"{r['stderr_median_sharpe']:.3f} | {r['min_median_sharpe']:+.3f} | "
            f"{r['max_median_sharpe']:+.3f} | {r['avg_mean_mdd']:+.1%} |"
        )
    lines += [
        "",
        "## DSR correction",
        "",
        f"- Sweep n_trials = {sweep_n_trials}",
        f"- Project n_trials = {project_n_trials}",
        f"- Baseline Sharpe = {baseline_sharpe:+.3f}",
        f"- Winner Sharpe = {winner_sharpe:+.3f}",
        f"- Winner label = `{winner_label}`",
        f"- DSR at sweep n_trials: {dsr_sweep:.4f}",
        f"- DSR at project n_trials: {dsr_project:.4f}",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
