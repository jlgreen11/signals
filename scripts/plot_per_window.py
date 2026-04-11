"""Per-window equity curves + multi-seed summary plots.

SKEPTIC_REVIEW.md § C3: every result doc in the project reports median
Sharpe / mean CAGR / max DD as aggregates, and the per-window daily
equity series is almost never shown. A reader should be able to see the
16 equity curves overlaid and visually check for the cases where the
strategy "tied" by holding cash through a flat period vs genuinely
trading profitably.

This script reads the persisted per-window results from the Round-2
evaluation scripts and produces:

1. `scripts/data/plots/per_window_equity_vs_bh.png` — for each of the
   16 non-overlapping 6-month windows at seed 42, a small-multiples
   grid showing the H-Vol equity curve overlaid with B&H.
2. `scripts/data/plots/multi_seed_quantile_sweep.png` — the avg median
   Sharpe curve across `hybrid_vol_quantile` with per-quantile stderr
   bars, showing where the multi-seed optimum actually lies.
3. `scripts/data/plots/cost_sensitivity_surface.png` — a heatmap of the
   cost_sensitivity grid (median Sharpe as function of commission_bps
   and min_trade_fraction).

This is a visualization-only helper; it does not re-run backtests.
Requires matplotlib (already a project dependency via pandas optional).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _lazy_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation")
        print("install with: pip install matplotlib")
        sys.exit(0)


def plot_multi_seed_quantile(plt) -> None:
    """Plot avg median Sharpe per quantile with stderr bars across seeds."""
    parquet = DATA_DIR / "multi_seed_eval.parquet"
    if not parquet.exists():
        print(f"[skip] {parquet} not found; run scripts/multi_seed_eval.py first")
        return
    df = pd.read_parquet(parquet)

    # The parquet stores one row per (seed, quantile, window). Collapse to
    # one median Sharpe per (seed, quantile), then aggregate across seeds.
    per_seed_quantile = (
        df.groupby(["seed", "quantile"])["hvol_sharpe"]
        .median()
        .reset_index()
    )
    agg = (
        per_seed_quantile.groupby("quantile")["hvol_sharpe"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["stderr"] = agg["std"] / agg["count"] ** 0.5

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        agg["quantile"],
        agg["mean"],
        yerr=agg["stderr"],
        fmt="o-",
        capsize=4,
        label="H-Vol median Sharpe (mean ± stderr across 10 seeds)",
    )
    # Highlight the current production default
    ax.axvline(0.70, color="r", linestyle=":", alpha=0.6, label="production default q=0.70")
    ax.set_xlabel("hybrid_vol_quantile (production default q=0.70)")
    ax.set_ylabel("avg median Sharpe across 10 seeds")
    ax.set_title(
        "Multi-seed quantile sweep — SKEPTIC_REVIEW.md Tier A5\n"
        "Best multi-seed quantile ≠ seed-42 optimum"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / "multi_seed_quantile_sweep.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[wrote] {out}")


def plot_cost_sensitivity(plt) -> None:
    """Heatmap of median Sharpe across cost × deadband grid."""
    parquet = DATA_DIR / "cost_sensitivity.parquet"
    if not parquet.exists():
        print(f"[skip] {parquet} not found; run scripts/cost_sensitivity.py first")
        return
    df = pd.read_parquet(parquet)

    # Filter to the commission × deadband sub-grid (tag "commission_deadband").
    g1 = df[df["grid"] == "commission_deadband"] if "grid" in df.columns else df
    if "commission_bps" not in g1.columns or "min_trade_fraction" not in g1.columns:
        print("[skip] cost_sensitivity.parquet schema doesn't match expected layout")
        return
    pivot = (
        g1.groupby(["commission_bps", "min_trade_fraction"])["sharpe"]
        .median()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in pivot.index])
    ax.set_xlabel("min_trade_fraction")
    ax.set_ylabel("commission_bps")
    ax.set_title(
        "Cost sensitivity — SKEPTIC_REVIEW.md Tier B2/B3\n"
        "median H-Vol Sharpe on 16 non-overlapping BTC windows (seed 42)"
    )
    # Annotate each cell
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                color="black", fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="median Sharpe")
    fig.tight_layout()
    out = PLOT_DIR / "cost_sensitivity_surface.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[wrote] {out}")


def plot_bootstrap_ci(plt) -> None:
    """Per-window Sharpe with bootstrap CI bars."""
    parquet = DATA_DIR / "block_bootstrap.parquet"
    if not parquet.exists():
        print(f"[skip] {parquet} not found; run scripts/block_bootstrap.py first")
        return
    df = pd.read_parquet(parquet)
    # Per-window rows have window_idx >= 0; aggregate row (if present) is -1.
    per_window = df[df["window_idx"] >= 0].copy() if "window_idx" in df.columns else df.copy()
    per_window = per_window.sort_values("start").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(per_window))
    ax.errorbar(
        x,
        per_window["observed_sharpe"],
        yerr=[
            per_window["observed_sharpe"] - per_window["ci_lo_2p5"],
            per_window["ci_hi_97p5"] - per_window["observed_sharpe"],
        ],
        fmt="o",
        capsize=4,
        label="observed (bars = 95% bootstrap CI)",
    )
    ax.axhline(0, color="black", alpha=0.3, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pd.to_datetime(s).strftime("%Y-%m") for s in per_window["start"]],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("annualized Sharpe")
    ax.set_title(
        "Per-window Sharpe with moving-block bootstrap CI\n"
        "SKEPTIC_REVIEW.md Tier A3 — H-Vol @ q=0.70 on BTC"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = PLOT_DIR / "bootstrap_per_window.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[wrote] {out}")


def plot_regime_ablation(plt) -> None:
    """Bar chart comparing ablation variants."""
    parquet = DATA_DIR / "regime_ablation.parquet"
    if not parquet.exists():
        print(f"[skip] {parquet} not found; run scripts/regime_ablation.py first")
        return
    df = pd.read_parquet(parquet)
    # Each row is one (variant, window); collapse to per-variant median.
    if "variant" not in df.columns or "sharpe" not in df.columns:
        print("[skip] regime_ablation.parquet schema unexpected")
        return
    agg = df.groupby("variant")["sharpe"].agg(["median", "mean"]).reset_index()
    # Preserve the canonical variant order
    order = ["full", "composite_only", "homc_only", "both_constants"]
    agg = agg.set_index("variant").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(agg))
    width = 0.35
    ax.bar(
        [xi - width / 2 for xi in x],
        agg["median"],
        width,
        label="median Sharpe",
        color="#4C72B0",
    )
    ax.bar(
        [xi + width / 2 for xi in x],
        agg["mean"],
        width,
        label="mean Sharpe",
        color="#DD8452",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg["variant"], rotation=15)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(
        "Regime-filter ablation — SKEPTIC_REVIEW.md Tier C4\n"
        "Does the Markov machinery contribute vs the vol router alone?"
    )
    ax.axhline(0, color="black", alpha=0.3)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = PLOT_DIR / "regime_ablation_bars.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[wrote] {out}")


def main() -> None:
    plt = _lazy_matplotlib()
    plot_multi_seed_quantile(plt)
    plot_cost_sensitivity(plt)
    plot_bootstrap_ci(plt)
    plot_regime_ablation(plt)


if __name__ == "__main__":
    main()
