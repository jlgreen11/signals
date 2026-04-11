"""Broad comparison across every strategy the project has evaluated.

Round 5 / Round 6 wrap-up. Aggregates multi-seed Sharpe numbers from the
persisted parquet files in `scripts/data/` rather than re-running
anything. Every input is already annualization-corrected by the Round-5
sweep-fix pass (commits 732111e + 603b96c).

For each strategy, reports:
  - asset class (BTC / SP / BTC+SP / 4-asset)
  - annualization convention (365/yr for crypto, 252/yr for equities
    and equity-calendar portfolios)
  - multi-seed avg median Sharpe ± stderr
  - min seed Sharpe, max seed Sharpe
  - source parquet

Output:
  scripts/BROAD_COMPARISON.md          — narrative summary + table
  scripts/data/broad_comparison.parquet — raw aggregation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

DATA = Path(__file__).parent / "data"


def _agg_per_seed(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str = "sharpe",
) -> pd.DataFrame:
    """Collapse (group × seed × window) → (group × seed) median → (group) aggregate."""
    per_seed = (
        df.groupby([*group_cols, "seed"])[metric_col]
        .median()
        .reset_index()
    )
    agg = (
        per_seed.groupby(group_cols)[metric_col]
        .agg(["mean", "sem", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_sharpe", "sem": "stderr",
                         "min": "min_seed", "max": "max_seed",
                         "count": "n_seeds"})
    )
    return agg


def _ann_label(calendar: str) -> str:
    return {"btc": "365/yr", "sp": "252/yr", "portfolio": "252/yr"}[calendar]


def _load_trivial_baselines() -> list[dict]:
    """Pull BTC B&H, Trend200, DualMA, VolFilterOnly, H-Vol hybrid from
    scripts/data/trivial_baselines_btc.parquet.

    The parquet stores one row per window with each strategy as its
    own `<strat>_sharpe` column (wide format). We reshape to long
    and then aggregate. Single seed (42) by design — N=1 windows-only.
    """
    path = DATA / "trivial_baselines_btc.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    strategy_mapping = {
        "bh": "B&H",
        "volonly": "VolFilterOnly",
        "trend200": "TrendFilter(200)",
        "gcross": "DualMA(50,200)",
        "hvol": "H-Vol hybrid (legacy defaults)",
    }
    out = []
    for col_prefix, label in strategy_mapping.items():
        sharpe_col = f"{col_prefix}_sharpe"
        if sharpe_col not in df.columns:
            continue
        series = df[sharpe_col].dropna()
        if series.empty:
            continue
        median = float(series.median())
        mn = float(series.min())
        mx = float(series.max())
        out.append({
            "strategy": label,
            "family": "BTC / single-asset",
            "annualization": "365/yr",
            # For single-seed sources we report the cross-window MEDIAN
            # as the headline (matching the convention used throughout
            # the project). stderr is set to 0 since we only have one seed.
            "avg_sharpe": median,
            "stderr": 0.0,
            "min_seed": mn,
            "max_seed": mx,
            "n_seeds": 1,
            "source": "trivial_baselines_btc.parquet (seed=42, window-median)",
        })
    return out


def _load_multi_seed() -> list[dict]:
    """BTC H-Vol hybrid per-quantile from multi_seed_eval.parquet.
    10 seeds × 6 quantiles on legacy defaults (rf=21, tw=1000)."""
    path = DATA / "multi_seed_eval.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    # Compute H-Vol per-quantile multi-seed averages
    hvol_cols = [c for c in df.columns if c.startswith("hvol_sharpe")]
    if hvol_cols:
        sharpe_col = hvol_cols[0]
    elif "hvol_sharpe" in df.columns:
        sharpe_col = "hvol_sharpe"
    else:
        return []
    per_seed = (
        df.groupby(["quantile", "seed"])[sharpe_col]
        .median()
        .reset_index()
    )
    agg = (
        per_seed.groupby("quantile")[sharpe_col]
        .agg(["mean", "sem", "min", "max", "count"])
        .reset_index()
    )
    out = []
    for _, r in agg.iterrows():
        out.append({
            "strategy": f"H-Vol q={r['quantile']:.2f} (legacy defaults)",
            "family": "BTC / single-asset",
            "annualization": "365/yr",
            "avg_sharpe": float(r["mean"]),
            "stderr": float(r["sem"]),
            "min_seed": float(r["min"]),
            "max_seed": float(r["max"]),
            "n_seeds": int(r["count"]),
            "source": "multi_seed_eval.parquet",
        })
    return out


def _load_vol_target() -> list[dict]:
    """BTC production bundle (baseline) + vol_target variants."""
    path = DATA / "vol_target_sweep.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    agg = _agg_per_seed(df, ["label"])
    out = []
    for _, r in agg.iterrows():
        # Rename the baseline row to be descriptive
        label = r["label"]
        if label == "baseline_no_vol_target":
            name = "BTC_HYBRID_PRODUCTION (q=0.50 rf=14 tw=750)"
        else:
            name = f"BTC hybrid + {label}"
        out.append({
            "strategy": name,
            "family": "BTC / single-asset",
            "annualization": "365/yr",
            "avg_sharpe": float(r["avg_sharpe"]),
            "stderr": float(r["stderr"]) if pd.notna(r["stderr"]) else 0.0,
            "min_seed": float(r["min_seed"]),
            "max_seed": float(r["max_seed"]),
            "n_seeds": int(r["n_seeds"]),
            "source": "vol_target_sweep.parquet",
        })
    return out


def _load_regime_ablation() -> list[dict]:
    path = DATA / "regime_ablation.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    # Single seed — use window-level median as the "multi-seed avg"
    if "seed" not in df.columns:
        df["seed"] = 42
    agg = _agg_per_seed(df, ["variant"])
    out = []
    for _, r in agg.iterrows():
        out.append({
            "strategy": f"Ablation: {r['variant']}",
            "family": "BTC / single-asset",
            "annualization": "365/yr",
            "avg_sharpe": float(r["avg_sharpe"]),
            "stderr": float(r["stderr"]) if pd.notna(r["stderr"]) else 0.0,
            "min_seed": float(r["min_seed"]),
            "max_seed": float(r["max_seed"]),
            "n_seeds": int(r["n_seeds"]),
            "source": "regime_ablation.parquet",
        })
    return out


def _load_confirm_winners() -> list[dict]:
    """10-seed confirmation of q=0.50 vs q=0.70 vs pure vol filter."""
    path = DATA / "confirm_winners.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    agg = _agg_per_seed(df, ["strategy"])
    out = []
    for _, r in agg.iterrows():
        out.append({
            "strategy": f"{r['strategy']} (10-seed confirm)",
            "family": "BTC / single-asset",
            "annualization": "365/yr",
            "avg_sharpe": float(r["avg_sharpe"]),
            "stderr": float(r["stderr"]) if pd.notna(r["stderr"]) else 0.0,
            "min_seed": float(r["min_seed"]),
            "max_seed": float(r["max_seed"]),
            "n_seeds": int(r["n_seeds"]),
            "source": "confirm_winners.parquet",
        })
    return out


def _load_risk_parity() -> list[dict]:
    """4-asset portfolio variants."""
    path = DATA / "risk_parity_4asset.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    agg = _agg_per_seed(df, ["label", "weighting"])
    out = []
    for _, r in agg.iterrows():
        out.append({
            "strategy": f"4-asset {r['weighting']} (BTC+SP+TLT+GLD)",
            "family": "Multi-asset portfolio",
            "annualization": "252/yr",
            "avg_sharpe": float(r["avg_sharpe"]),
            "stderr": float(r["stderr"]) if pd.notna(r["stderr"]) else 0.0,
            "min_seed": float(r["min_seed"]),
            "max_seed": float(r["max_seed"]),
            "n_seeds": int(r["n_seeds"]),
            "source": "risk_parity_4asset.parquet",
        })
    return out


def _load_absolute_encoder() -> list[dict]:
    """Markov-closure Experiment 1 — absolute-granularity HOMC."""
    path = DATA / "absolute_encoder_eval.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    agg = _agg_per_seed(df, ["label"])
    # Only report the top row (winner) to keep the comparison table
    # compact. Full grid is in scripts/ABSOLUTE_ENCODER_RESULTS.md.
    top = agg.sort_values("avg_sharpe", ascending=False).head(1)
    out = []
    for _, r in top.iterrows():
        out.append({
            "strategy": f"HOMC absolute-encoder winner ({r['label']})",
            "family": "BTC / Markov sunset",
            "annualization": "365/yr",
            "avg_sharpe": float(r["avg_sharpe"]),
            "stderr": float(r["stderr"]) if pd.notna(r["stderr"]) else 0.0,
            "min_seed": float(r["min_seed"]),
            "max_seed": float(r["max_seed"]),
            "n_seeds": int(r["n_seeds"]),
            "source": "absolute_encoder_eval.parquet",
        })
    return out


def _load_rule_based() -> list[dict]:
    """Markov-closure Experiment 2 — rule-based signals."""
    path = DATA / "rule_based_eval.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    agg = _agg_per_seed(df, ["label"])
    top = agg.sort_values("avg_sharpe", ascending=False).head(1)
    out = []
    for _, r in top.iterrows():
        out.append({
            "strategy": f"Rule-based HOMC winner ({r['label']})",
            "family": "BTC / Markov sunset",
            "annualization": "365/yr",
            "avg_sharpe": float(r["avg_sharpe"]),
            "stderr": float(r["stderr"]) if pd.notna(r["stderr"]) else 0.0,
            "min_seed": float(r["min_seed"]),
            "max_seed": float(r["max_seed"]),
            "n_seeds": int(r["n_seeds"]),
            "source": "rule_based_eval.parquet",
        })
    return out


def main() -> None:
    rows: list[dict] = []
    for loader in (
        _load_trivial_baselines,
        _load_multi_seed,
        _load_vol_target,
        _load_regime_ablation,
        _load_confirm_winners,
        _load_risk_parity,
        _load_absolute_encoder,
        _load_rule_based,
    ):
        try:
            rows.extend(loader())
        except Exception as e:
            print(f"  [warn] {loader.__name__}: {e}")

    if not rows:
        print("No data found.")
        return

    df = pd.DataFrame(rows).sort_values(
        ["family", "avg_sharpe"], ascending=[True, False]
    )

    out_parquet = DATA / "broad_comparison.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)

    # Build a plain-text table for stdout
    print("=" * 110)
    print("Broad Sharpe comparison — all strategies, correct per-asset annualization")
    print("=" * 110)
    print(
        f"{'strategy':<52}{'family':<24}{'ann':<8}"
        f"{'avg':>8}{'±se':>7}{'min':>8}{'max':>8}{'N':>4}"
    )
    print("-" * 110)
    for _, r in df.iterrows():
        print(
            f"{r['strategy'][:52]:<52}{r['family'][:24]:<24}"
            f"{r['annualization']:<8}"
            f"{r['avg_sharpe']:>+8.3f}{r['stderr']:>+7.3f}"
            f"{r['min_seed']:>+8.3f}{r['max_seed']:>+8.3f}"
            f"{r['n_seeds']:>4d}"
        )

    # Markdown output
    md_lines = [
        "# Broad Sharpe comparison",
        "",
        "Every strategy the project has evaluated, aggregated from the "
        "persisted parquets in `scripts/data/` after the Round-5 sweep-fix "
        "pass (commits 732111e + 603b96c). Every row uses the correct "
        "per-asset annualization: **365/yr for BTC-only**, **252/yr for "
        "equity-only and equity-calendar portfolios**.",
        "",
        "Ranking is within each family; rows sort by `avg_sharpe` descending.",
        "",
        "| strategy | family | annualization | avg Sharpe | stderr | min seed | max seed | N |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| `{r['strategy']}` | {r['family']} | {r['annualization']} | "
            f"{r['avg_sharpe']:+.3f} | {r['stderr']:+.3f} | "
            f"{r['min_seed']:+.3f} | {r['max_seed']:+.3f} | {r['n_seeds']} |"
        )
    md_lines += [
        "",
        "## How to read",
        "",
        "- **BTC / single-asset**: evaluated on BTC-USD at 365/yr annualization.",
        "- **BTC / Markov sunset**: the closure experiments for the Markov "
        "class — retained in the code, shown here only for reference.",
        "- **Multi-asset portfolio**: 4-asset equal-weight and risk-parity "
        "variants on the equity shared calendar at 252/yr.",
        "- **N**: number of pre-registered seeds the aggregation covers. "
        "Some sources are single-seed (N=1) — those are flagged in the "
        "source-parquet comments in `scripts/broad_comparison.py`.",
        "- **Legacy H-Vol q=0.70** measured on default rf=21/tw=1000 "
        "differs from `BTC_HYBRID_PRODUCTION (q=0.50 rf=14 tw=750)` "
        "because the Round-3 winner tunes all three parameters together.",
        "",
        "**Production recommendations** (per README.md):",
        "",
        "1. 4-asset equal-weight risk-parity basket — highest Sharpe on the "
        "equity-calendar number, strongest diversification behavior in "
        "stress windows.",
        "2. BTC alone via `BTC_HYBRID_PRODUCTION` — simpler operationally, "
        "stable multi-seed stderr, 365/yr annualization.",
        "3. ^GSPC alone — buy & hold. No active strategy in the project "
        "beats B&H on S&P across 4 model classes tested.",
        "",
    ]
    out_md = Path(__file__).parent / "BROAD_COMPARISON.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"\n[wrote] {out_parquet}")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
