"""Round-4 improvement #4 — widen signal hysteresis.

Rationale: `scripts/cost_sensitivity.py` showed `min_trade_fraction` is
inert at the current production params (0.05-0.30 all produce the same
median Sharpe). So the deadband on trade *size* isn't the lever. But
the deadband between the BUY and SELL signal thresholds — the hysteresis
around regime boundaries — may be. Current production:

    buy_threshold_bps  =  25
    sell_threshold_bps = -35

dead zone = 60 bps. Widening to ~100 bps could reduce whipsaw around
regime crossings (the vol router in particular flips on single-bar vol
moves). This is a small expected-impact experiment (±0.05 Sharpe) that
should only run after #1 and #2 in case those change the baseline.

Pre-registered grid (DO NOT EXPAND per D1):

    (buy_bps, sell_bps) ∈ {(25, -35), (30, -50), (40, -60)}
    — 3 pairs × 10 seeds = 30 configs

The first pair is the current production default (control). The other
two widen the dead zone while keeping the asymmetry (sell threshold
stays more negative than buy is positive, because of BTC's asymmetric
drawdown risk).

Success criterion: Δ ≥ 0.05 over the production baseline at 10-seed
multi-seed avg median Sharpe AND min-seed dominance.
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

# Pre-registered grid: (buy_bps, sell_bps) pairs
THRESHOLD_PAIRS = (
    (25.0, -35.0),   # production default — control
    (30.0, -50.0),   # 80 bps dead zone
    (40.0, -60.0),   # 100 bps dead zone
)


@dataclass
class Config:
    label: str
    buy_bps: float
    sell_bps: float


def _build_grid() -> list[Config]:
    return [
        Config(
            label=f"hyst_buy{int(b)}_sell{int(s)}",
            buy_bps=b,
            sell_bps=s,
        )
        for b, s in THRESHOLD_PAIRS
    ]


def _make_cfg(c: Config) -> BacktestConfig:
    base = dict(BTC_HYBRID_PRODUCTION)
    base["risk_free_rate"] = historical_usd_rate("2018-2024")
    base["buy_threshold_bps"] = c.buy_bps
    base["sell_threshold_bps"] = c.sell_bps
    return BacktestConfig(**base)


def _run_one_window(
    cfg: BacktestConfig, prices: pd.DataFrame, start_i: int, end_i: int
) -> tuple[float, float, float]:
    slice_start = max(0, start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD)
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]
    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=SYMBOL)
    except Exception:
        return 0.0, 0.0, 0.0
    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return 0.0, 0.0, 0.0
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    m = compute_metrics(
        eq_rebased,
        [],
        risk_free_rate=cfg.risk_free_rate,
        periods_per_year=cfg.periods_per_year,
    )
    return m.sharpe, m.cagr, m.max_drawdown


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars")

    grid = _build_grid()
    print(f"Pre-registered grid: {len(grid)} pairs × {len(SEEDS)} seeds")

    t0 = time.time()
    all_rows: list[pd.DataFrame] = []
    for i, c in enumerate(grid, start=1):
        print(f"\n[{i}/{len(grid)}] {c.label}")
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
                sharpe, cagr, mdd = _run_one_window(cfg, prices, start_i, end_i)
                rows.append({
                    "label": c.label,
                    "buy_bps": c.buy_bps,
                    "sell_bps": c.sell_bps,
                    "seed": seed,
                    "window_idx": w,
                    "start": prices.index[start_i],
                    "end": prices.index[end_i - 1],
                    "sharpe": sharpe,
                    "cagr": cagr,
                    "max_dd": mdd,
                })
        df = pd.DataFrame(rows)
        all_rows.append(df)
        per_seed = df.groupby("seed")["sharpe"].median()
        elapsed = time.time() - t0
        print(f"  avg {per_seed.mean():+.3f} ± {per_seed.sem():.3f}  "
              f"(min {per_seed.min():+.3f}, max {per_seed.max():+.3f})  "
              f"elapsed {elapsed:.0f}s")

    full_df = pd.concat(all_rows, ignore_index=True)
    out_parquet = Path(__file__).parent / "data" / "hysteresis_sweep.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}")

    per_seed = (
        full_df.groupby(["label", "buy_bps", "sell_bps", "seed"])["sharpe"]
        .median()
        .reset_index()
    )
    agg = (
        per_seed.groupby(["label", "buy_bps", "sell_bps"])["sharpe"]
        .agg(["mean", "sem", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print("\n" + "=" * 80)
    print("Multi-seed ranking — hysteresis widening")
    print("=" * 80)
    print(agg.to_string(index=False))

    baseline_row = agg[agg["label"] == "hyst_buy25_sell-35"].iloc[0]
    winner = agg.iloc[0]
    delta = winner["mean"] - baseline_row["mean"]
    print(f"\nBaseline (buy=25, sell=-35):  {baseline_row['mean']:+.3f}")
    print(f"Winner ({winner['label']}):    {winner['mean']:+.3f}  "
          f"(Δ = {delta:+.3f})")
    materiality_ok = delta >= 0.05
    min_seed_ok = winner["min"] > baseline_row["min"]
    success = materiality_ok and min_seed_ok and winner["label"] != "hyst_buy25_sell-35"
    print(f"Materiality (Δ ≥ 0.05): {'PASS' if materiality_ok else 'FAIL'}")
    print(f"Min-seed dominance:     {'PASS' if min_seed_ok else 'FAIL'}")
    print(f"Overall: {'SUCCESS' if success else 'NO CHANGE'}")

    out_md = Path(__file__).parent / "HYSTERESIS_SWEEP_RESULTS.md"
    lines = [
        "# Round-4 #4 — signal hysteresis widening",
        "",
        "**Run date**: 2026-04-11",
        "**Script**: `scripts/hysteresis_sweep.py`",
        "**Test parameters**:",
        "",
        "- Base config: `BTC_HYBRID_PRODUCTION`",
        f"- Threshold pairs (buy_bps, sell_bps): {list(THRESHOLD_PAIRS)}",
        f"- Seeds: {SEEDS}",
        f"- Windows: {N_WINDOWS} non-overlapping 6-month per seed",
        "",
        "## Multi-seed ranking",
        "",
        "| label | buy_bps | sell_bps | avg Sharpe | stderr | min seed | max seed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| `{r['label']}` | {r['buy_bps']:+.0f} | {r['sell_bps']:+.0f} | "
            f"{r['mean']:+.3f} | {r['sem']:.3f} | "
            f"{r['min']:+.3f} | {r['max']:+.3f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- Delta vs production: {delta:+.3f}",
        f"- Materiality (Δ ≥ 0.05): **{'PASS' if materiality_ok else 'FAIL'}**",
        f"- Min-seed dominance: **{'PASS' if min_seed_ok else 'FAIL'}**",
        f"- **Overall: {'SUCCESS' if success else 'NO CHANGE'}**",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
