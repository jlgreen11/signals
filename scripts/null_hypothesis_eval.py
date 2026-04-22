"""Null hypothesis evaluation: does the Markov chain beat a simple vol filter?

This script answers the single most important question about the signals
project: is the Markov chain machinery (composite, HOMC, hybrid routing)
doing anything that a trivial vol threshold can't?

The NaiveVolFilter model does exactly what the hybrid's vol routing does —
go flat when trailing vol exceeds a quantile threshold — but without any
Markov chain, transition matrices, or state prediction. If the hybrid
can't beat it, the entire model class is decoration.

Experiments run:
  1. HEAD-TO-HEAD: NaiveVolFilter vs Hybrid (production config) vs B&H
     across 4 seeds × N windows on BTC-USD
  2. CLEAN HOLDOUT: Tune on 2018-2022, evaluate on 2023-2024
     (the test the project has never run)
  3. COST SENSITIVITY: Sweep commission+slippage from 5 to 50 bps

Usage:
    cd /Users/jlg/claude/signals
    .venv/bin/python scripts/null_hypothesis_eval.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _window_sampler import draw_non_overlapping_starts

from signals.backtest.engine import BTC_HYBRID_PRODUCTION, BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

PERIODS_PER_YEAR = 365.0  # BTC
RISK_FREE = historical_usd_rate("2018-2024")
SIX_MONTHS = 126


@dataclass
class StrategyConfig:
    name: str
    cfg: BacktestConfig


def _build_strategies(vol_window: int = 10) -> list[StrategyConfig]:
    """Build the three contestants: vol_filter, hybrid (production), and
    hybrid (legacy q=0.75 for comparison)."""
    # The production hybrid config
    prod = dict(BTC_HYBRID_PRODUCTION)
    prod["vol_window"] = vol_window

    return [
        # NULL HYPOTHESIS: simple vol filter, same quantile as production hybrid
        StrategyConfig(
            name="vol_filter",
            cfg=BacktestConfig(
                model_type="vol_filter",
                train_window=prod.get("train_window", 750),
                retrain_freq=prod.get("retrain_freq", 14),
                vol_window=vol_window,
                vol_filter_quantile=prod.get("hybrid_vol_quantile", 0.50),
            ),
        ),
        # PRODUCTION HYBRID (the model we're testing)
        StrategyConfig(
            name="hybrid_prod",
            cfg=BacktestConfig(**prod),
        ),
        # LEGACY HYBRID (q=0.75, train_window=1000) for reference
        StrategyConfig(
            name="hybrid_legacy",
            cfg=BacktestConfig(
                model_type="hybrid",
                train_window=1000,
                retrain_freq=21,
                n_states=5,
                order=5,
                return_bins=3,
                volatility_bins=3,
                vol_window=vol_window,
                laplace_alpha=0.01,
                hybrid_routing_strategy="vol",
                hybrid_vol_quantile=0.75,
            ),
        ),
    ]


def _run_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
) -> Metrics:
    warmup_pad = 5
    slice_start = max(0, start_i - cfg.train_window - cfg.vol_window - warmup_pad)
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]

    try:
        result = BacktestEngine(cfg).run(engine_input, symbol="BTC-USD")
    except Exception as e:
        print(f"    [{cfg.model_type}] error: {e}")
        return compute_metrics(pd.Series(dtype=float), [])

    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(eq_rebased, [], risk_free_rate=RISK_FREE, periods_per_year=PERIODS_PER_YEAR)


# ── Experiment 1: Multi-seed head-to-head ────────────────────────────


def experiment_1_multiseed(prices: pd.DataFrame) -> pd.DataFrame:
    """Run vol_filter vs hybrid across 4 seeds × N windows."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Multi-seed head-to-head (NaiveVolFilter vs Hybrid)")
    print("=" * 70)

    seeds = [42, 7, 100, 999]
    vol_window = 10
    warmup_pad = 5
    train_window = 750  # production config

    min_start = train_window + vol_window + warmup_pad
    max_start = len(prices) - SIX_MONTHS - 1

    strategies = _build_strategies(vol_window)
    all_rows = []

    for seed in seeds:
        starts = draw_non_overlapping_starts(
            seed=seed,
            min_start=min_start,
            max_start=max_start,
            window_len=SIX_MONTHS,
            n_windows=16,
        )
        n_windows = len(starts)
        print(f"\n  Seed {seed}: {n_windows} windows")

        for i, start_i in enumerate(starts):
            end_i = start_i + SIX_MONTHS
            window_prices = prices.iloc[start_i:end_i]
            start_date = window_prices.index[0].date()
            end_date = window_prices.index[-1].date()

            # Buy & hold
            bh_eq = (window_prices["close"] / window_prices["close"].iloc[0]) * 10_000.0
            m_bh = compute_metrics(bh_eq, [], risk_free_rate=RISK_FREE, periods_per_year=PERIODS_PER_YEAR)

            row = {
                "seed": seed,
                "window": i + 1,
                "start": start_date,
                "end": end_date,
                "bh_sharpe": m_bh.sharpe,
                "bh_cagr": m_bh.cagr,
                "bh_mdd": m_bh.max_drawdown,
            }

            for strat in strategies:
                m = _run_on_window(strat.cfg, prices, start_i, end_i)
                row[f"{strat.name}_sharpe"] = m.sharpe
                row[f"{strat.name}_cagr"] = m.cagr
                row[f"{strat.name}_mdd"] = m.max_drawdown

            all_rows.append(row)
            print(
                f"    w{i+1:02d} {start_date}→{end_date}  "
                f"B&H={m_bh.sharpe:+.2f}  "
                f"vol_filter={row['vol_filter_sharpe']:+.2f}  "
                f"hybrid_prod={row['hybrid_prod_sharpe']:+.2f}"
            )

    df = pd.DataFrame(all_rows)

    # Summary by seed
    print("\n  ─── Per-seed summary ───")
    print(f"  {'Seed':>6}  {'B&H':>10}  {'VolFilter':>10}  {'HybridProd':>10}  {'HybridLeg':>10}  {'VF>Hyb?':>8}")
    for seed in seeds:
        s = df[df["seed"] == seed]
        bh = s["bh_sharpe"].median()
        vf = s["vol_filter_sharpe"].median()
        hp = s["hybrid_prod_sharpe"].median()
        hl = s["hybrid_legacy_sharpe"].median()
        vf_wins = (s["vol_filter_sharpe"] >= s["hybrid_prod_sharpe"]).sum()
        n = len(s)
        print(f"  {seed:>6}  {bh:>10.2f}  {vf:>10.2f}  {hp:>10.2f}  {hl:>10.2f}  {vf_wins:>3}/{n}")

    # Grand summary
    print("\n  ─── Grand summary (all seeds) ───")
    for col_prefix, label in [("bh", "Buy & Hold"), ("vol_filter", "NaiveVolFilter"),
                               ("hybrid_prod", "Hybrid (prod)"), ("hybrid_legacy", "Hybrid (legacy)")]:
        sharpes = df[f"{col_prefix}_sharpe"]
        print(
            f"  {label:<20s}  mean={sharpes.mean():+.3f}  median={sharpes.median():+.3f}  "
            f"std={sharpes.std():.3f}  min={sharpes.min():+.3f}  max={sharpes.max():+.3f}"
        )

    vf_wins_total = (df["vol_filter_sharpe"] >= df["hybrid_prod_sharpe"]).sum()
    n_total = len(df)
    print(f"\n  Vol filter beats hybrid in {vf_wins_total}/{n_total} windows ({100*vf_wins_total/n_total:.0f}%)")

    return df


# ── Experiment 2: Clean holdout ──────────────────────────────────────


def experiment_2_clean_holdout(prices: pd.DataFrame) -> dict:
    """Tune on 2018-01 → 2022-12, evaluate on 2023-01 → 2024-12.

    This is the test the project has never run: a truly pristine
    out-of-sample evaluation where no parameters were tuned on the
    holdout period.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Clean holdout (train 2018-2022, test 2023-2024)")
    print("=" * 70)

    train_end = pd.Timestamp("2022-12-31", tz="UTC")
    test_start = pd.Timestamp("2023-01-01", tz="UTC")

    train_prices = prices.loc[prices.index <= train_end]
    # For the test period, we need warmup bars before test_start
    # so the model can train on recent data during walk-forward
    test_warmup = 1000 + 10 + 5  # train_window + vol_window + pad
    test_start_idx = prices.index.get_indexer([test_start], method="nearest")[0]
    test_slice_start = max(0, test_start_idx - test_warmup)
    test_prices = prices.iloc[test_slice_start:]

    print(f"  Train: {train_prices.index[0].date()} → {train_prices.index[-1].date()} ({len(train_prices)} bars)")
    print(f"  Test:  {test_start.date()} → {prices.index[-1].date()} (eval window)")

    strategies = _build_strategies()
    results = {}

    for strat in strategies:
        # Train period
        try:
            train_result = BacktestEngine(strat.cfg).run(train_prices, symbol="BTC-USD")
            train_m = train_result.metrics
        except Exception as e:
            print(f"  [{strat.name}] train error: {e}")
            train_m = compute_metrics(pd.Series(dtype=float), [])

        # Test period (walk-forward continues from scratch on test data)
        try:
            test_result = BacktestEngine(strat.cfg).run(test_prices, symbol="BTC-USD")
            # Only evaluate equity from test_start onward
            eq = test_result.equity_curve.loc[test_result.equity_curve.index >= test_start]
            if not eq.empty and eq.iloc[0] > 0:
                eq_rebased = (eq / eq.iloc[0]) * strat.cfg.initial_cash
                test_m = compute_metrics(eq_rebased, [], risk_free_rate=RISK_FREE, periods_per_year=PERIODS_PER_YEAR)
            else:
                test_m = compute_metrics(pd.Series(dtype=float), [])
        except Exception as e:
            print(f"  [{strat.name}] test error: {e}")
            test_m = compute_metrics(pd.Series(dtype=float), [])

        results[strat.name] = {"train": train_m, "test": test_m}

    # Buy & hold on test period
    test_only = prices.loc[prices.index >= test_start]
    bh_eq = (test_only["close"] / test_only["close"].iloc[0]) * 10_000.0
    bh_m = compute_metrics(bh_eq, [], risk_free_rate=RISK_FREE, periods_per_year=PERIODS_PER_YEAR)
    results["buy_hold"] = {"train": None, "test": bh_m}

    print(f"\n  {'Strategy':<20s}  {'Train Sharpe':>12}  {'Test Sharpe':>12}  {'Test CAGR':>10}  {'Test MDD':>10}")
    print("  " + "─" * 68)
    for name, r in results.items():
        train_s = f"{r['train'].sharpe:+.3f}" if r["train"] else "   N/A"
        test_s = f"{r['test'].sharpe:+.3f}"
        test_c = f"{r['test'].cagr:+.1%}"
        test_d = f"{r['test'].max_drawdown:.1%}"
        print(f"  {name:<20s}  {train_s:>12}  {test_s:>12}  {test_c:>10}  {test_d:>10}")

    return results


# ── Experiment 3: Transaction cost sensitivity ───────────��───────────


def experiment_3_cost_sensitivity(prices: pd.DataFrame) -> pd.DataFrame:
    """Sweep commission+slippage from 5 to 50 bps for vol_filter and hybrid."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Transaction cost sensitivity")
    print("=" * 70)

    cost_levels = [5, 10, 15, 20, 30, 50]  # bps each for commission AND slippage
    seed = 42
    vol_window = 10
    warmup_pad = 5
    train_window = 750

    min_start = train_window + vol_window + warmup_pad
    max_start = len(prices) - SIX_MONTHS - 1

    starts = draw_non_overlapping_starts(
        seed=seed,
        min_start=min_start,
        max_start=max_start,
        window_len=SIX_MONTHS,
        n_windows=16,
    )

    rows = []
    for cost_bps in cost_levels:
        print(f"\n  Cost level: {cost_bps} bps commission + {cost_bps} bps slippage")

        prod = dict(BTC_HYBRID_PRODUCTION)
        prod["vol_window"] = vol_window
        prod["commission_bps"] = float(cost_bps)
        prod["slippage_bps"] = float(cost_bps)

        strats = [
            StrategyConfig(
                name="vol_filter",
                cfg=BacktestConfig(
                    model_type="vol_filter",
                    train_window=prod["train_window"],
                    retrain_freq=prod["retrain_freq"],
                    vol_window=vol_window,
                    vol_filter_quantile=prod.get("hybrid_vol_quantile", 0.50),
                    commission_bps=float(cost_bps),
                    slippage_bps=float(cost_bps),
                ),
            ),
            StrategyConfig(
                name="hybrid_prod",
                cfg=BacktestConfig(**prod),
            ),
        ]

        sharpes = {s.name: [] for s in strats}
        for start_i in starts:
            end_i = start_i + SIX_MONTHS
            for strat in strats:
                m = _run_on_window(strat.cfg, prices, start_i, end_i)
                sharpes[strat.name].append(m.sharpe)

        for name, s_list in sharpes.items():
            arr = np.array(s_list)
            rows.append({
                "cost_bps": cost_bps,
                "strategy": name,
                "mean_sharpe": arr.mean(),
                "median_sharpe": np.median(arr),
                "std_sharpe": arr.std(),
            })
            print(
                f"    {name:<15s}  mean={arr.mean():+.3f}  median={np.median(arr):+.3f}  "
                f"std={arr.std():.3f}"
            )

    df = pd.DataFrame(rows)

    print("\n  ─── Cost sensitivity summary ───")
    print(f"  {'Cost':>6}  {'VF mean':>10}  {'Hyb mean':>10}  {'VF-Hyb':>8}")
    for cost_bps in cost_levels:
        vf = df[(df["cost_bps"] == cost_bps) & (df["strategy"] == "vol_filter")]["mean_sharpe"].iloc[0]
        hp = df[(df["cost_bps"] == cost_bps) & (df["strategy"] == "hybrid_prod")]["mean_sharpe"].iloc[0]
        print(f"  {cost_bps:>4}bp  {vf:>10.3f}  {hp:>10.3f}  {vf-hp:>+8.3f}")

    return df


# ── Main ─────────────────────────────────────────────────────────────


def main():
    store = DataStore(SETTINGS.data.dir)
    prices = store.load("BTC-USD", "1d").sort_index()
    prices = prices.loc[
        (prices.index >= pd.Timestamp("2015-01-01", tz="UTC"))
        & (prices.index <= pd.Timestamp("2024-12-31", tz="UTC"))
    ]
    print(f"BTC-USD: {len(prices)} bars ({prices.index[0].date()} → {prices.index[-1].date()})")

    df1 = experiment_1_multiseed(prices)
    results2 = experiment_2_clean_holdout(prices)
    df3 = experiment_3_cost_sensitivity(prices)

    # Save results
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(exist_ok=True)
    df1.to_parquet(out_dir / "null_hypothesis_multiseed.parquet")
    df3.to_parquet(out_dir / "null_hypothesis_cost_sensitivity.parquet")

    # ── VERDICT ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    vf_mean = df1["vol_filter_sharpe"].mean()
    hp_mean = df1["hybrid_prod_sharpe"].mean()
    delta = hp_mean - vf_mean
    vf_wins = (df1["vol_filter_sharpe"] >= df1["hybrid_prod_sharpe"]).sum()
    n = len(df1)

    print(f"\n  NaiveVolFilter avg Sharpe: {vf_mean:+.3f}")
    print(f"  Hybrid (prod)  avg Sharpe: {hp_mean:+.3f}")
    print(f"  Delta (Hybrid - VolFilter): {delta:+.3f}")
    print(f"  VolFilter wins: {vf_wins}/{n} windows ({100*vf_wins/n:.0f}%)")

    if delta > 0.10:
        print("\n  >> The Markov chain adds measurable value over the naive vol filter.")
        print("  >> The hybrid is NOT just a vol filter in disguise.")
    elif delta > -0.10:
        print("\n  >> The Markov chain and the naive vol filter are statistically")
        print("  >> indistinguishable. The Markov machinery adds no value.")
        print("  >> Consider replacing the hybrid with a NaiveVolFilter.")
    else:
        print("\n  >> The Markov chain is WORSE than the naive vol filter.")
        print("  >> The model complexity is actively hurting performance.")
        print("  >> Replace the hybrid with a NaiveVolFilter immediately.")

    # Clean holdout verdict
    if "vol_filter" in results2 and "hybrid_prod" in results2:
        vf_test = results2["vol_filter"]["test"].sharpe
        hp_test = results2["hybrid_prod"]["test"].sharpe
        bh_test = results2["buy_hold"]["test"].sharpe
        print("\n  Clean holdout (2023-2024):")
        print(f"    Buy & Hold:     {bh_test:+.3f}")
        print(f"    NaiveVolFilter: {vf_test:+.3f}")
        print(f"    Hybrid (prod):  {hp_test:+.3f}")

    print()


if __name__ == "__main__":
    main()
