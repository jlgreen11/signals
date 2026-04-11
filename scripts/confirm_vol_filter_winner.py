"""10-seed confirmation of the Tier-1 vol filter winner vs production baseline.

This is a focused, fast validation run that complements (or replaces, if
explore_improvements.py takes too long) the Tier-4 confirmation inside
explore_improvements.py.

Compares three configs at 10 seeds × 16 non-overlapping windows:
  1. `vf_vw14_q0.60_rf7` — Tier 1 winner (pure vol filter, no Markov)
  2. `hvol_q0.70`         — old production default (for back-compat)
  3. `hvol_q0.50`         — new production default (Round 2 multi-seed winner)

Output:
  scripts/data/confirm_winners.parquet — raw per-window results
  scripts/data/confirm_winners.md       — multi-seed summary + DSR verdict
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import (
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
N_WINDOWS = 16
WARMUP_PAD = 5
CONFIRM_SEEDS = [42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23]

LEGACY_PROJECT_TRIALS = 1_900 + 144  # + Round-3 explore_improvements budget


def _run_vol_filter_one_window(
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    *,
    vol_window: int,
    quantile: float,
    retrain_freq: int,
    train_window: int = HOMC_TRAIN_WINDOW,
) -> tuple[float, float, float]:
    """Pure vol filter: long when vol < q-th percentile of training vol, flat otherwise.

    Returns (sharpe, cagr, mdd). Self-contained — does not use BacktestEngine.
    """
    slice_start = max(0, start_i - train_window - vol_window - WARMUP_PAD)
    engine_input = prices.iloc[slice_start:end_i].copy()
    engine_input["return_1d"] = log_returns(engine_input["close"])
    engine_input["vol"] = rolling_volatility(engine_input["return_1d"], window=vol_window)
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
                import numpy as np
                vol_threshold = float(np.quantile(window_vols, quantile))
            bars_since_retrain = 0

        current_vol = float(engine_input.iloc[i]["vol"])
        target = 1.0 if (vol_threshold is not None and current_vol < vol_threshold) else 0.0

        p.set_target(next_ts, next_open, target, min_trade_fraction=0.10)
        p.mark(ts, close_price)
        bars_since_retrain += 1

    last_ts = engine_input.index[-1]
    last_close = float(engine_input.iloc[-1]["close"])
    p.flatten(last_ts, last_close)
    p.mark(last_ts, last_close)

    eq = p.equity_series().loc[lambda s: s.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return 0.0, 0.0, 0.0
    eq_rebased = (eq / eq.iloc[0]) * 10_000.0
    m = compute_metrics(
        eq_rebased, [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=365.0,
    )
    return m.sharpe, m.cagr, m.max_drawdown


def _run_hybrid_one_window(
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    *,
    hybrid_vol_quantile: float,
) -> tuple[float, float, float]:
    cfg = BacktestConfig(
        model_type="hybrid",
        train_window=HOMC_TRAIN_WINDOW,
        retrain_freq=21,
        n_states=5,
        order=5,
        return_bins=3,
        volatility_bins=3,
        vol_window=10,
        laplace_alpha=0.01,
        hybrid_routing_strategy="vol",
        hybrid_vol_quantile=hybrid_vol_quantile,
        periods_per_year=365.0,
        risk_free_rate=historical_usd_rate("2018-2024"),
    )
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
        eq_rebased, [],
        risk_free_rate=cfg.risk_free_rate,
        periods_per_year=cfg.periods_per_year,
    )
    return m.sharpe, m.cagr, m.max_drawdown


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars")

    strategies = {
        "vf_vw14_q0.60_rf7": {
            "family": "vol_filter",
            "vol_window": 14,
            "quantile": 0.60,
            "retrain_freq": 7,
        },
        "hvol_q0.70_legacy": {
            "family": "hybrid",
            "hybrid_vol_quantile": 0.70,
        },
        "hvol_q0.50_new_default": {
            "family": "hybrid",
            "hybrid_vol_quantile": 0.50,
        },
    }

    rows: list[dict] = []
    min_start = HOMC_TRAIN_WINDOW + 14 + WARMUP_PAD  # use max vol_window across strategies
    max_start = len(prices) - SIX_MONTHS - 1

    t0 = time.time()
    for s_name, s_cfg in strategies.items():
        print(f"\n=== {s_name} ===")
        for seed in CONFIRM_SEEDS:
            starts = draw_non_overlapping_starts(
                seed=seed,
                min_start=min_start,
                max_start=max_start,
                window_len=SIX_MONTHS,
                n_windows=N_WINDOWS,
            )
            for w, start_i in enumerate(starts):
                end_i = start_i + SIX_MONTHS
                if s_cfg["family"] == "vol_filter":
                    sharpe, cagr, mdd = _run_vol_filter_one_window(
                        prices, start_i, end_i,
                        vol_window=s_cfg["vol_window"],
                        quantile=s_cfg["quantile"],
                        retrain_freq=s_cfg["retrain_freq"],
                    )
                else:
                    sharpe, cagr, mdd = _run_hybrid_one_window(
                        prices, start_i, end_i,
                        hybrid_vol_quantile=s_cfg["hybrid_vol_quantile"],
                    )
                rows.append({
                    "strategy": s_name,
                    "seed": seed,
                    "window_idx": w,
                    "start": prices.index[start_i],
                    "end": prices.index[end_i - 1],
                    "sharpe": sharpe,
                    "cagr": cagr,
                    "max_dd": mdd,
                })
            elapsed = time.time() - t0
            print(f"  seed={seed:4d}  elapsed={elapsed:.0f}s")

    df = pd.DataFrame(rows)

    # Per-strategy summary across seeds
    per_seed = (
        df.groupby(["strategy", "seed"])["sharpe"]
        .agg(["median", "mean"])
        .reset_index()
    )
    agg = (
        per_seed.groupby("strategy")["median"]
        .agg(["mean", "sem", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    print("\n" + "=" * 72)
    print("10-seed confirmation — multi-seed avg of per-seed median Sharpe")
    print("=" * 72)
    print(agg.to_string(index=False))

    # DSR correction
    winner = agg.iloc[0]
    winner_sharpe = winner["mean"]
    n_obs = N_WINDOWS * 126
    dsr_sweep = deflated_sharpe_ratio(winner_sharpe, 144, n_obs)
    dsr_project = deflated_sharpe_ratio(winner_sharpe, LEGACY_PROJECT_TRIALS, n_obs)
    print(f"\nWinner: {winner['strategy']}")
    print(f"  multi-seed mean Sharpe = {winner_sharpe:.3f} ± {winner['sem']:.3f}")
    print(f"  DSR at sweep n_trials=144:   {dsr_sweep:.4f}")
    print(f"  DSR at project n_trials={LEGACY_PROJECT_TRIALS}: {dsr_project:.4f}")

    # Persist
    out_parquet = Path(__file__).parent / "data" / "confirm_winners.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}")

    out_md = Path(__file__).parent / "data" / "confirm_winners.md"
    lines = [
        "# 10-seed confirmation — Tier 1 vol filter vs production hybrid",
        "",
        "Confirms the Tier-1 winner from `explore_improvements.py` at 10 seeds,",
        "side-by-side with the old q=0.70 legacy default and the new q=0.50",
        "Round-2 default.",
        "",
        "## Multi-seed summary",
        "",
        "| strategy | multi-seed mean Sharpe | stderr | min seed | max seed |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| `{r['strategy']}` | {r['mean']:+.3f} | {r['sem']:.3f} | "
            f"{r['min']:+.3f} | {r['max']:+.3f} |"
        )
    lines += [
        "",
        f"**Winner**: `{winner['strategy']}`  multi-seed mean Sharpe "
        f"{winner_sharpe:+.3f} ± {winner['sem']:.3f}",
        "",
        "## DSR correction",
        "",
        "- n_trials at sweep level (explore_improvements.py): 144",
        f"- n_trials at project level (legacy + round 3): {LEGACY_PROJECT_TRIALS}",
        f"- DSR at sweep: {dsr_sweep:.4f}",
        f"- DSR at project: {dsr_project:.4f}",
        "",
        "## Interpretation",
        "",
        "If the Tier-1 vol filter winner survives DSR at project-level "
        "n_trials (DSR >= 0.95), it should be promoted to the new production "
        "default. Otherwise, document as 'winner by point estimate but not "
        "distinguishable from noise at the project-level multi-trial correction'.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
