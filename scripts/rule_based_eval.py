"""Experiment 2 — Rule-based signal generator on HOMC.

Closes the second arm of the SKEPTIC_REVIEW follow-up: is the Markov
class worthless on BTC, or is the current `SignalGenerator` just using
HOMC wrong?

The Nascimento et al. (2022) paper's actual proposal is rule
extraction: trade only when the current k-tuple matches one of the
top-K most frequent training rules with strong directional consensus
(P(direction) ≥ 0.7). Otherwise HOLD. This is qualitatively different
from the project's default SignalGenerator, which trades the full
marginal expectation from every state.

Pre-registered grid (DO NOT EXPAND per Round-3 guardrail D1):

    top_k      ∈ {10, 20}              # how many top rules to extract
    p_threshold ∈ {0.60, 0.70}          # min direction probability
    order      ∈ {3, 5}                 # Markov memory depth
    encoder    ∈ {quantile-5, quantile-7}  # re-use the existing quantile encoder

Total: 2 × 2 × 2 × 2 = 16 configs.

(We deliberately do NOT cross with the absolute encoder from
Experiment 1; that's a separate question. If Experiment 1 proves the
absolute encoder is materially better, a follow-up can try rule
extraction on top of it.)

Evaluation:
  - In-sample 10-seed random-window eval on 2015-01-01 → 2022-12-31.
  - Pristine holdout: single-shot walk-forward on 2023-01-01 → 2024-12-31.

Success criterion:
  - Multi-seed avg median Sharpe ≥ 1.30 (≥0.15 over pure vol filter)
  - Winner survives holdout with Sharpe > 0

Failure criterion:
  - Miss either → write negative result to `RULE_BASED_RESULTS.md`
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.metrics import compute_metrics
from signals.backtest.portfolio import Portfolio
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore
from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.homc import HigherOrderMarkovChain
from signals.model.rule_signals import RuleBasedSignalGenerator

SYMBOL = "BTC-USD"
START = pd.Timestamp("2015-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
HOLDOUT_START = pd.Timestamp("2023-01-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2024-12-31", tz="UTC")

SIX_MONTHS = 126
VOL_WINDOW = 10
WARMUP_PAD = 5
HOMC_TRAIN_WINDOW = 1000
RETRAIN_FREQ = 21
N_SEEDS = 10
SEEDS = [42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23]
N_WINDOWS_INSAMPLE = 10

# Pre-registered grid — DO NOT EXPAND.
TOP_K_VALUES = (10, 20)
P_THRESHOLDS = (0.60, 0.70)
ORDERS = (3, 5)
STATE_COUNTS = (5, 7)


@dataclass
class Config:
    label: str
    top_k: int
    p_threshold: float
    order: int
    n_states: int


def _build_grid() -> list[Config]:
    grid: list[Config] = []
    for tk in TOP_K_VALUES:
        for pt in P_THRESHOLDS:
            for order in ORDERS:
                for n_states in STATE_COUNTS:
                    grid.append(Config(
                        label=f"rule_k{tk}_p{pt:.2f}_o{order}_s{n_states}",
                        top_k=tk,
                        p_threshold=pt,
                        order=order,
                        n_states=n_states,
                    ))
    return grid


def _run_one_window(
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    *,
    cfg: Config,
    train_window: int,
    retrain_freq: int,
) -> tuple[float, float, float, int]:
    slice_start = max(0, start_i - train_window - VOL_WINDOW - WARMUP_PAD)
    engine_input = prices.iloc[slice_start:end_i].copy()
    engine_input["return_1d"] = log_returns(engine_input["close"])
    engine_input["volatility"] = rolling_volatility(
        engine_input["return_1d"], window=VOL_WINDOW
    )
    engine_input = engine_input.dropna(subset=["return_1d", "volatility"])

    if len(engine_input) < train_window + 10:
        return 0.0, 0.0, 0.0, 0

    eval_start_ts = prices.index[start_i]
    portfolio = Portfolio(initial_cash=10_000.0, commission_bps=5.0, slippage_bps=5.0)

    model: HigherOrderMarkovChain | None = None
    generator: RuleBasedSignalGenerator | None = None
    bars_since_retrain = retrain_freq

    for i in range(train_window, len(engine_input) - 1):
        ts = engine_input.index[i]
        next_ts = engine_input.index[i + 1]
        close_price = float(engine_input.iloc[i]["close"])
        next_open = float(engine_input.iloc[i + 1]["open"])

        if bars_since_retrain >= retrain_freq:
            window = engine_input.iloc[i - train_window : i]
            try:
                model = HigherOrderMarkovChain(
                    n_states=cfg.n_states,
                    order=cfg.order,
                    alpha=1.0,
                )
                model.fit(window, feature_col="return_1d", return_col="return_1d")
                generator = RuleBasedSignalGenerator(
                    model=model,
                    top_k=cfg.top_k,
                    p_threshold=cfg.p_threshold,
                    max_long=1.0,
                    max_short=1.0,
                    allow_short=False,
                )
                bars_since_retrain = 0
            except Exception:
                portfolio.mark(ts, close_price)
                bars_since_retrain += 1
                continue

        assert model is not None and generator is not None

        inference_window = engine_input.iloc[i - train_window + 1 : i + 1]
        try:
            current_state = model.predict_state(inference_window)
        except Exception:
            portfolio.mark(ts, close_price)
            bars_since_retrain += 1
            continue

        decision = generator.generate(current_state)
        target = decision.target_position
        if decision.signal.value == "HOLD":
            # Rule-based: HOLD means "stay in current position"
            target = portfolio.position_fraction(close_price)
        portfolio.set_target(next_ts, next_open, target, min_trade_fraction=0.20)
        portfolio.mark(ts, close_price)
        bars_since_retrain += 1

    last_ts = engine_input.index[-1]
    last_close = float(engine_input.iloc[-1]["close"])
    portfolio.flatten(last_ts, last_close)
    portfolio.mark(last_ts, last_close)

    eq = portfolio.equity_series().loc[lambda s: s.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return 0.0, 0.0, 0.0, len(portfolio.trades)
    eq_rebased = (eq / eq.iloc[0]) * 10_000.0
    m = compute_metrics(
        eq_rebased,
        portfolio.trades,
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=365.0,
    )
    return m.sharpe, m.cagr, m.max_drawdown, m.n_trades


def _evaluate_config_insample(
    cfg: Config, train_prices: pd.DataFrame
) -> pd.DataFrame:
    min_start = HOMC_TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = len(train_prices) - SIX_MONTHS - 1

    rows: list[dict] = []
    for seed in SEEDS:
        starts = draw_non_overlapping_starts(
            seed=seed,
            min_start=min_start,
            max_start=max_start,
            window_len=SIX_MONTHS,
            n_windows=N_WINDOWS_INSAMPLE,
        )
        for w, start_i in enumerate(starts):
            end_i = start_i + SIX_MONTHS
            sharpe, cagr, mdd, n_trades = _run_one_window(
                train_prices,
                start_i,
                end_i,
                cfg=cfg,
                train_window=HOMC_TRAIN_WINDOW,
                retrain_freq=RETRAIN_FREQ,
            )
            rows.append({
                "label": cfg.label,
                "top_k": cfg.top_k,
                "p_threshold": cfg.p_threshold,
                "order": cfg.order,
                "n_states": cfg.n_states,
                "seed": seed,
                "window_idx": w,
                "start": train_prices.index[start_i],
                "end": train_prices.index[end_i - 1],
                "sharpe": sharpe,
                "cagr": cagr,
                "max_dd": mdd,
                "n_trades": n_trades,
            })
    return pd.DataFrame(rows)


def _evaluate_holdout(
    cfg: Config, full_prices: pd.DataFrame
) -> tuple[float, float, float, int]:
    holdout_start_i = full_prices.index.searchsorted(HOLDOUT_START)
    holdout_end_i = full_prices.index.searchsorted(HOLDOUT_END, side="right")
    return _run_one_window(
        full_prices,
        holdout_start_i,
        holdout_end_i,
        cfg=cfg,
        train_window=HOMC_TRAIN_WINDOW,
        retrain_freq=RETRAIN_FREQ,
    )


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    full_prices = store.load(SYMBOL, "1d").sort_index()
    full_prices = full_prices.loc[
        (full_prices.index >= START) & (full_prices.index <= HOLDOUT_END)
    ]
    train_prices = full_prices.loc[
        (full_prices.index >= START) & (full_prices.index <= TRAIN_END)
    ]
    print(f"{SYMBOL}: train={len(train_prices)} bars, full={len(full_prices)} bars")

    grid = _build_grid()
    print(f"Pre-registered grid: {len(grid)} configs")

    t0 = time.time()
    all_rows: list[pd.DataFrame] = []
    for i, cfg in enumerate(grid, start=1):
        print(f"\n[{i}/{len(grid)}] {cfg.label}")
        df = _evaluate_config_insample(cfg, train_prices)
        all_rows.append(df)
        per_seed_median = df.groupby("seed")["sharpe"].median()
        avg = per_seed_median.mean()
        stderr = per_seed_median.sem()
        mins = per_seed_median.min()
        maxs = per_seed_median.max()
        avg_trades = df["n_trades"].mean()
        elapsed = time.time() - t0
        print(
            f"  avg Sharpe: {avg:+.3f} ± {stderr:.3f}  "
            f"(min {mins:+.3f}, max {maxs:+.3f})  "
            f"avg_trades={avg_trades:.1f}  "
            f"elapsed {elapsed:.0f}s"
        )

    full_df = pd.concat(all_rows, ignore_index=True)
    out_parquet = Path(__file__).parent / "data" / "rule_based_eval.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}  ({len(full_df)} rows)")

    per_seed = (
        full_df.groupby(["label", "top_k", "p_threshold", "order", "n_states", "seed"])["sharpe"]
        .median()
        .reset_index()
    )
    agg = (
        per_seed.groupby(["label", "top_k", "p_threshold", "order", "n_states"])["sharpe"]
        .agg(["mean", "sem", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print("\n" + "=" * 80)
    print("In-sample ranking")
    print("=" * 80)
    print(agg.to_string(index=False))

    winner = agg.iloc[0]
    winner_cfg = next(c for c in grid if c.label == winner["label"])
    print(f"\nWinner: {winner['label']}  avg Sharpe {winner['mean']:+.3f}")

    h_sharpe, h_cagr, h_mdd, h_trades = _evaluate_holdout(winner_cfg, full_prices)
    print(f"Holdout Sharpe: {h_sharpe:+.3f}  CAGR {h_cagr:+.1%}  "
          f"MDD {h_mdd:+.1%}  trades {h_trades}")

    baseline_vf_sharpe = 1.15
    materiality_threshold = baseline_vf_sharpe + 0.15
    insample_ok = winner["mean"] >= materiality_threshold
    holdout_ok = h_sharpe > 0
    success = insample_ok and holdout_ok

    print("\n" + "=" * 80)
    print("Verdict")
    print("=" * 80)
    print(f"  In-sample (≥ {materiality_threshold:.2f}): "
          f"{'PASS' if insample_ok else 'FAIL'}  ({winner['mean']:+.3f})")
    print(f"  Holdout (> 0):                {'PASS' if holdout_ok else 'FAIL'}  "
          f"({h_sharpe:+.3f})")
    print(f"  Overall: {'SUCCESS' if success else 'FAILURE'}")

    out_md = Path(__file__).parent / "RULE_BASED_RESULTS.md"
    lines = [
        "# Experiment 2 — RuleBasedSignalGenerator on HOMC",
        "",
        "**Run date**: 2026-04-11 (Round-3 follow-up)",
        "**Script**: `scripts/rule_based_eval.py`",
        "**Test parameters**:",
        "",
        "- Model: HOMC (quantile encoder) + `RuleBasedSignalGenerator`",
        f"- Grid: `top_k ∈ {list(TOP_K_VALUES)}` × "
        f"`p_threshold ∈ {list(P_THRESHOLDS)}` × "
        f"`order ∈ {list(ORDERS)}` × `n_states ∈ {list(STATE_COUNTS)}` = "
        f"{len(grid)} configs",
        "- Training: 2015-01-01 → 2022-12-31, train_window=1000, retrain_freq=21",
        f"- Seeds: 10 pre-registered ({SEEDS})",
        "- Windows: 10 non-overlapping 6-month per seed",
        "- Pristine holdout: 2023-01-01 → 2024-12-31, single-shot",
        "- Baseline: pure vol filter ~1.15 Sharpe median",
        "- Materiality: avg Sharpe ≥ 1.30 AND holdout > 0",
        "",
        "## In-sample ranking",
        "",
        "| label | top_k | p_thr | order | n_states | avg Sharpe | stderr | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| `{r['label']}` | {int(r['top_k'])} | {r['p_threshold']:.2f} | "
            f"{int(r['order'])} | {int(r['n_states'])} | "
            f"{r['mean']:+.3f} | {r['sem']:.3f} | "
            f"{r['min']:+.3f} | {r['max']:+.3f} |"
        )
    lines += [
        "",
        "## Pristine holdout — winner only",
        "",
        f"Winner: `{winner['label']}`  in-sample {winner['mean']:+.3f}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| Holdout Sharpe | {h_sharpe:+.3f} |",
        f"| Holdout CAGR | {h_cagr:+.1%} |",
        f"| Holdout Max DD | {h_mdd:+.1%} |",
        f"| Holdout trades | {h_trades} |",
        "",
        "## Verdict",
        "",
        f"- In-sample (≥ {materiality_threshold:.2f}): "
        f"**{'PASS' if insample_ok else 'FAIL'}** ({winner['mean']:+.3f})",
        f"- Holdout (> 0): **{'PASS' if holdout_ok else 'FAIL'}** ({h_sharpe:+.3f})",
        f"- **Overall: {'SUCCESS' if success else 'FAILURE'}**",
        "",
    ]
    if not success:
        lines.append(
            "**Negative result.** Per D1, the grid is NOT expanded. Rule-extraction "
            "arm of the Markov-layer question is closed as a fail."
        )
    else:
        lines.append(
            "**Positive result.** Narrow claim: rule extraction with "
            f"{winner['label']} beats the vol filter baseline by "
            f"{winner['mean'] - baseline_vf_sharpe:+.3f} Sharpe."
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
