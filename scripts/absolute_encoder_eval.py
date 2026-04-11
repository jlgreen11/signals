"""Experiment 1 — HOMC with absolute-granularity binning.

Closes the SKEPTIC_REVIEW follow-up question: is the Markov class
worthless on BTC, or only the tested (quantile) configurations?

The Nascimento et al. (2022) paper claims BTC has 7 steps of memory
at absolute 1% return bins. The project's existing HOMC uses quantile
bins, which redraw the distribution each retrain — potentially
destroying the memory structure the paper found. This script
evaluates HOMC with a fixed-width `AbsoluteGranularityEncoder`
at the paper's granularity and two neighbors.

Pre-registered grid (DO NOT EXPAND per Round-3 epistemic guardrail D1):

    bin_width ∈ {0.005, 0.01, 0.02}    # 0.5%, 1%, 2% return bins
    order     ∈ {3, 5, 7}              # Markov memory depth

Total: 9 configs.

Evaluation:
  - In-sample 10-seed random-window eval on 2015-01-01 → 2022-12-31
    (the pristine training slice). 10 seeds × 10 non-overlap windows
    = 100 window evaluations per config.
  - Pristine holdout: single-shot walk-forward on 2023-01-01 → 2024-12-31
    for the winner by in-sample multi-seed avg median Sharpe.

Success criterion (from task brief):
  - Multi-seed avg median Sharpe ≥ 1.30
    (≥0.15 over the pure vol filter baseline ~1.15 from
    `trivial_baselines_btc.py` median)
  - Winner survives the pristine holdout with Sharpe > 0

Failure criterion:
  - Winner misses the +0.15 threshold OR fails the holdout
  - On failure: write negative result to scripts/ABSOLUTE_ENCODER_RESULTS.md
    and do NOT expand the grid

Compute: ~900 walk-forward backtests. Each HOMC backtest at absolute
bins is slow because the state space is larger (fixed-width bins
can produce 20-40 states for BTC returns). Estimated wall time
60-120 minutes.
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
from signals.model.signals import SignalGenerator
from signals.model.states import AbsoluteGranularityEncoder

SYMBOL = "BTC-USD"
START = pd.Timestamp("2015-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
HOLDOUT_START = pd.Timestamp("2023-01-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2024-12-31", tz="UTC")

SIX_MONTHS = 126
VOL_WINDOW = 10
WARMUP_PAD = 5

# Pre-registered grid — DO NOT EXPAND.
BIN_WIDTHS = (0.005, 0.01, 0.02)
ORDERS = (3, 5, 7)

HOMC_TRAIN_WINDOW = 1000
RETRAIN_FREQ = 21
N_SEEDS = 10
SEEDS = [42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23]
N_WINDOWS_INSAMPLE = 10  # tight because the 2015-2022 slice is narrower

BUY_THRESHOLD_BPS = 25.0
SELL_THRESHOLD_BPS = -35.0
TARGET_SCALE_BPS = 20.0


@dataclass
class Config:
    label: str
    bin_width: float
    order: int


def _build_grid() -> list[Config]:
    grid: list[Config] = []
    for bw in BIN_WIDTHS:
        for order in ORDERS:
            grid.append(Config(
                label=f"abs_bw{bw:.3f}_o{order}",
                bin_width=bw,
                order=order,
            ))
    return grid


def _run_one_window(
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    *,
    bin_width: float,
    order: int,
    train_window: int,
    retrain_freq: int,
) -> tuple[float, float, float, int]:
    """Walk-forward backtest of HOMC-with-absolute-encoder on one window.

    Returns (sharpe, cagr, mdd, n_trades). Self-contained (does not use
    BacktestEngine) because the engine's model factory doesn't currently
    accept custom encoders.
    """
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
    generator: SignalGenerator | None = None
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
                    n_states=5,  # rewritten by the encoder's actual bin count
                    order=order,
                    alpha=1.0,
                    encoder=AbsoluteGranularityEncoder(bin_width=bin_width),
                )
                model.fit(window, feature_col="return_1d", return_col="return_1d")
                generator = SignalGenerator(
                    model=model,
                    buy_threshold_bps=BUY_THRESHOLD_BPS,
                    sell_threshold_bps=SELL_THRESHOLD_BPS,
                    target_scale_bps=TARGET_SCALE_BPS,
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
    """Multi-seed random-window eval on the 2015-2022 training slice."""
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
                bin_width=cfg.bin_width,
                order=cfg.order,
                train_window=HOMC_TRAIN_WINDOW,
                retrain_freq=RETRAIN_FREQ,
            )
            rows.append({
                "label": cfg.label,
                "bin_width": cfg.bin_width,
                "order": cfg.order,
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
    """Single-shot walk-forward on 2023-2024 holdout."""
    holdout_start_i = full_prices.index.searchsorted(HOLDOUT_START)
    holdout_end_i = full_prices.index.searchsorted(HOLDOUT_END, side="right")
    return _run_one_window(
        full_prices,
        holdout_start_i,
        holdout_end_i,
        bin_width=cfg.bin_width,
        order=cfg.order,
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
    print(
        f"{SYMBOL}: train={len(train_prices)} bars, full={len(full_prices)} bars"
    )

    grid = _build_grid()
    print(f"Pre-registered grid: {len(grid)} configs, "
          f"{N_SEEDS} seeds, {N_WINDOWS_INSAMPLE} windows each")
    print("  Bin widths (absolute return %):",
          ", ".join(f"{bw*100:.1f}%" for bw in BIN_WIDTHS))
    print("  Orders:", ORDERS)

    t0 = time.time()
    all_rows: list[pd.DataFrame] = []
    for i, cfg in enumerate(grid, start=1):
        print(f"\n[{i}/{len(grid)}] {cfg.label}")
        df = _evaluate_config_insample(cfg, train_prices)
        all_rows.append(df)

        # Per-config multi-seed summary
        per_seed_median = df.groupby("seed")["sharpe"].median()
        avg = per_seed_median.mean()
        stderr = per_seed_median.sem()
        mins = per_seed_median.min()
        maxs = per_seed_median.max()
        elapsed = time.time() - t0
        print(
            f"  avg median Sharpe: {avg:+.3f} ± {stderr:.3f}  "
            f"(min seed {mins:+.3f}, max {maxs:+.3f})  "
            f"elapsed {elapsed:.0f}s"
        )

    full_df = pd.concat(all_rows, ignore_index=True)
    out_parquet = Path(__file__).parent / "data" / "absolute_encoder_eval.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}  ({len(full_df)} rows)")

    # Per-config aggregation
    per_seed = (
        full_df.groupby(["label", "bin_width", "order", "seed"])["sharpe"]
        .median()
        .reset_index()
    )
    agg = (
        per_seed.groupby(["label", "bin_width", "order"])["sharpe"]
        .agg(["mean", "sem", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print("\n" + "=" * 80)
    print("In-sample ranking (2015-2022, 10 seeds × 10 non-overlap windows)")
    print("=" * 80)
    print(agg.to_string(index=False))

    # Pick the winner and evaluate on the pristine holdout
    winner = agg.iloc[0]
    winner_cfg = next(c for c in grid if c.label == winner["label"])
    print(f"\nIn-sample winner: {winner['label']}  "
          f"avg Sharpe {winner['mean']:+.3f} ± {winner['sem']:.3f}")

    print("\nEvaluating winner on pristine holdout (2023-2024)...")
    h_sharpe, h_cagr, h_mdd, h_trades = _evaluate_holdout(winner_cfg, full_prices)
    print(f"  Holdout Sharpe  : {h_sharpe:+.3f}")
    print(f"  Holdout CAGR    : {h_cagr:+.1%}")
    print(f"  Holdout Max DD  : {h_mdd:+.1%}")
    print(f"  Holdout trades  : {h_trades}")

    # Success / failure verdict
    baseline_vf_sharpe = 1.15  # from scripts/trivial_baselines_btc.py
    materiality_threshold = baseline_vf_sharpe + 0.15
    insample_ok = winner["mean"] >= materiality_threshold
    holdout_ok = h_sharpe > 0
    success = insample_ok and holdout_ok

    print("\n" + "=" * 80)
    print("Verdict")
    print("=" * 80)
    print(f"  In-sample criterion (≥ {materiality_threshold:.2f}):  "
          f"{'PASS' if insample_ok else 'FAIL'}  "
          f"({winner['mean']:+.3f})")
    print(f"  Holdout criterion (> 0):          "
          f"{'PASS' if holdout_ok else 'FAIL'}  ({h_sharpe:+.3f})")
    print(f"  Overall: {'SUCCESS' if success else 'FAILURE'}")

    # Write a brief results doc
    out_md = Path(__file__).parent / "ABSOLUTE_ENCODER_RESULTS.md"
    lines = [
        "# Experiment 1 — AbsoluteGranularityEncoder HOMC",
        "",
        "**Run date**: 2026-04-11 (Round-3 follow-up)",
        "**Script**: `scripts/absolute_encoder_eval.py`",
        "**Test parameters**:",
        "",
        f"- Model: HOMC with `AbsoluteGranularityEncoder(bin_width)` at "
        f"`order ∈ {list(ORDERS)}` × `bin_width ∈ {list(BIN_WIDTHS)}`",
        "- Training window: 2015-01-01 → 2022-12-31, HOMC train_window=1000, "
        "retrain_freq=21",
        f"- Seeds: 10 pre-registered ({SEEDS})",
        "- Windows: 10 non-overlapping 6-month windows per seed",
        "- Pristine holdout: 2023-01-01 → 2024-12-31, single-shot walk-forward",
        "- Baseline for comparison: pure vol filter median Sharpe ~1.15 from "
        "`scripts/trivial_baselines_btc.py`",
        "- Materiality threshold: in-sample avg Sharpe ≥ 1.30 AND holdout > 0",
        "",
        "## Pre-registered grid (per D1 — do not expand)",
        "",
        f"- `bin_width ∈ {list(BIN_WIDTHS)}`  (0.5%, 1%, 2% absolute return bins)",
        f"- `order ∈ {list(ORDERS)}`  (Markov memory depth)",
        "- 9 total configs",
        "",
        "## In-sample ranking",
        "",
        "| label | bin_width | order | avg Sharpe | stderr | min seed | max seed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| `{r['label']}` | {r['bin_width']:.3f} | {int(r['order'])} | "
            f"{r['mean']:+.3f} | {r['sem']:.3f} | "
            f"{r['min']:+.3f} | {r['max']:+.3f} |"
        )
    lines += [
        "",
        "## Pristine holdout — winner only",
        "",
        f"Winner: `{winner['label']}`  (in-sample {winner['mean']:+.3f})",
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
        f"- In-sample criterion (≥ {materiality_threshold:.2f}): "
        f"**{'PASS' if insample_ok else 'FAIL'}** ({winner['mean']:+.3f})",
        f"- Holdout criterion (> 0): "
        f"**{'PASS' if holdout_ok else 'FAIL'}** ({h_sharpe:+.3f})",
        f"- **Overall: {'SUCCESS' if success else 'FAILURE'}**",
        "",
    ]
    if not success:
        lines.append(
            "**Negative result.** Per epistemic guardrail D1, the grid is "
            "NOT expanded. The `AbsoluteGranularityEncoder` branch of the "
            "Markov-layer question is closed as a fail."
        )
    else:
        lines.append(
            "**Positive result.** Narrow claim: HOMC at absolute-granularity "
            f"{winner['label']} beats the pure vol filter baseline by "
            f"{winner['mean'] - baseline_vf_sharpe:+.3f} Sharpe on 10-seed "
            "multi-seed evaluation AND survives the pristine 2023-2024 holdout."
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
