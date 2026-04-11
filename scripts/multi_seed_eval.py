"""Multi-seed validation of the H-Vol hybrid's 16-window random evaluation.

Addresses Tier A1 and A5 of SKEPTIC_REVIEW.md:

  A1 — The headline "Sharpe 2.15" number from random_window_eval.py is
       computed at a single seed (42) and a single quantile (0.75). A
       single seed determines which 16 random 6-month windows are drawn,
       so the headline could be a seed-specific artifact. This script
       re-runs the SAME 16-window evaluation across 10 seeds and reports
       the mean +/- stderr of H-Vol's median Sharpe, so we can see how
       much of the headline survives reseeding.

  A5 — The vol_quantile_sweep.py script ranks the quantile grid only at
       seed=42. This script crosses the 10 seeds with the 6-quantile grid
       (60 configurations total) and reports the multi-seed average of
       each quantile's median Sharpe, so we can see whether q=0.70 is
       genuinely the multi-seed optimum or a seed-42 artifact.

Scope is intentionally limited to BTC-USD and the current production
H-Vol configuration (hybrid, order=5, n_states=5, train_window=1000,
vol_window=10). This is a validation pass, not a new hyperparameter
sweep — the 60-config budget (10 seeds x 6 quantiles) is the ceiling.

Outputs are saved to scripts/data/multi_seed_eval.parquet for downstream
analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

# Production H-Vol config (per signals_findings_2026_04.md and IMPROVEMENTS.md):
#   hybrid model, order=5, n_states=5, train_window=1000, vol_window=10,
#   hybrid_vol_quantile=0.70 is the current production default from the
#   seed=42 sweep. A5 tests whether that 0.70 pick survives reseeding.
SYMBOL = "BTC-USD"
START_TS = pd.Timestamp("2015-01-01", tz="UTC")
END_TS = pd.Timestamp("2024-12-31", tz="UTC")

SEEDS = [42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23]
QUANTILE_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
PRODUCTION_QUANTILE = 0.70

N_WINDOWS = 16
SIX_MONTHS = 126
VOL_WINDOW = 10
HOMC_TRAIN_WINDOW = 1000
WARMUP_PAD = 5

OUTPUT_PARQUET = Path(__file__).parent / "data" / "multi_seed_eval.parquet"


@dataclass
class WindowResult:
    seed: int
    quantile: float
    window_idx: int
    start: pd.Timestamp
    end: pd.Timestamp
    hvol_cagr: float
    hvol_sharpe: float
    hvol_mdd: float
    bh_cagr: float
    bh_sharpe: float
    bh_mdd: float


def _make_hvol_cfg(quantile: float) -> BacktestConfig:
    """Current production H-Vol config — matches random_window_eval.py's
    _build_strategies() hvol branch except the quantile is parameterized."""
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
        hybrid_vol_quantile=quantile,
    )


# Copied from random_window_eval.py::_run_strategy_on_window (with the
# print-on-error behavior kept). Duplicated rather than imported so this
# script stays self-contained and survives any signature changes to the
# reference script.
def _run_strategy_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
) -> Metrics:
    slice_start = start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]

    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=symbol)
    except Exception as e:
        print(f"  [{cfg.model_type}/q={cfg.hybrid_vol_quantile}] engine error: {e}")
        return compute_metrics(pd.Series(dtype=float), [])

    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(
        eq_rebased,
        [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=365.0,
    )


def _draw_starts(n_bars: int, seed: int) -> list[int]:
    """Same window-drawing logic as random_window_eval.py::_evaluate_symbol."""
    min_start = HOMC_TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = n_bars - SIX_MONTHS - 1
    if max_start - min_start < N_WINDOWS:
        raise ValueError(
            f"too few bars for {N_WINDOWS} {SIX_MONTHS}-bar windows with a "
            f"{HOMC_TRAIN_WINDOW}-bar warmup (min_start={min_start}, "
            f"max_start={max_start})"
        )
    from _window_sampler import draw_non_overlapping_starts
    return draw_non_overlapping_starts(
        seed=seed,
        min_start=min_start,
        max_start=max_start,
        window_len=SIX_MONTHS,
        n_windows=N_WINDOWS,
    )


def _evaluate_seed_quantile(
    prices: pd.DataFrame,
    seed: int,
    quantile: float,
) -> list[WindowResult]:
    """Run the 16-window evaluation at a single (seed, quantile) pair."""
    starts = _draw_starts(len(prices), seed)
    cfg = _make_hvol_cfg(quantile)

    rows: list[WindowResult] = []
    for i, start_i in enumerate(starts, start=1):
        end_i = start_i + SIX_MONTHS
        eval_window = prices.iloc[start_i:end_i]

        m_hvol = _run_strategy_on_window(cfg, prices, start_i, end_i, SYMBOL)

        bh_eq = (eval_window["close"] / eval_window["close"].iloc[0]) * 10_000.0
        m_bh = compute_metrics(
            bh_eq,
            [],
            risk_free_rate=historical_usd_rate("2018-2024"),
            periods_per_year=365.0,
        )

        rows.append(
            WindowResult(
                seed=seed,
                quantile=quantile,
                window_idx=i,
                start=eval_window.index[0],
                end=eval_window.index[-1],
                hvol_cagr=m_hvol.cagr,
                hvol_sharpe=m_hvol.sharpe,
                hvol_mdd=m_hvol.max_drawdown,
                bh_cagr=m_bh.cagr,
                bh_sharpe=m_bh.sharpe,
                bh_mdd=m_bh.max_drawdown,
            )
        )
    return rows


def _load_prices() -> pd.DataFrame:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START_TS) & (prices.index <= END_TS)]
    if prices.empty:
        raise ValueError(f"No data for {SYMBOL} in {START_TS.date()} → {END_TS.date()}")
    return prices


def _seed_summary(df: pd.DataFrame, seed: int) -> dict:
    """One-row summary for a seed at the production quantile."""
    sub = df[(df["seed"] == seed) & (df["quantile"] == PRODUCTION_QUANTILE)]
    return {
        "seed": seed,
        "hvol_median_sharpe": float(sub["hvol_sharpe"].median()),
        "bh_median_sharpe":   float(sub["bh_sharpe"].median()),
        "hvol_median_cagr":   float(sub["hvol_cagr"].median()),
        "bh_median_cagr":     float(sub["bh_cagr"].median()),
        "hvol_positive_cagr": int((sub["hvol_cagr"] > 0).sum()),
        "n_windows":          int(len(sub)),
    }


def _mean_stderr(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=float)
    n = len(arr)
    if n <= 1:
        return float(arr.mean()) if n else float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(n))


def _print_seed_table(df: pd.DataFrame) -> None:
    print()
    print("=" * 110)
    print(
        f"Per-seed results @ production quantile q={PRODUCTION_QUANTILE:.2f} — "
        f"{SYMBOL} — 16 random 6-month windows"
    )
    print("=" * 110)
    print(
        f"{'seed':>6}  {'H-Vol Sh':>9}  {'B&H Sh':>9}  "
        f"{'H-Vol CAGR':>11}  {'B&H CAGR':>11}  {'pos CAGR':>10}"
    )
    print("-" * 110)

    summaries = [_seed_summary(df, s) for s in SEEDS]
    for r in summaries:
        print(
            f"{r['seed']:>6}  "
            f"{r['hvol_median_sharpe']:>9.2f}  "
            f"{r['bh_median_sharpe']:>9.2f}  "
            f"{r['hvol_median_cagr'] * 100:>10.1f}%  "
            f"{r['bh_median_cagr'] * 100:>10.1f}%  "
            f"{r['hvol_positive_cagr']:>3}/{r['n_windows']:<6}"
        )

    print("-" * 110)
    hvol_sh_mean, hvol_sh_se = _mean_stderr([r["hvol_median_sharpe"] for r in summaries])
    bh_sh_mean,   bh_sh_se   = _mean_stderr([r["bh_median_sharpe"] for r in summaries])
    hvol_c_mean,  hvol_c_se  = _mean_stderr([r["hvol_median_cagr"] for r in summaries])
    bh_c_mean,    bh_c_se    = _mean_stderr([r["bh_median_cagr"] for r in summaries])
    pos_mean,     pos_se     = _mean_stderr([r["hvol_positive_cagr"] for r in summaries])

    print(
        f"{'mean':>6}  "
        f"{hvol_sh_mean:>9.2f}  {bh_sh_mean:>9.2f}  "
        f"{hvol_c_mean * 100:>10.1f}%  {bh_c_mean * 100:>10.1f}%  "
        f"{pos_mean:>10.2f}"
    )
    print(
        f"{'stderr':>6}  "
        f"{hvol_sh_se:>9.2f}  {bh_sh_se:>9.2f}  "
        f"{hvol_c_se * 100:>10.1f}%  {bh_c_se * 100:>10.1f}%  "
        f"{pos_se:>10.2f}"
    )
    print()
    print(
        f"A1 verdict: H-Vol median Sharpe across {len(SEEDS)} seeds = "
        f"{hvol_sh_mean:.2f} +/- {hvol_sh_se:.2f} (stderr)"
    )


def _print_quantile_table(df: pd.DataFrame) -> None:
    print()
    print("=" * 110)
    print(
        f"Multi-seed quantile sweep — {SYMBOL} — "
        f"{len(SEEDS)} seeds x {len(QUANTILE_GRID)} quantiles = "
        f"{len(SEEDS) * len(QUANTILE_GRID)} configs"
    )
    print("=" * 110)
    print(
        f"{'quantile':>9}  {'avg median Sh':>14}  {'stderr':>9}  "
        f"{'avg median CAGR':>17}  {'min seed Sh':>12}  {'max seed Sh':>12}"
    )
    print("-" * 110)

    rows = []
    for q in QUANTILE_GRID:
        per_seed_median_sh: list[float] = []
        per_seed_median_cagr: list[float] = []
        for s in SEEDS:
            sub = df[(df["seed"] == s) & (df["quantile"] == q)]
            per_seed_median_sh.append(float(sub["hvol_sharpe"].median()))
            per_seed_median_cagr.append(float(sub["hvol_cagr"].median()))
        mean_sh, se_sh = _mean_stderr(per_seed_median_sh)
        mean_c,  _     = _mean_stderr(per_seed_median_cagr)
        rows.append(
            {
                "quantile": q,
                "avg_median_sharpe": mean_sh,
                "stderr_median_sharpe": se_sh,
                "avg_median_cagr": mean_c,
                "min_seed_sharpe": float(np.min(per_seed_median_sh)),
                "max_seed_sharpe": float(np.max(per_seed_median_sh)),
            }
        )
        marker = "  <- current default" if abs(q - PRODUCTION_QUANTILE) < 1e-9 else ""
        print(
            f"{q:>9.2f}  "
            f"{mean_sh:>14.2f}  {se_sh:>9.2f}  "
            f"{mean_c * 100:>16.1f}%  "
            f"{rows[-1]['min_seed_sharpe']:>12.2f}  "
            f"{rows[-1]['max_seed_sharpe']:>12.2f}"
            f"{marker}"
        )

    best = max(rows, key=lambda r: r["avg_median_sharpe"])
    default = next(r for r in rows if abs(r["quantile"] - PRODUCTION_QUANTILE) < 1e-9)
    print()
    print(
        f"A5 verdict: multi-seed best quantile = {best['quantile']:.2f} "
        f"(avg median Sh = {best['avg_median_sharpe']:.2f} "
        f"+/- {best['stderr_median_sharpe']:.2f}); "
        f"current default q={PRODUCTION_QUANTILE:.2f} avg median Sh = "
        f"{default['avg_median_sharpe']:.2f}  "
        f"(delta = {best['avg_median_sharpe'] - default['avg_median_sharpe']:+.2f})"
    )


def main() -> None:
    prices = _load_prices()
    print(
        f"{SYMBOL}: {len(prices)} bars  "
        f"({prices.index[0].date()} → {prices.index[-1].date()})"
    )
    print(
        f"Running {len(SEEDS)} seeds x {len(QUANTILE_GRID)} quantiles x "
        f"{N_WINDOWS} windows = {len(SEEDS) * len(QUANTILE_GRID) * N_WINDOWS} "
        f"backtests"
    )

    all_rows: list[WindowResult] = []
    total = len(SEEDS) * len(QUANTILE_GRID)
    done = 0
    for seed in SEEDS:
        for q in QUANTILE_GRID:
            done += 1
            print(f"  [{done:>2}/{total}] seed={seed:<5} q={q:.2f}")
            all_rows.extend(_evaluate_seed_quantile(prices, seed, q))

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nWrote {len(df)} rows to {OUTPUT_PARQUET}")

    _print_seed_table(df)
    _print_quantile_table(df)


if __name__ == "__main__":
    main()
