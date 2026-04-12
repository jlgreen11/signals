"""Comprehensive multi-stock algo evaluation — top SP500 + NASDAQ names.

Runs every model variant the project ships against buy-and-hold on each
individual equity ticker, to answer: "Can ANY of the project's algos
beat B&H on any major stock, on average across seeds?"

Tickers (20, de-duped across SP500 top-15 and NASDAQ top-10):
  SP500:  AAPL MSFT NVDA AMZN GOOGL META TSLA BRK-B UNH JNJ JPM V PG XOM LLY
  NASDAQ: AVGO COST NFLX AMD ADBE  (plus overlaps already above)

Model variants tested:
  1. composite (1st-order Markov, 3×3, legacy defaults)
  2. homc (higher-order chain, order=5, n_states=5, tw=1000)
  3. hybrid H-Vol (vol-routed, q=0.50, rf=14, tw=750 — production bundle)
  4. trend (TrendFilter(200))
  5. golden_cross (DualMovingAverage(50,200))

For each (ticker, model) pair:
  - 5 pre-registered seeds × 12 non-overlapping 6-month windows
  - 252/yr annualization (equity calendar)
  - historical_usd_rate("2018-2024") risk-free rate
  - Compare: model Sharpe and CAGR vs B&H Sharpe and CAGR on the same windows

The question being answered: across 20 major stocks × 5 model variants
= 100 comparisons, how many times does an algo beat B&H on multi-seed
avg Sharpe? If the answer is "rarely" or "never," the project's algo
layer adds no value on equities.

Output:
  scripts/data/multi_stock_algo_eval.parquet  — raw per-window data
  scripts/MULTI_STOCK_ALGO_EVAL.md            — summary tables
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

# Suppress sunset warnings since we're intentionally testing all models
warnings.filterwarnings("ignore", category=DeprecationWarning)

TICKERS = [
    # SP500 top 15 by market cap
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "XOM", "LLY",
    # NASDAQ additional top-10 (not already in SP500 list)
    "AVGO", "COST", "NFLX", "AMD", "ADBE",
]

START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2026-04-01", tz="UTC")

SIX_MONTHS = 126
WARMUP_PAD = 5
N_WINDOWS = 12  # slightly fewer since equity data is ~2800 bars
SEEDS = [42, 7, 100, 999, 1337]  # 5 seeds to keep compute manageable

RF = historical_usd_rate("2018-2024")
PPY = 252.0  # equity calendar


@dataclass
class ModelConfig:
    name: str
    cfg: BacktestConfig


def _build_models() -> list[ModelConfig]:
    """Five model variants matching the project's current inventory."""
    return [
        ModelConfig(
            name="composite",
            cfg=BacktestConfig(
                model_type="composite",
                train_window=252,
                retrain_freq=21,
                return_bins=3,
                volatility_bins=3,
                vol_window=10,
                laplace_alpha=0.01,
                risk_free_rate=RF,
                periods_per_year=PPY,
            ),
        ),
        ModelConfig(
            name="homc_o5",
            cfg=BacktestConfig(
                model_type="homc",
                train_window=1000,
                retrain_freq=21,
                n_states=5,
                order=5,
                vol_window=10,
                laplace_alpha=1.0,
                risk_free_rate=RF,
                periods_per_year=PPY,
            ),
        ),
        ModelConfig(
            name="hybrid_hvol",
            cfg=BacktestConfig(
                model_type="hybrid",
                train_window=750,
                retrain_freq=14,
                n_states=5,
                order=5,
                return_bins=3,
                volatility_bins=3,
                vol_window=10,
                laplace_alpha=0.01,
                hybrid_routing_strategy="vol",
                hybrid_vol_quantile=0.50,
                risk_free_rate=RF,
                periods_per_year=PPY,
            ),
        ),
        ModelConfig(
            name="trend_200",
            cfg=BacktestConfig(
                model_type="trend",
                train_window=252,
                retrain_freq=21,
                trend_window=200,
                vol_window=10,
                risk_free_rate=RF,
                periods_per_year=PPY,
            ),
        ),
        ModelConfig(
            name="golden_cross",
            cfg=BacktestConfig(
                model_type="golden_cross",
                train_window=252,
                retrain_freq=21,
                trend_fast_window=50,
                trend_slow_window=200,
                vol_window=10,
                risk_free_rate=RF,
                periods_per_year=PPY,
            ),
        ),
    ]


def _run_one_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
) -> tuple[float, float]:
    """Returns (sharpe, cagr) for the model on one window."""
    slice_start = max(0, start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD)
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]
    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=symbol)
    except Exception:
        return 0.0, 0.0
    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return 0.0, 0.0
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    m = compute_metrics(
        eq_rebased, [],
        risk_free_rate=cfg.risk_free_rate,
        periods_per_year=cfg.periods_per_year,
    )
    return m.sharpe, m.cagr


def _bh_one_window(
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
) -> tuple[float, float]:
    """Returns (sharpe, cagr) for buy-and-hold on one window."""
    sl = prices.iloc[start_i:end_i]
    if sl.empty:
        return 0.0, 0.0
    eq = (sl["close"] / sl["close"].iloc[0]) * 10_000.0
    m = compute_metrics(eq, [], risk_free_rate=RF, periods_per_year=PPY)
    return m.sharpe, m.cagr


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    models = _build_models()

    all_rows: list[dict] = []
    t0 = time.time()

    for ti, ticker in enumerate(TICKERS, start=1):
        prices = store.load(ticker, "1d").sort_index()
        prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
        if len(prices) < 1200:
            print(f"[{ti}/{len(TICKERS)}] {ticker}: only {len(prices)} bars, skipping")
            continue

        # Use the max train_window across all models for min_start so
        # windows are identical across models for fair comparison.
        max_tw = max(m.cfg.train_window for m in models)
        min_start = max_tw + 10 + WARMUP_PAD
        max_start = len(prices) - SIX_MONTHS - 1
        if max_start - min_start < SIX_MONTHS:
            print(f"[{ti}/{len(TICKERS)}] {ticker}: too short for non-overlap windows")
            continue

        elapsed = time.time() - t0
        print(f"[{ti}/{len(TICKERS)}] {ticker} ({len(prices)} bars)  elapsed={elapsed:.0f}s")

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
                bh_sharpe, bh_cagr = _bh_one_window(prices, start_i, end_i)

                for mc in models:
                    m_sharpe, m_cagr = _run_one_window(
                        mc.cfg, prices, start_i, end_i, ticker
                    )
                    all_rows.append({
                        "ticker": ticker,
                        "model": mc.name,
                        "seed": seed,
                        "window_idx": w,
                        "start": prices.index[start_i],
                        "end": prices.index[end_i - 1],
                        "model_sharpe": m_sharpe,
                        "model_cagr": m_cagr,
                        "bh_sharpe": bh_sharpe,
                        "bh_cagr": bh_cagr,
                        "sharpe_delta": m_sharpe - bh_sharpe,
                        "cagr_delta": m_cagr - bh_cagr,
                    })

    df = pd.DataFrame(all_rows)
    out_parquet = Path(__file__).parent / "data" / "multi_stock_algo_eval.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    elapsed = time.time() - t0
    print(f"\n[wrote] {out_parquet}  ({len(df)} rows, {elapsed:.0f}s total)")

    # Aggregate: per-(ticker, model) multi-seed avg of per-seed median
    per_seed = (
        df.groupby(["ticker", "model", "seed"])
        .agg(
            model_sharpe=("model_sharpe", "median"),
            model_cagr=("model_cagr", "median"),
            bh_sharpe=("bh_sharpe", "median"),
            bh_cagr=("bh_cagr", "median"),
            sharpe_delta=("sharpe_delta", "median"),
            cagr_delta=("cagr_delta", "median"),
        )
        .reset_index()
    )
    agg = (
        per_seed.groupby(["ticker", "model"])
        .agg(
            avg_model_sharpe=("model_sharpe", "mean"),
            avg_model_cagr=("model_cagr", "mean"),
            avg_bh_sharpe=("bh_sharpe", "mean"),
            avg_bh_cagr=("bh_cagr", "mean"),
            avg_sharpe_delta=("sharpe_delta", "mean"),
            avg_cagr_delta=("cagr_delta", "mean"),
        )
        .reset_index()
    )

    # How often does each model beat B&H on multi-seed avg Sharpe?
    agg["beats_bh_sharpe"] = agg["avg_sharpe_delta"] > 0
    agg["beats_bh_cagr"] = agg["avg_cagr_delta"] > 0

    print("\n" + "=" * 100)
    print("MODEL-LEVEL SUMMARY — across all 20 tickers")
    print("=" * 100)
    model_summary = (
        agg.groupby("model")
        .agg(
            n_tickers=("ticker", "count"),
            sharpe_wins=("beats_bh_sharpe", "sum"),
            cagr_wins=("beats_bh_cagr", "sum"),
            avg_sharpe_delta=("avg_sharpe_delta", "mean"),
            avg_cagr_delta=("avg_cagr_delta", "mean"),
        )
        .reset_index()
    )
    model_summary["sharpe_win_rate"] = model_summary["sharpe_wins"] / model_summary["n_tickers"]
    model_summary["cagr_win_rate"] = model_summary["cagr_wins"] / model_summary["n_tickers"]
    model_summary = model_summary.sort_values("sharpe_win_rate", ascending=False)

    print(
        f"  {'model':<18}{'tickers':>8}{'Sh wins':>10}{'Sh win%':>10}"
        f"{'CAGR wins':>11}{'CAGR win%':>11}{'avg Δ Sh':>10}{'avg Δ CAGR':>12}"
    )
    print("  " + "-" * 98)
    for _, r in model_summary.iterrows():
        print(
            f"  {r['model']:<18}{int(r['n_tickers']):>8}"
            f"{int(r['sharpe_wins']):>10}{r['sharpe_win_rate']:>9.0%}"
            f"{int(r['cagr_wins']):>11}{r['cagr_win_rate']:>10.0%}"
            f"{r['avg_sharpe_delta']:>+10.3f}"
            f"{r['avg_cagr_delta']:>+11.1%}"
        )

    # Per-ticker best model
    print("\n" + "=" * 100)
    print("PER-TICKER BEST MODEL (by multi-seed avg Sharpe delta vs B&H)")
    print("=" * 100)
    best_per_ticker = agg.sort_values("avg_sharpe_delta", ascending=False).groupby("ticker").first().reset_index()
    best_per_ticker = best_per_ticker.sort_values("avg_sharpe_delta", ascending=False)

    print(
        f"  {'ticker':<10}{'best model':<18}"
        f"{'algo Sh':>10}{'B&H Sh':>10}{'Δ Sh':>10}"
        f"{'algo CAGR':>12}{'B&H CAGR':>12}{'Δ CAGR':>12}"
    )
    print("  " + "-" * 92)
    for _, r in best_per_ticker.iterrows():
        marker = "✓" if r["avg_sharpe_delta"] > 0 else "✗"
        print(
            f"  {r['ticker']:<10}{r['model']:<18}"
            f"{r['avg_model_sharpe']:>+10.3f}{r['avg_bh_sharpe']:>+10.3f}"
            f"{r['avg_sharpe_delta']:>+10.3f}"
            f"{r['avg_model_cagr']:>+11.1%}{r['avg_bh_cagr']:>+11.1%}"
            f"{r['avg_cagr_delta']:>+11.1%}"
            f" {marker}"
        )

    # Overall verdict
    total_pairs = len(agg)
    sharpe_wins = int(agg["beats_bh_sharpe"].sum())
    cagr_wins = int(agg["beats_bh_cagr"].sum())
    ticker_wins = int((best_per_ticker["avg_sharpe_delta"] > 0).sum())

    print(f"\n{'=' * 100}")
    print("VERDICT")
    print(f"{'=' * 100}")
    print(f"  Total (ticker × model) pairs: {total_pairs}")
    print(f"  Pairs where algo beats B&H on Sharpe: {sharpe_wins}/{total_pairs} "
          f"({sharpe_wins/total_pairs:.0%})")
    print(f"  Pairs where algo beats B&H on CAGR: {cagr_wins}/{total_pairs} "
          f"({cagr_wins/total_pairs:.0%})")
    print(f"  Tickers where BEST algo beats B&H on Sharpe: {ticker_wins}/{len(TICKERS)} "
          f"({ticker_wins/len(TICKERS):.0%})")

    if sharpe_wins / total_pairs < 0.50:
        print("\n  >>> CONCLUSION: the project's algos do NOT reliably beat buy-and-hold")
        print("      on major US equities. The algo layer adds no value on this universe.")
    else:
        print("\n  >>> CONCLUSION: the algos beat B&H on a majority of pairs.")

    # Markdown
    out_md = Path(__file__).parent / "MULTI_STOCK_ALGO_EVAL.md"
    md_lines = [
        "# Multi-stock algo evaluation — top SP500 + NASDAQ names",
        "",
        "**Question**: can ANY of the project's model variants beat buy-and-hold on "
        "major US equities, across multiple random seeds?",
        "",
        f"**Universe**: {len(TICKERS)} tickers — {', '.join(TICKERS)}",
        f"**Models**: {', '.join(m.name for m in models)}",
        f"**Seeds**: {SEEDS} ({len(SEEDS)} pre-registered)",
        f"**Windows**: {N_WINDOWS} non-overlapping 6-month per seed",
        "**Annualization**: 252/yr (equity calendar), rf ≈ 2.3%",
        "",
        "## Model-level summary",
        "",
        "| model | tickers | Sharpe wins | Sharpe win% | CAGR wins | CAGR win% | avg Δ Sharpe | avg Δ CAGR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in model_summary.iterrows():
        md_lines.append(
            f"| `{r['model']}` | {int(r['n_tickers'])} | "
            f"{int(r['sharpe_wins'])} | {r['sharpe_win_rate']:.0%} | "
            f"{int(r['cagr_wins'])} | {r['cagr_win_rate']:.0%} | "
            f"{r['avg_sharpe_delta']:+.3f} | {r['avg_cagr_delta']:+.1%} |"
        )
    md_lines += [
        "",
        "## Per-ticker best model (by Sharpe delta vs B&H)",
        "",
        "| ticker | best model | algo Sharpe | B&H Sharpe | Δ Sharpe | algo CAGR | B&H CAGR | Δ CAGR | wins? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in best_per_ticker.iterrows():
        marker = "✓" if r["avg_sharpe_delta"] > 0 else "✗"
        md_lines.append(
            f"| `{r['ticker']}` | `{r['model']}` | "
            f"{r['avg_model_sharpe']:+.3f} | {r['avg_bh_sharpe']:+.3f} | "
            f"{r['avg_sharpe_delta']:+.3f} | "
            f"{r['avg_model_cagr']:+.1%} | {r['avg_bh_cagr']:+.1%} | "
            f"{r['avg_cagr_delta']:+.1%} | {marker} |"
        )
    md_lines += [
        "",
        "## Verdict",
        "",
        f"- Total (ticker × model) pairs: **{total_pairs}**",
        f"- Algo beats B&H on Sharpe: **{sharpe_wins}/{total_pairs} ({sharpe_wins/total_pairs:.0%})**",
        f"- Algo beats B&H on CAGR: **{cagr_wins}/{total_pairs} ({cagr_wins/total_pairs:.0%})**",
        f"- Tickers where best algo beats B&H: **{ticker_wins}/{len(TICKERS)} ({ticker_wins/len(TICKERS):.0%})**",
        "",
    ]
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
