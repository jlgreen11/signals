"""Tier B4: trivial baselines for BTC — is the Markov chain actually load-bearing?

SKEPTIC_REVIEW.md Tier B4 calls out that the BTC hybrid strategy has never
been compared against trivial baselines. The S&P 500 work (scripts/sp500_trend_eval.py)
compares against 200-day MA and golden cross, but BTC evaluation has only
been benchmarked against buy & hold. That leaves an obvious unanswered question:

  Could a one-line volatility filter — long when vol is low, flat when high —
  capture most of what H-Vol hybrid produces, making the Markov chain
  machinery decorative?

This script runs FIVE strategies on the same seed-42 16-random-window BTC
evaluation used throughout the project, so the comparison is apples-to-apples
with every other prior BTC result:

  1. B&H                    — buy and hold
  2. TrendFilter(200)       — signals.model.trend.TrendFilter, Faber 200d rule
  3. DualMovingAverage(50,200) — golden cross / death cross
  4. VolFilterOnly          — IN-SCRIPT, NO MARKOV CHAIN: target = +1.0 if
                              current 20d vol < q70-of-training-vol else 0.0.
                              Retrained every 21 bars on a 1000-bar window.
                              Uses the same Portfolio class with 5 bps comm,
                              5 bps slippage — same execution model as the
                              engine, so the only difference vs H-Vol is the
                              Markov chain and the routing.
  5. H-Vol hybrid           — the current production default

Verdict: if VolFilterOnly comes within 0.2 Sharpe of H-Vol on median, then
the Markov chain contribution is negligible and the strategy is really just
"buy when vol is low, sell when vol is high" dressed up in state machinery.
That's the whole point of this script.

Running this script: `python scripts/trivial_baselines_btc.py`
Output artifact: `scripts/data/trivial_baselines_btc.parquet`
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.backtest.portfolio import Portfolio
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "BTC-USD"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")

# Evaluation methodology — match scripts/random_window_eval.py exactly.
SEED = 42
N_WINDOWS = 16
SIX_MONTHS = 126
VOL_WINDOW = 10          # engine default (for H-Vol hybrid comparability)
HOMC_TRAIN_WINDOW = 1000
WARMUP_PAD = 5

# VolFilterOnly parameters — the spec says "20d realized vol, q70 of train".
# We deliberately deviate from the engine's vol_window=10/q75 production
# defaults here because the spec's literal ablation (20d/q70) is what
# answers the "is the Markov chain decorative?" question. If VolFilterOnly
# under THESE simple knobs already captures most of the Sharpe, then the
# engine's particular vol_window/quantile tuning is not where the edge lives
# either.
VOLFILTER_LOOKBACK = 20
VOLFILTER_QUANTILE = 0.70
VOLFILTER_TRAIN_WINDOW = 1000
VOLFILTER_RETRAIN_FREQ = 21
VOLFILTER_COMMISSION_BPS = 5.0
VOLFILTER_SLIPPAGE_BPS = 5.0

OUTPUT_PARQUET = Path(__file__).resolve().parent / "data" / "trivial_baselines_btc.parquet"


@dataclass
class StrategyConfig:
    name: str
    cfg: BacktestConfig


def _build_engine_strategies() -> list[StrategyConfig]:
    """Strategies that run through the existing BacktestEngine.

    VolFilterOnly is NOT in this list — it's self-contained (see
    `_run_vol_filter_only`) to make sure no Markov-chain code path is even
    imported, so there's no chance of accidentally borrowing model machinery.
    """
    return [
        StrategyConfig(
            name="trend200",
            cfg=BacktestConfig(
                model_type="trend",
                train_window=220,  # >= trend_window (200) + vol_window slack
                retrain_freq=21,
                trend_window=200,
                vol_window=VOL_WINDOW,
            ),
        ),
        StrategyConfig(
            name="gcross",
            cfg=BacktestConfig(
                model_type="golden_cross",
                train_window=220,
                retrain_freq=21,
                trend_fast_window=50,
                trend_slow_window=200,
                vol_window=VOL_WINDOW,
            ),
        ),
        StrategyConfig(
            name="hvol",
            cfg=BacktestConfig(
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
                hybrid_vol_quantile=0.75,  # production default
            ),
        ),
    ]


def _run_engine_strategy_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
) -> Metrics:
    """Mirror of the same helper in random_window_eval.py."""
    slice_start = start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]

    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=symbol)
    except Exception as e:
        print(f"  [{cfg.model_type}] engine error: {e}")
        return compute_metrics(pd.Series(dtype=float), [])

    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(eq_rebased, [])


def _run_vol_filter_only(
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
) -> Metrics:
    """Pure volatility filter — no Markov chain, no state machine, nothing.

    The rule, in full:

        target = +1.0 if vol20d(t) < q70(vol20d over last 1000 bars)
                 else 0.0

    The threshold is recomputed every 21 bars on the trailing 1000-bar window,
    mimicking the engine's walk-forward retrain cadence. Fills happen at the
    next bar's open, just like the engine's `set_target` convention. The
    equity curve is rebased to `initial_cash` at eval-start so the Sharpe/CAGR/
    MDD numbers are directly comparable to the engine strategies' output.

    This deliberately uses the SAME Portfolio class the engine uses, with the
    same commission and slippage, so the only difference vs H-Vol hybrid is
    the signal-generation logic itself. If this strategy's Sharpe is close to
    H-Vol's, the Markov chain is decorative.
    """
    df = prices.sort_index()
    close = df["close"].astype(float)
    # 20d realized volatility = rolling stdev of daily log-ish returns.
    # We use close-to-close pct change for consistency with the engine's
    # `rolling_volatility(returns, window)` helper, just with window=20.
    returns = close.pct_change()
    vol20 = returns.rolling(window=VOLFILTER_LOOKBACK, min_periods=VOLFILTER_LOOKBACK).std()

    initial_cash = 10_000.0
    p = Portfolio(
        initial_cash=initial_cash,
        commission_bps=VOLFILTER_COMMISSION_BPS,
        slippage_bps=VOLFILTER_SLIPPAGE_BPS,
    )

    # Walk from the start of the evaluation window to the end, bar by bar.
    # At each bar we need:
    #   - a threshold from the trailing 1000-bar vol window (refreshed
    #     every VOLFILTER_RETRAIN_FREQ bars)
    #   - a decision based on the CURRENT bar's vol20 vs that threshold
    #   - a fill at the NEXT bar's open (engine convention) for the target
    #
    # We mark equity at each bar's close. On the final bar we flatten.
    eval_start_ts = df.index[start_i]
    current_threshold: float | None = None
    bars_since_retrain = VOLFILTER_RETRAIN_FREQ  # force refit on first bar

    # Iterate over [start_i, end_i - 1] so we can always fill at i+1.
    last_i = min(end_i - 1, len(df) - 1)
    for i in range(start_i, last_i):
        ts = df.index[i]
        price_now = float(df.iloc[i]["close"])
        next_ts = df.index[i + 1]
        next_open = float(df.iloc[i + 1]["open"])

        # Refit threshold on the trailing 1000 bars (strictly historical —
        # bars [i - TRAIN_WINDOW, i), exclusive of the current bar, which is
        # the same walk-forward convention as the engine).
        if bars_since_retrain >= VOLFILTER_RETRAIN_FREQ:
            train_lo = max(0, i - VOLFILTER_TRAIN_WINDOW)
            train_vol = vol20.iloc[train_lo:i].dropna()
            if len(train_vol) >= 50:
                current_threshold = float(train_vol.quantile(VOLFILTER_QUANTILE))
            bars_since_retrain = 0
        bars_since_retrain += 1

        # Decide target from CURRENT bar's vol20 (also strictly historical:
        # computed from returns up to and including bar i, which were known
        # at bar i's close). Long if vol is low (below q70), flat otherwise.
        current_vol = vol20.iloc[i]
        if current_threshold is None or pd.isna(current_vol):
            target = 0.0
        else:
            target = 1.0 if float(current_vol) < current_threshold else 0.0

        # Fill at next bar's open, then mark the current bar at its close.
        p.set_target(next_ts, next_open, target)
        p.mark(ts, price_now)

    # Flatten and mark the final bar.
    last_ts = df.index[last_i]
    last_close = float(df.iloc[last_i]["close"])
    p.flatten(last_ts, last_close)
    p.mark(last_ts, last_close)

    # Rebase equity to the eval-start point for comparability with the
    # engine strategies (which also rebase in _run_engine_strategy_on_window).
    eq = p.equity_series()
    eq = eq.loc[eq.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * initial_cash
    return compute_metrics(eq_rebased, [])


def _evaluate() -> pd.DataFrame:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    if prices.empty:
        raise ValueError(f"No {SYMBOL} data in [{START.date()}, {END.date()}]")

    print(f"{SYMBOL}: {len(prices)} bars  ({prices.index[0].date()} -> {prices.index[-1].date()})")

    min_start = HOMC_TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = len(prices) - SIX_MONTHS - 1
    if max_start - min_start < N_WINDOWS:
        raise ValueError(
            f"{SYMBOL} has too few bars for {N_WINDOWS} {SIX_MONTHS}-bar windows"
        )

    rng = random.Random(SEED)
    starts = sorted(rng.sample(range(min_start, max_start), N_WINDOWS))
    strategies = _build_engine_strategies()

    rows: list[dict] = []
    for i, start_i in enumerate(starts, start=1):
        end_i = start_i + SIX_MONTHS
        eval_window = prices.iloc[start_i:end_i]
        print(
            f"  window {i}/{N_WINDOWS}: "
            f"{eval_window.index[0].date()} -> {eval_window.index[-1].date()}"
        )

        # Buy & hold: constant long, rebased to 10k.
        bh_eq = (eval_window["close"] / eval_window["close"].iloc[0]) * 10_000.0
        m_bh = compute_metrics(bh_eq, [])

        # Engine-based strategies.
        strat_metrics: dict[str, Metrics] = {}
        for strat in strategies:
            strat_metrics[strat.name] = _run_engine_strategy_on_window(
                strat.cfg, prices, start_i, end_i, SYMBOL
            )

        # VolFilterOnly (self-contained; no Markov chain code path).
        m_volonly = _run_vol_filter_only(prices, start_i, end_i)

        row = {
            "symbol": SYMBOL,
            "start": eval_window.index[0].date(),
            "end": eval_window.index[-1].date(),
            "bh_cagr": m_bh.cagr,
            "bh_sharpe": m_bh.sharpe,
            "bh_mdd": m_bh.max_drawdown,
            "volonly_cagr": m_volonly.cagr,
            "volonly_sharpe": m_volonly.sharpe,
            "volonly_mdd": m_volonly.max_drawdown,
        }
        for strat in strategies:
            m = strat_metrics[strat.name]
            row[f"{strat.name}_cagr"] = m.cagr
            row[f"{strat.name}_sharpe"] = m.sharpe
            row[f"{strat.name}_mdd"] = m.max_drawdown
        rows.append(row)

    return pd.DataFrame(rows)


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "   n/a"
    return f"{x * 100:+6.1f}%"


def _fmt_s(x: float) -> str:
    if pd.isna(x):
        return " n/a "
    return f"{x:5.2f}"


def _print_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 104)
    print(
        f"Trivial-baseline table -- {SYMBOL}, {N_WINDOWS} random 6-month windows, seed {SEED}"
    )
    print("=" * 104)

    strategies = [
        ("B&H",             "bh"),
        ("TrendFilter(200)", "trend200"),
        ("DualMA(50,200)",  "gcross"),
        ("VolFilterOnly",   "volonly"),
        ("H-Vol hybrid",    "hvol"),
    ]

    header = (
        f"{'strategy':<18}"
        f"{'med Sharpe':>12}"
        f"{'mean Sharpe':>13}"
        f"{'med CAGR':>12}"
        f"{'mean CAGR':>12}"
        f"{'med MDD':>11}"
        f"{'pos CAGR':>12}"
    )
    print(header)
    print("-" * 104)
    for label, col in strategies:
        med_sharpe = df[f"{col}_sharpe"].median()
        mean_sharpe = df[f"{col}_sharpe"].mean()
        med_cagr = df[f"{col}_cagr"].median()
        mean_cagr = df[f"{col}_cagr"].mean()
        med_mdd = df[f"{col}_mdd"].median()
        pos = int((df[f"{col}_cagr"] > 0).sum())
        print(
            f"{label:<18}"
            f"{_fmt_s(med_sharpe):>12}"
            f"{_fmt_s(mean_sharpe):>13}"
            f"{_fmt_pct(med_cagr):>12}"
            f"{_fmt_pct(mean_cagr):>12}"
            f"{_fmt_pct(med_mdd):>11}"
            f"{pos:>8}/{len(df):<3}"
        )

    # The money question: does VolFilterOnly come within 0.2 Sharpe of H-Vol?
    print()
    print("=" * 104)
    print("Tier-B4 verdict: is the Markov chain machinery decorative?")
    print("=" * 104)
    vol_med = df["volonly_sharpe"].median()
    hvol_med = df["hvol_sharpe"].median()
    vol_mean = df["volonly_sharpe"].mean()
    hvol_mean = df["hvol_sharpe"].mean()
    gap_med = hvol_med - vol_med
    gap_mean = hvol_mean - vol_mean
    print(
        f"  VolFilterOnly Sharpe:  median={vol_med:5.2f}  mean={vol_mean:5.2f}"
    )
    print(
        f"  H-Vol hybrid  Sharpe:  median={hvol_med:5.2f}  mean={hvol_mean:5.2f}"
    )
    print(
        f"  H-Vol - VolFilterOnly: median gap={gap_med:+.2f}  mean gap={gap_mean:+.2f}"
    )
    print()
    within_threshold = abs(gap_med) < 0.2
    if within_threshold:
        print(
            "  >>> VERDICT: VolFilterOnly is WITHIN 0.2 Sharpe of H-Vol on median."
        )
        print(
            "      The Markov chain machinery is decorative on BTC for this window set."
        )
        print(
            "      Whatever edge H-Vol has is explained by the volatility filter alone."
        )
    else:
        print(
            "  >>> VERDICT: H-Vol's median Sharpe is more than 0.2 above VolFilterOnly."
        )
        print(
            "      The Markov chain machinery is contributing NON-TRIVIAL signal"
        )
        print(
            "      beyond what a one-line volatility filter produces."
        )

    # Head-to-head window count
    hvol_wins = int((df["hvol_sharpe"] > df["volonly_sharpe"]).sum())
    print()
    print(
        f"  Head-to-head: H-Vol beats VolFilterOnly on Sharpe in {hvol_wins}/{len(df)} windows"
    )


def main() -> None:
    df = _evaluate()

    # Persist results.
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    # start/end are datetime.date which parquet handles via object dtype;
    # cast to string so the parquet is portable.
    out = df.copy()
    out["start"] = out["start"].astype(str)
    out["end"] = out["end"].astype(str)
    out.to_parquet(OUTPUT_PARQUET, index=False)
    print()
    print(f"Wrote {OUTPUT_PARQUET}")

    _print_summary(df)


if __name__ == "__main__":
    main()
