"""S&P 500 trend filters on a LONG-HORIZON evaluation (24-month windows).

Motivation: the Tier-1S 6-month random-window eval showed that the
200-day MA trend filter reduced drawdowns on S&P but gave up too much
return via whipsaws — net negative on median Sharpe. The hypothesis for
this script is that 6 months is too short for a trend filter to pay off.
The classical Faber (2007) result used ~40 years of monthly data; on
shorter horizons the whipsaw cost dominates.

This script reruns the same 6-strategy lineup on 24-month (504 trading
bar) random windows and reports whether the tradeoff changes. If the
trend filter beats B&H on long horizons, that's a genuine finding worth
shipping as a long-term S&P recommendation (e.g. for retirement
accounts with multi-year horizons).

Scope:
  - ^GSPC only (BTC's 24-month behavior is already known)
  - seed 42 primary + multi-seed robustness ({42, 7, 100, 999})
  - Strategies: B&H, TrendFilter(200), DualMovingAverage(50, 200)
  - 16 random windows (fewer than 6-month case because the pool is
    smaller with 2x the window length)

Saves per-window raw results to
scripts/data/sp500_trend_long_horizon.parquet.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "^GSPC"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")
SEEDS = [42, 7, 100, 999]
N_WINDOWS = 16
TWO_YEARS = 504   # ~24 trading months
VOL_WINDOW = 10
WARMUP_PAD = 5


@dataclass
class StrategyConfig:
    name: str
    cfg: BacktestConfig


def _strategies() -> list[StrategyConfig]:
    return [
        StrategyConfig(
            name="trend200",
            cfg=BacktestConfig(
                model_type="trend",
                train_window=220,
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
    ]


def _run_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
) -> Metrics:
    slice_start = start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]
    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=SYMBOL)
    except Exception as e:
        print(f"    engine error: {e}")
        return compute_metrics(pd.Series(dtype=float), [])
    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    return compute_metrics((eq / eq.iloc[0]) * cfg.initial_cash, [])


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars ({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"Window size: {TWO_YEARS} bars (~24 months)")

    strategies = _strategies()
    min_start = 220 + VOL_WINDOW + WARMUP_PAD
    max_start = len(prices) - TWO_YEARS - 1
    if max_start - min_start < N_WINDOWS:
        raise ValueError(
            f"Not enough data for {N_WINDOWS} windows of {TWO_YEARS} bars"
        )

    rows: list[dict] = []
    overall_t0 = time.time()
    for seed in SEEDS:
        rng = random.Random(seed)
        starts = sorted(rng.sample(range(min_start, max_start), N_WINDOWS))
        print(f"\n=== seed {seed} — {len(starts)} windows ===")
        for win_idx, start_i in enumerate(starts):
            end_i = start_i + TWO_YEARS
            eval_window = prices.iloc[start_i:end_i]
            win_start = eval_window.index[0].date()
            win_end = eval_window.index[-1].date()

            # Strategies
            strategy_metrics: dict[str, Metrics] = {}
            for strat in strategies:
                strategy_metrics[strat.name] = _run_on_window(
                    strat.cfg, prices, start_i, end_i
                )

            # Buy & hold baseline
            bh_eq = (eval_window["close"] / eval_window["close"].iloc[0]) * 10_000.0
            m_bh = compute_metrics(bh_eq, [])

            row = {
                "seed": seed,
                "window_idx": win_idx,
                "window_start": str(win_start),
                "window_end": str(win_end),
                "bh_cagr": float(m_bh.cagr),
                "bh_sharpe": float(m_bh.sharpe),
                "bh_max_dd": float(m_bh.max_drawdown),
            }
            for strat in strategies:
                m = strategy_metrics[strat.name]
                row[f"{strat.name}_cagr"] = float(m.cagr)
                row[f"{strat.name}_sharpe"] = float(m.sharpe)
                row[f"{strat.name}_max_dd"] = float(m.max_drawdown)
            rows.append(row)

    print(f"\nElapsed: {time.time() - overall_t0:.0f}s")
    print(f"Rows collected: {len(rows)}")

    df = pd.DataFrame(rows)
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "sp500_trend_long_horizon.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved: {path}")

    # Per-seed aggregate
    print()
    print("=" * 100)
    print("Per-seed median Sharpe (24-month windows)")
    print("=" * 100)
    print(f"{'seed':>6}  {'B&H':>8}  {'Trend(200)':>12}  {'GoldenCross':>13}  {'Trend beats B&H?':>20}")
    for seed in SEEDS:
        sub = df[df["seed"] == seed]
        bh = sub["bh_sharpe"].median()
        trend = sub["trend200_sharpe"].median()
        gcross = sub["gcross_sharpe"].median()
        trend_wins = "YES" if trend > bh else "no"
        print(f"{seed:>6}  {bh:>8.2f}  {trend:>12.2f}  {gcross:>13.2f}  {trend_wins:>20}")

    # Overall average across seeds
    print()
    print("=" * 100)
    print("Average median-Sharpe across all 4 seeds")
    print("=" * 100)
    bh_avg = df.groupby("seed")["bh_sharpe"].median().mean()
    trend_avg = df.groupby("seed")["trend200_sharpe"].median().mean()
    gcross_avg = df.groupby("seed")["gcross_sharpe"].median().mean()
    print(f"  Buy & hold              : {bh_avg:.2f}")
    print(f"  Trend(200)              : {trend_avg:.2f}  {'(beats B&H)' if trend_avg > bh_avg else '(loses)'}")
    print(f"  GoldenCross(50,200)     : {gcross_avg:.2f}  {'(beats B&H)' if gcross_avg > bh_avg else '(loses)'}")

    # Drawdown comparison
    print()
    print("Average mean-max-drawdown across seeds:")
    bh_dd = df.groupby("seed")["bh_max_dd"].mean().mean()
    trend_dd = df.groupby("seed")["trend200_max_dd"].mean().mean()
    gcross_dd = df.groupby("seed")["gcross_max_dd"].mean().mean()
    print(f"  Buy & hold              : {bh_dd * 100:+6.1f}%")
    print(f"  Trend(200)              : {trend_dd * 100:+6.1f}%  "
          f"({'smaller' if trend_dd > bh_dd else 'larger'} than B&H)")
    print(f"  GoldenCross(50,200)     : {gcross_dd * 100:+6.1f}%")


if __name__ == "__main__":
    main()
