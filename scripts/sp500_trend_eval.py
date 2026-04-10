"""S&P 500 random-window eval: classic trend filters vs Markov models vs B&H.

The Markov-chain models in this project (composite, HOMC, hybrid) were
empirically shown to underperform buy & hold on ^GSPC in the Tier-0e
evaluation. The hypothesis is that equity indices need a model class
designed for secular uptrends with rare sharp drawdowns — specifically,
trend-following filters like the 200-day moving average rule (Faber 2007)
or the golden cross (50/200 dual-MA).

This script runs the full 6-strategy lineup on 16 random 6-month ^GSPC
windows and reports which (if any) beats buy & hold:

  1. Buy & hold
  2. composite-3×3           (legacy Markov default)
  3. HOMC@order=5/w1000      (Markov bull specialist)
  4. H-Vol @ q=0.70           (hybrid Markov default for BTC)
  5. TrendFilter(200)        (single 200-day MA long/flat — Faber 2007)
  6. DualMovingAverage(50/200) (golden cross / death cross)

Uses the same seed-42 random-window methodology as scripts/random_window_eval.py
for consistency with prior results.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.backtest.portfolio import Portfolio
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "^GSPC"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")


@dataclass
class StrategyConfig:
    name: str
    cfg: BacktestConfig


def _build_strategies(vol_window: int, homc_train_window: int) -> list[StrategyConfig]:
    return [
        StrategyConfig(
            name="composite",
            cfg=BacktestConfig(
                model_type="composite",
                train_window=252,
                retrain_freq=21,
                return_bins=3,
                volatility_bins=3,
                vol_window=vol_window,
                laplace_alpha=0.01,
            ),
        ),
        StrategyConfig(
            name="homc",
            cfg=BacktestConfig(
                model_type="homc",
                train_window=homc_train_window,
                retrain_freq=21,
                n_states=5,
                order=5,
                vol_window=vol_window,
                laplace_alpha=1.0,
            ),
        ),
        StrategyConfig(
            name="hvol",
            cfg=BacktestConfig(
                model_type="hybrid",
                train_window=homc_train_window,
                retrain_freq=21,
                n_states=5,
                order=5,
                return_bins=3,
                volatility_bins=3,
                vol_window=vol_window,
                laplace_alpha=0.01,
                hybrid_routing_strategy="vol",
                hybrid_vol_quantile=0.70,
            ),
        ),
        StrategyConfig(
            name="trend200",
            cfg=BacktestConfig(
                model_type="trend",
                train_window=220,  # needs >= trend_window (200) + vol_window
                retrain_freq=21,
                trend_window=200,
                vol_window=vol_window,
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
                vol_window=vol_window,
            ),
        ),
    ]


def run_perfect_oracle(
    prices: pd.DataFrame,
    *,
    allow_short: bool = False,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    initial_cash: float = 10_000.0,
) -> Portfolio:
    df = prices.sort_index()
    p = Portfolio(
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    end_i = len(df) - 1
    for i in range(end_i):
        ts = df.index[i]
        next_ts = df.index[i + 1]
        next_open = float(df.iloc[i + 1]["open"])
        next_close = float(df.iloc[i + 1]["close"])
        if next_close > next_open:
            target = 1.0
        elif next_close < next_open and allow_short:
            target = -1.0
        else:
            target = 0.0
        p.set_target(next_ts, next_open, target)
        p.mark(ts, float(df.iloc[i]["close"]))
    last_ts = df.index[end_i]
    last_close = float(df.iloc[end_i]["close"])
    p.flatten(last_ts, last_close)
    p.mark(last_ts, last_close)
    return p


def _run_strategy_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
) -> Metrics:
    warmup_pad = 5
    slice_start = start_i - cfg.train_window - cfg.vol_window - warmup_pad
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


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars ({prices.index[0].date()} → {prices.index[-1].date()})")

    vol_window = 10
    homc_train_window = 1000
    warmup_pad = 5
    six_months = 126
    n_windows = 16
    seed = 42

    min_start = homc_train_window + vol_window + warmup_pad
    max_start = len(prices) - six_months - 1
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(min_start, max_start), n_windows))
    strategies = _build_strategies(vol_window, homc_train_window)

    rows: list[dict] = []
    for i, start_i in enumerate(starts, start=1):
        end_i = start_i + six_months
        eval_window = prices.iloc[start_i:end_i]
        print(f"  window {i}/{n_windows}: {eval_window.index[0].date()} → {eval_window.index[-1].date()}")

        strat_metrics: dict[str, Metrics] = {}
        for strat in strategies:
            strat_metrics[strat.name] = _run_strategy_on_window(
                strat.cfg, prices, start_i, end_i, SYMBOL
            )

        p_oracle = run_perfect_oracle(eval_window, allow_short=False)
        m_oracle = compute_metrics(p_oracle.equity_series(), p_oracle.trades)

        bh_eq = (eval_window["close"] / eval_window["close"].iloc[0]) * 10_000.0
        m_bh = compute_metrics(bh_eq, [])

        row = {
            "start": eval_window.index[0].date(),
            "end": eval_window.index[-1].date(),
            "bh_cagr": m_bh.cagr,
            "bh_sharpe": m_bh.sharpe,
            "bh_mdd": m_bh.max_drawdown,
            "oracle_cagr": m_oracle.cagr,
            "oracle_sharpe": m_oracle.sharpe,
        }
        for strat in strategies:
            m = strat_metrics[strat.name]
            row[f"{strat.name}_cagr"] = m.cagr
            row[f"{strat.name}_sharpe"] = m.sharpe
            row[f"{strat.name}_mdd"] = m.max_drawdown
        rows.append(row)

    df = pd.DataFrame(rows)

    def pct(x: float) -> str:
        return f"{x * 100:+6.1f}%"

    def s(x: float) -> str:
        return f"{x:5.2f}"

    print()
    print("=" * 140)
    print(f"Per-window results: {SYMBOL}, 16 random 6-month windows, seed 42")
    print("=" * 140)
    print(
        f"{'window':<26} "
        f"{'B&H':>9} {'Comp':>9} {'HOMC':>9} {'H-Vol':>9} {'Trend':>9} {'GCross':>9} "
        f"{'Oracle':>10}   "
        f"{'B&H':>5} {'Comp':>5} {'HOMC':>5} {'HVol':>5} {'Tr':>5} {'GC':>5}"
    )
    for r in df.to_dict("records"):
        win = f"{r['start']} → {r['end']}"
        print(
            f"{win:<26} "
            f"{pct(r['bh_cagr']):>9} {pct(r['composite_cagr']):>9} "
            f"{pct(r['homc_cagr']):>9} {pct(r['hvol_cagr']):>9} "
            f"{pct(r['trend200_cagr']):>9} {pct(r['gcross_cagr']):>9} "
            f"{pct(r['oracle_cagr']):>10}   "
            f"{s(r['bh_sharpe']):>5} {s(r['composite_sharpe']):>5} "
            f"{s(r['homc_sharpe']):>5} {s(r['hvol_sharpe']):>5} "
            f"{s(r['trend200_sharpe']):>5} {s(r['gcross_sharpe']):>5}"
        )

    print()
    print("=" * 140)
    print(f"Aggregate — {SYMBOL}")
    print("=" * 140)

    def agg(name: str, series: pd.Series, formatter=pct) -> None:
        print(
            f"{name:<32} mean={formatter(series.mean()):>10}  "
            f"median={formatter(series.median()):>10}  "
            f"min={formatter(series.min()):>10}  max={formatter(series.max()):>10}"
        )

    agg("Buy & hold CAGR", df["bh_cagr"])
    agg("Composite CAGR", df["composite_cagr"])
    agg("HOMC CAGR", df["homc_cagr"])
    agg("H-Vol CAGR", df["hvol_cagr"])
    agg("Trend(200) CAGR", df["trend200_cagr"])
    agg("GoldenCross(50,200) CAGR", df["gcross_cagr"])
    agg("Oracle CAGR", df["oracle_cagr"])
    print()
    agg("Buy & hold Sharpe", df["bh_sharpe"], formatter=s)
    agg("Composite Sharpe", df["composite_sharpe"], formatter=s)
    agg("HOMC Sharpe", df["homc_sharpe"], formatter=s)
    agg("H-Vol Sharpe", df["hvol_sharpe"], formatter=s)
    agg("Trend(200) Sharpe", df["trend200_sharpe"], formatter=s)
    agg("GoldenCross(50,200) Sharpe", df["gcross_sharpe"], formatter=s)
    agg("Oracle Sharpe", df["oracle_sharpe"], formatter=s)
    print()
    agg("Buy & hold Max DD", df["bh_mdd"])
    agg("Composite Max DD", df["composite_mdd"])
    agg("HOMC Max DD", df["homc_mdd"])
    agg("H-Vol Max DD", df["hvol_mdd"])
    agg("Trend(200) Max DD", df["trend200_mdd"])
    agg("GoldenCross Max DD", df["gcross_mdd"])

    print()
    print("=" * 140)
    print(f"Head-to-head vs buy & hold on Sharpe — {SYMBOL}")
    print("=" * 140)

    total = len(df)
    for label, col in [
        ("Composite       ", "composite"),
        ("HOMC            ", "homc"),
        ("H-Vol           ", "hvol"),
        ("Trend(200)      ", "trend200"),
        ("GoldenCross(50,200)", "gcross"),
    ]:
        wins_sharpe = int((df[f"{col}_sharpe"] > df["bh_sharpe"]).sum())
        wins_cagr = int((df[f"{col}_cagr"] > df["bh_cagr"]).sum())
        better_mdd = int((df[f"{col}_mdd"] > df["bh_mdd"]).sum())  # less negative = better
        print(
            f"{label}: beats B&H on Sharpe {wins_sharpe}/{total}, "
            f"on CAGR {wins_cagr}/{total}, smaller max DD {better_mdd}/{total}"
        )

    # Verdict
    print()
    print("=" * 140)
    print("Verdict (by median Sharpe)")
    print("=" * 140)
    medians = {
        "Buy & hold": df["bh_sharpe"].median(),
        "Composite": df["composite_sharpe"].median(),
        "HOMC": df["homc_sharpe"].median(),
        "H-Vol": df["hvol_sharpe"].median(),
        "Trend(200)": df["trend200_sharpe"].median(),
        "GoldenCross(50,200)": df["gcross_sharpe"].median(),
    }
    ranked = sorted(medians.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (name, sh) in enumerate(ranked, 1):
        marker = "🏆" if rank == 1 else "  "
        print(f"  {marker} {rank}. {name:<24} median Sharpe {sh:5.2f}")
    print()
    best_name, best_sh = ranked[0]
    bh_sh = medians["Buy & hold"]
    if best_name == "Buy & hold":
        print("  Buy & hold is still the winner. No strategy beats it on S&P 500.")
    else:
        delta = best_sh - bh_sh
        print(f"  Best active strategy: {best_name} (median Sharpe {best_sh:.2f})")
        print(f"  Improvement over B&H: {delta:+.2f} Sharpe")


if __name__ == "__main__":
    main()
