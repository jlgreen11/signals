"""Evaluate the composite strategy on 16 random 6-month BTC windows
against buy & hold and a perfect-foresight oracle.

The oracle is the theoretical ceiling on what any daily-rebalanced strategy
could achieve given the same execution model (next-open fills, 5 bps slippage,
5 bps commission). It uses ground-truth knowledge of next bar's open→close
direction to choose its position. Two variants:

  oracle (long/flat) — long when next bar will close higher than its open
  oracle (long/short) — same, but goes short when next bar will close lower

The "capture ratio" is what fraction of the oracle's CAGR the composite
strategy picks up. 100% would mean perfect direction prediction; 0% means
no edge over flat.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import compute_metrics
from signals.backtest.portfolio import Portfolio
from signals.config import SETTINGS
from signals.data.storage import DataStore


def run_perfect_oracle(
    prices: pd.DataFrame,
    *,
    allow_short: bool,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    initial_cash: float = 10_000.0,
) -> Portfolio:
    """Walks bar-by-bar with perfect knowledge of NEXT bar's open→close move.

    At bar i (close), looks at bar i+1's open and close. If close > open
    target +1.0; if close < open and allow_short target -1.0; else 0. Trades
    are placed at bar i+1's open — same execution model as the engine.
    """
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


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load("BTC-USD", "1d").sort_index()
    prices = prices.loc[
        (prices.index >= pd.Timestamp("2018-01-01", tz="UTC"))
        & (prices.index <= pd.Timestamp("2024-12-31", tz="UTC"))
    ]
    print(f"BTC-USD bars: {len(prices)}  ({prices.index[0].date()} → {prices.index[-1].date()})")

    train_window = 252
    vol_window = 10
    six_months = 126  # ~6 trading months
    warmup_pad = 5

    # Need: train_window valid features before the eval window starts.
    min_start = train_window + vol_window + warmup_pad
    max_start = len(prices) - six_months - 1

    rng = random.Random(42)
    starts = sorted(rng.sample(range(min_start, max_start), 16))

    cfg = BacktestConfig(
        model_type="composite",
        train_window=train_window,
        retrain_freq=21,
        return_bins=3,
        volatility_bins=3,
        vol_window=vol_window,
        laplace_alpha=0.01,
    )

    rows: list[dict] = []
    for start_i in starts:
        end_i = start_i + six_months
        slice_start = start_i - train_window - vol_window - warmup_pad
        engine_input = prices.iloc[slice_start:end_i]
        eval_window = prices.iloc[start_i:end_i]
        eval_start_ts = eval_window.index[0]

        # Composite strategy: run engine on the warmup + eval window, then
        # trim and rebase the equity curve to the eval window only.
        result = BacktestEngine(cfg).run(engine_input, symbol="BTC-USD")
        eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
        if not eq.empty:
            eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
            m_strat = compute_metrics(eq_rebased, [])
        else:
            m_strat = compute_metrics(pd.Series(dtype=float), [])

        # Perfect oracle long/flat
        p_oracle = run_perfect_oracle(eval_window, allow_short=False)
        m_oracle = compute_metrics(p_oracle.equity_series(), p_oracle.trades)

        # Perfect oracle long/short
        p_oracle_ls = run_perfect_oracle(eval_window, allow_short=True)
        m_oracle_ls = compute_metrics(p_oracle_ls.equity_series(), p_oracle_ls.trades)

        # Buy & hold
        bh_eq = (eval_window["close"] / eval_window["close"].iloc[0]) * cfg.initial_cash
        m_bh = compute_metrics(bh_eq, [])

        rows.append({
            "start": eval_window.index[0].date(),
            "end": eval_window.index[-1].date(),
            "bh_cagr": m_bh.cagr,
            "strat_cagr": m_strat.cagr,
            "oracle_cagr": m_oracle.cagr,
            "oracle_ls_cagr": m_oracle_ls.cagr,
            "bh_sharpe": m_bh.sharpe,
            "strat_sharpe": m_strat.sharpe,
            "oracle_sharpe": m_oracle.sharpe,
            "oracle_ls_sharpe": m_oracle_ls.sharpe,
            "bh_mdd": m_bh.max_drawdown,
            "strat_mdd": m_strat.max_drawdown,
            "oracle_mdd": m_oracle.max_drawdown,
        })

    df = pd.DataFrame(rows)

    def pct(x: float) -> str:
        return f"{x * 100:+6.1f}%"

    def s(x: float) -> str:
        return f"{x:5.2f}"

    print()
    print("=" * 120)
    print("Per-window results (CAGR / Sharpe)")
    print("=" * 120)
    print(
        f"{'window':<26} {'B&H':>14} {'Strategy':>14} {'Oracle L/F':>14} {'Oracle L/S':>14}   "
        f"{'B&H Sh':>7} {'Strat Sh':>9} {'Orac Sh':>8}"
    )
    for r in df.to_dict("records"):
        win = f"{r['start']} → {r['end']}"
        print(
            f"{win:<26} {pct(r['bh_cagr']):>14} {pct(r['strat_cagr']):>14} "
            f"{pct(r['oracle_cagr']):>14} {pct(r['oracle_ls_cagr']):>14}   "
            f"{s(r['bh_sharpe']):>7} {s(r['strat_sharpe']):>9} {s(r['oracle_sharpe']):>8}"
        )

    # Capture ratio = strategy CAGR / oracle CAGR (long-only oracle).
    # Negative captures mean the strategy *lost* during a window the oracle
    # would have made money on — i.e., it picked the wrong side.
    capt_lf = df["strat_cagr"] / df["oracle_cagr"].replace(0, np.nan)
    capt_ls = df["strat_cagr"] / df["oracle_ls_cagr"].replace(0, np.nan)

    print()
    print("=" * 120)
    print("Aggregate (across the 16 random 6-month windows)")
    print("=" * 120)

    def agg(name: str, series: pd.Series, formatter=pct) -> None:
        print(
            f"{name:<28} mean={formatter(series.mean()):>10}  "
            f"median={formatter(series.median()):>10}  "
            f"min={formatter(series.min()):>10}  max={formatter(series.max()):>10}"
        )

    agg("Buy & hold CAGR", df["bh_cagr"])
    agg("Composite strat CAGR", df["strat_cagr"])
    agg("Oracle (long/flat) CAGR", df["oracle_cagr"])
    agg("Oracle (long/short) CAGR", df["oracle_ls_cagr"])
    print()
    agg("Buy & hold Sharpe", df["bh_sharpe"], formatter=s)
    agg("Composite strat Sharpe", df["strat_sharpe"], formatter=s)
    agg("Oracle (long/flat) Sharpe", df["oracle_sharpe"], formatter=s)
    agg("Oracle (long/short) Sharpe", df["oracle_ls_sharpe"], formatter=s)
    print()
    agg("Buy & hold Max DD", df["bh_mdd"])
    agg("Composite strat Max DD", df["strat_mdd"])
    agg("Oracle (long/flat) Max DD", df["oracle_mdd"])

    print()
    print("=" * 120)
    print("Capture ratio: strategy CAGR / oracle CAGR")
    print("=" * 120)
    print(
        f"vs long/flat oracle  : mean={capt_lf.mean() * 100:+6.1f}%  "
        f"median={capt_lf.median() * 100:+6.1f}%  "
        f"min={capt_lf.min() * 100:+6.1f}%  max={capt_lf.max() * 100:+6.1f}%"
    )
    print(
        f"vs long/short oracle : mean={capt_ls.mean() * 100:+6.1f}%  "
        f"median={capt_ls.median() * 100:+6.1f}%  "
        f"min={capt_ls.min() * 100:+6.1f}%  max={capt_ls.max() * 100:+6.1f}%"
    )

    # How often does the strategy beat each baseline?
    win_vs_bh = (df["strat_cagr"] > df["bh_cagr"]).sum()
    pos_strat = (df["strat_cagr"] > 0).sum()
    print()
    print(f"Strategy beats buy & hold in {win_vs_bh}/16 windows")
    print(f"Strategy positive CAGR in     {pos_strat}/16 windows")
    print(f"Buy & hold positive CAGR in   {(df['bh_cagr'] > 0).sum()}/16 windows")


if __name__ == "__main__":
    main()
