"""Evaluate composite, HOMC, and hybrid strategies on 16 random 6-month BTC
windows against buy & hold and a perfect-foresight oracle.

The oracle is the theoretical ceiling on what any daily-rebalanced strategy
could achieve given the same execution model (next-open fills, 5 bps slippage,
5 bps commission). It uses ground-truth knowledge of next bar's open→close
direction to choose its position:

  oracle (long/flat) — long when next bar will close higher than its open
  oracle (long/short) — same, but goes short when next bar will close lower

The "capture ratio" is what fraction of the oracle's CAGR the strategy picks
up. 100% would mean perfect direction prediction; 0% means no edge over flat.

Three strategies are evaluated on the SAME random windows for direct comparison:

  composite-3×3 with train_window=252         — production default
  HOMC@order=5 with train_window=1000         — Tier-0a candidate (bull specialist)
  hybrid (HMM regime → composite|HOMC)        — Tier-3 #13 (regime-routed ensemble)
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


def _run_strategy_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
):
    """Run the engine on a warmup + eval slice, return rebased equity metrics."""
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
    prices = store.load("BTC-USD", "1d").sort_index()
    # Pull a wider range so HOMC's 1000-bar training window has enough data.
    prices = prices.loc[
        (prices.index >= pd.Timestamp("2015-01-01", tz="UTC"))
        & (prices.index <= pd.Timestamp("2024-12-31", tz="UTC"))
    ]
    print(f"BTC-USD bars: {len(prices)}  ({prices.index[0].date()} → {prices.index[-1].date()})")

    vol_window = 10
    six_months = 126  # ~6 trading months
    warmup_pad = 5

    # HOMC needs 1000 training bars; that's the binding constraint. Use the
    # same min_start for both models so they sample the same 16 windows.
    homc_train_window = 1000
    composite_train_window = 252
    min_start = homc_train_window + vol_window + warmup_pad
    max_start = len(prices) - six_months - 1

    rng = random.Random(42)
    starts = sorted(rng.sample(range(min_start, max_start), 16))

    composite_cfg = BacktestConfig(
        model_type="composite",
        train_window=composite_train_window,
        retrain_freq=21,
        return_bins=3,
        volatility_bins=3,
        vol_window=vol_window,
        laplace_alpha=0.01,
    )

    homc_cfg = BacktestConfig(
        model_type="homc",
        train_window=homc_train_window,
        retrain_freq=21,
        n_states=5,
        order=5,
        vol_window=vol_window,
        laplace_alpha=1.0,
    )

    hybrid_hmm_cfg = BacktestConfig(
        model_type="hybrid",
        train_window=homc_train_window,  # HOMC binds
        retrain_freq=21,
        n_states=5,
        order=5,
        return_bins=3,
        volatility_bins=3,
        vol_window=vol_window,
        laplace_alpha=0.01,
        hybrid_routing_strategy="hmm",
    )

    hybrid_vol_cfg = BacktestConfig(
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
        hybrid_vol_quantile=0.75,
    )

    rows: list[dict] = []
    for i, start_i in enumerate(starts, start=1):
        end_i = start_i + six_months
        eval_window = prices.iloc[start_i:end_i]

        print(f"  window {i}/16: {eval_window.index[0].date()} → {eval_window.index[-1].date()}")

        m_composite = _run_strategy_on_window(
            composite_cfg, prices, start_i, end_i, "BTC-USD"
        )
        m_homc = _run_strategy_on_window(
            homc_cfg, prices, start_i, end_i, "BTC-USD"
        )
        m_hybrid_hmm = _run_strategy_on_window(
            hybrid_hmm_cfg, prices, start_i, end_i, "BTC-USD"
        )
        m_hybrid_vol = _run_strategy_on_window(
            hybrid_vol_cfg, prices, start_i, end_i, "BTC-USD"
        )

        # Perfect oracle long/flat on the same eval window
        p_oracle = run_perfect_oracle(eval_window, allow_short=False)
        m_oracle = compute_metrics(p_oracle.equity_series(), p_oracle.trades)

        # Buy & hold
        bh_eq = (eval_window["close"] / eval_window["close"].iloc[0]) * composite_cfg.initial_cash
        m_bh = compute_metrics(bh_eq, [])

        rows.append({
            "start": eval_window.index[0].date(),
            "end": eval_window.index[-1].date(),
            "bh_cagr": m_bh.cagr,
            "comp_cagr": m_composite.cagr,
            "homc_cagr": m_homc.cagr,
            "hhmm_cagr": m_hybrid_hmm.cagr,
            "hvol_cagr": m_hybrid_vol.cagr,
            "oracle_cagr": m_oracle.cagr,
            "bh_sharpe": m_bh.sharpe,
            "comp_sharpe": m_composite.sharpe,
            "homc_sharpe": m_homc.sharpe,
            "hhmm_sharpe": m_hybrid_hmm.sharpe,
            "hvol_sharpe": m_hybrid_vol.sharpe,
            "oracle_sharpe": m_oracle.sharpe,
            "bh_mdd": m_bh.max_drawdown,
            "comp_mdd": m_composite.max_drawdown,
            "homc_mdd": m_homc.max_drawdown,
            "hhmm_mdd": m_hybrid_hmm.max_drawdown,
            "hvol_mdd": m_hybrid_vol.max_drawdown,
        })

    df = pd.DataFrame(rows)

    def pct(x: float) -> str:
        return f"{x * 100:+6.1f}%"

    def s(x: float) -> str:
        return f"{x:5.2f}"

    print()
    print("=" * 150)
    print("Per-window results: BTC-USD, 16 random 6-month windows, seed 42")
    print("=" * 150)
    header = (
        f"{'window':<26} "
        f"{'B&H':>9} {'Comp':>9} {'HOMC':>9} {'H-HMM':>9} {'H-Vol':>9} "
        f"{'Oracle':>11}   "
        f"{'B&H':>5} {'Comp':>5} {'HOMC':>5} {'HMM':>5} {'Vol':>5}"
    )
    print(header)
    for r in df.to_dict("records"):
        win = f"{r['start']} → {r['end']}"
        print(
            f"{win:<26} "
            f"{pct(r['bh_cagr']):>9} {pct(r['comp_cagr']):>9} "
            f"{pct(r['homc_cagr']):>9} {pct(r['hhmm_cagr']):>9} "
            f"{pct(r['hvol_cagr']):>9} {pct(r['oracle_cagr']):>11}   "
            f"{s(r['bh_sharpe']):>5} {s(r['comp_sharpe']):>5} "
            f"{s(r['homc_sharpe']):>5} {s(r['hhmm_sharpe']):>5} "
            f"{s(r['hvol_sharpe']):>5}"
        )

    print()
    print("=" * 150)
    print("Aggregate (across the 16 random 6-month windows)")
    print("=" * 150)

    def agg(name: str, series: pd.Series, formatter=pct) -> None:
        print(
            f"{name:<32} mean={formatter(series.mean()):>10}  "
            f"median={formatter(series.median()):>10}  "
            f"min={formatter(series.min()):>10}  max={formatter(series.max()):>10}"
        )

    agg("Buy & hold CAGR", df["bh_cagr"])
    agg("Composite CAGR", df["comp_cagr"])
    agg("HOMC@order=5 CAGR", df["homc_cagr"])
    agg("Hybrid (HMM-routed) CAGR", df["hhmm_cagr"])
    agg("Hybrid (Vol-routed) CAGR", df["hvol_cagr"])
    agg("Oracle (long/flat) CAGR", df["oracle_cagr"])
    print()
    agg("Buy & hold Sharpe", df["bh_sharpe"], formatter=s)
    agg("Composite Sharpe", df["comp_sharpe"], formatter=s)
    agg("HOMC@order=5 Sharpe", df["homc_sharpe"], formatter=s)
    agg("Hybrid (HMM) Sharpe", df["hhmm_sharpe"], formatter=s)
    agg("Hybrid (Vol) Sharpe", df["hvol_sharpe"], formatter=s)
    agg("Oracle Sharpe", df["oracle_sharpe"], formatter=s)
    print()
    agg("Buy & hold Max DD", df["bh_mdd"])
    agg("Composite Max DD", df["comp_mdd"])
    agg("HOMC@order=5 Max DD", df["homc_mdd"])
    agg("Hybrid (HMM) Max DD", df["hhmm_mdd"])
    agg("Hybrid (Vol) Max DD", df["hvol_mdd"])

    # Capture ratios
    print()
    print("=" * 150)
    print("Sharpe capture vs perfect long/flat oracle")
    print("=" * 150)
    comp_capture = df["comp_sharpe"] / df["oracle_sharpe"].replace(0, np.nan)
    homc_capture = df["homc_sharpe"] / df["oracle_sharpe"].replace(0, np.nan)
    hhmm_capture = df["hhmm_sharpe"] / df["oracle_sharpe"].replace(0, np.nan)
    hvol_capture = df["hvol_sharpe"] / df["oracle_sharpe"].replace(0, np.nan)
    print(
        f"Composite     : mean={comp_capture.mean() * 100:+6.1f}%  "
        f"median={comp_capture.median() * 100:+6.1f}%"
    )
    print(
        f"HOMC@5        : mean={homc_capture.mean() * 100:+6.1f}%  "
        f"median={homc_capture.median() * 100:+6.1f}%"
    )
    print(
        f"Hybrid (HMM)  : mean={hhmm_capture.mean() * 100:+6.1f}%  "
        f"median={hhmm_capture.median() * 100:+6.1f}%"
    )
    print(
        f"Hybrid (Vol)  : mean={hvol_capture.mean() * 100:+6.1f}%  "
        f"median={hvol_capture.median() * 100:+6.1f}%"
    )

    # Head-to-head
    print()
    print("=" * 150)
    print("Head-to-head on Sharpe")
    print("=" * 150)
    def h2h(a: str, b: str) -> int:
        return int((df[f"{a}_sharpe"] > df[f"{b}_sharpe"]).sum())

    print(f"H-HMM beats Composite : {h2h('hhmm','comp')}/16 windows")
    print(f"H-HMM beats HOMC      : {h2h('hhmm','homc')}/16 windows")
    print(f"H-HMM beats Buy & Hold: {h2h('hhmm','bh')}/16 windows")
    print(f"H-Vol beats Composite : {h2h('hvol','comp')}/16 windows")
    print(f"H-Vol beats HOMC      : {h2h('hvol','homc')}/16 windows")
    print(f"H-Vol beats Buy & Hold: {h2h('hvol','bh')}/16 windows")
    print(f"H-Vol beats H-HMM     : {h2h('hvol','hhmm')}/16 windows")
    print(f"HOMC beats Composite  : {h2h('homc','comp')}/16 windows")
    print(f"HOMC beats Buy & Hold : {h2h('homc','bh')}/16 windows")
    print(f"Composite beats B&H   : {h2h('comp','bh')}/16 windows")
    print()
    print(f"H-HMM positive CAGR   : {(df['hhmm_cagr'] > 0).sum()}/16")
    print(f"H-Vol positive CAGR   : {(df['hvol_cagr'] > 0).sum()}/16")
    print(f"HOMC positive CAGR    : {(df['homc_cagr'] > 0).sum()}/16")
    print(f"Composite positive CAGR: {(df['comp_cagr'] > 0).sum()}/16")


if __name__ == "__main__":
    main()
