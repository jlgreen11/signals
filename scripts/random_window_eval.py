"""Evaluate composite, HOMC, and hybrid strategies on 16 random 6-month
windows across BTC-USD and ^GSPC (S&P 500) against buy & hold and a
perfect-foresight oracle.

The oracle is the theoretical ceiling on what any daily-rebalanced strategy
could achieve given the same execution model (next-open fills, 5 bps slippage,
5 bps commission). It uses ground-truth knowledge of next bar's open→close
direction to choose its position:

  oracle (long/flat) — long when next bar will close higher than its open

The "capture ratio" is what fraction of the oracle's Sharpe the strategy
picks up. 100% would mean perfect direction prediction.

Five strategies are evaluated on the SAME random windows for direct comparison:

  composite-3×3 with train_window=252                 — legacy default
  HOMC@order=5 with train_window=1000                 — bull specialist
  H-Vol   (hybrid, hard switch at vol 75th pctile)    — Tier 0c default
  H-Blend (hybrid, linear ramp 50th→85th pctile)      — Tier 0e continuous blend

Scope as of 2026-04-10 (late): only BTC-USD and ^GSPC. ETH and SOL were
deprioritized after the Tier-0b/0c results showed neither the single
models nor the hybrid reliably beat buy & hold on ETH, and the user's
production focus is BTC + S&P 500.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.backtest.portfolio import Portfolio
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore


def _periods_per_year(symbol: str) -> float:
    """Return the correct annualization factor for a given symbol.

    BTC / crypto trades 365 days/year; US equity indices trade ~252/year.
    This helper keeps the per-symbol convention explicit inside scripts
    that evaluate both calendars in the same run.
    """
    # Crypto symbols include a currency suffix like "-USD"; equity
    # indices in the project use the "^" prefix ("^GSPC", "^IXIC") or
    # plain equity tickers (TLT, GLD).
    if symbol.endswith("-USD"):
        return 365.0
    return 252.0


SYMBOLS = [
    ("BTC-USD", pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")),
    ("^GSPC",   pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")),
]


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
                hybrid_vol_quantile=0.75,
            ),
        ),
        StrategyConfig(
            name="hblend",
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
                hybrid_routing_strategy="blend",
                hybrid_blend_low=0.50,
                hybrid_blend_high=0.85,
            ),
        ),
    ]


def run_perfect_oracle(
    prices: pd.DataFrame,
    *,
    allow_short: bool,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    initial_cash: float = 10_000.0,
) -> Portfolio:
    """Walks bar-by-bar with perfect knowledge of NEXT bar's open→close move."""
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
        print(f"  [{cfg.model_type}/{cfg.hybrid_routing_strategy}] engine error: {e}")
        return compute_metrics(pd.Series(dtype=float), [])

    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(
        eq_rebased,
        [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=_periods_per_year(symbol),
    )


def _evaluate_symbol(
    store: DataStore,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    n_windows: int = 16,
    six_months: int = 126,
    seed: int = 42,
) -> pd.DataFrame:
    prices = store.load(symbol, "1d").sort_index()
    prices = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)]
    if prices.empty:
        raise ValueError(f"No data for {symbol} in the requested date range")

    print(f"{symbol}: {len(prices)} bars  ({prices.index[0].date()} → {prices.index[-1].date()})")

    vol_window = 10
    warmup_pad = 5
    homc_train_window = 1000

    min_start = homc_train_window + vol_window + warmup_pad
    max_start = len(prices) - six_months - 1
    if max_start - min_start < n_windows:
        raise ValueError(
            f"{symbol} has too few bars for {n_windows} {six_months}-bar windows "
            f"with a {homc_train_window}-bar warmup"
        )

    rng = random.Random(seed)
    # SKEPTIC_REVIEW.md § 2 / Tier A2 fix: enforce non-overlap. Previously
    # used rng.sample(...) which draws distinct starts but does NOT guarantee
    # any spacing between them — four of the original seed-42 windows clustered
    # inside January 2019 and shared 75-95% of their bars. We now reject-sample
    # until every pair of starts is at least `six_months` bars apart. Because
    # the eligible range is finite (~1400 bars for BTC 2015-2024 after the
    # 1000-bar HOMC warmup), the maximum number of truly non-overlapping
    # windows is floor((max_start - min_start) / six_months). If the caller
    # asks for more, we log a warning and return as many as fit.
    max_fit = (max_start - min_start) // six_months
    if n_windows > max_fit:
        print(
            f"  [WARN] requested n_windows={n_windows} exceeds max non-overlapping "
            f"count {max_fit} for this range; clamping to {max_fit}"
        )
        n_windows = max_fit
    starts: list[int] = []
    attempts = 0
    max_attempts = 10_000
    while len(starts) < n_windows and attempts < max_attempts:
        candidate = rng.randrange(min_start, max_start)
        if all(abs(candidate - s) >= six_months for s in starts):
            starts.append(candidate)
        attempts += 1
    if len(starts) < n_windows:
        raise RuntimeError(
            f"Could not place {n_windows} non-overlapping {six_months}-bar windows "
            f"in [{min_start}, {max_start}) after {max_attempts} attempts"
        )
    starts.sort()
    strategies = _build_strategies(vol_window, homc_train_window)

    rows: list[dict] = []
    for i, start_i in enumerate(starts, start=1):
        end_i = start_i + six_months
        eval_window = prices.iloc[start_i:end_i]

        print(f"  window {i}/{n_windows}: {eval_window.index[0].date()} → {eval_window.index[-1].date()}")

        strat_metrics: dict[str, Metrics] = {}
        for strat in strategies:
            strat_metrics[strat.name] = _run_strategy_on_window(
                strat.cfg, prices, start_i, end_i, symbol
            )

        p_oracle = run_perfect_oracle(eval_window, allow_short=False)
        m_oracle = compute_metrics(
            p_oracle.equity_series(),
            p_oracle.trades,
            risk_free_rate=historical_usd_rate("2018-2024"),
            periods_per_year=_periods_per_year(symbol),
        )

        bh_eq = (eval_window["close"] / eval_window["close"].iloc[0]) * 10_000.0
        m_bh = compute_metrics(
            bh_eq,
            [],
            risk_free_rate=historical_usd_rate("2018-2024"),
            periods_per_year=_periods_per_year(symbol),
        )

        row = {
            "symbol": symbol,
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

    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return f"{x * 100:+6.1f}%"


def _s(x: float) -> str:
    return f"{x:5.2f}"


def _print_per_window(df: pd.DataFrame, symbol: str) -> None:
    print()
    print("=" * 140)
    print(f"Per-window results — {symbol} — 16 random 6-month windows, seed 42")
    print("=" * 140)
    print(
        f"{'window':<26} "
        f"{'B&H':>9} {'Comp':>9} {'HOMC':>9} {'H-Vol':>9} {'H-Blend':>9} "
        f"{'Oracle':>11}   "
        f"{'B&H':>5} {'Comp':>5} {'HOMC':>5} {'HVol':>5} {'HBln':>5}"
    )
    for r in df.to_dict("records"):
        win = f"{r['start']} → {r['end']}"
        print(
            f"{win:<26} "
            f"{_pct(r['bh_cagr']):>9} {_pct(r['composite_cagr']):>9} "
            f"{_pct(r['homc_cagr']):>9} {_pct(r['hvol_cagr']):>9} "
            f"{_pct(r['hblend_cagr']):>9} {_pct(r['oracle_cagr']):>11}   "
            f"{_s(r['bh_sharpe']):>5} {_s(r['composite_sharpe']):>5} "
            f"{_s(r['homc_sharpe']):>5} {_s(r['hvol_sharpe']):>5} "
            f"{_s(r['hblend_sharpe']):>5}"
        )


def _print_aggregate(df: pd.DataFrame, symbol: str) -> None:
    print()
    print("=" * 140)
    print(f"Aggregate — {symbol}")
    print("=" * 140)

    def agg(name: str, series: pd.Series, formatter=_pct) -> None:
        print(
            f"{name:<32} mean={formatter(series.mean()):>10}  "
            f"median={formatter(series.median()):>10}  "
            f"min={formatter(series.min()):>10}  max={formatter(series.max()):>10}"
        )

    agg("Buy & hold CAGR", df["bh_cagr"])
    agg("Composite CAGR", df["composite_cagr"])
    agg("HOMC CAGR", df["homc_cagr"])
    agg("H-Vol CAGR", df["hvol_cagr"])
    agg("H-Blend CAGR", df["hblend_cagr"])
    agg("Oracle CAGR", df["oracle_cagr"])
    print()
    agg("Buy & hold Sharpe", df["bh_sharpe"], formatter=_s)
    agg("Composite Sharpe", df["composite_sharpe"], formatter=_s)
    agg("HOMC Sharpe", df["homc_sharpe"], formatter=_s)
    agg("H-Vol Sharpe", df["hvol_sharpe"], formatter=_s)
    agg("H-Blend Sharpe", df["hblend_sharpe"], formatter=_s)
    agg("Oracle Sharpe", df["oracle_sharpe"], formatter=_s)
    print()
    agg("Buy & hold Max DD", df["bh_mdd"])
    agg("Composite Max DD", df["composite_mdd"])
    agg("HOMC Max DD", df["homc_mdd"])
    agg("H-Vol Max DD", df["hvol_mdd"])
    agg("H-Blend Max DD", df["hblend_mdd"])

    # Capture ratios
    print()
    print(f"Sharpe capture vs perfect long/flat oracle — {symbol}")
    print("-" * 60)
    for label, col in [
        ("Composite  ", "composite_sharpe"),
        ("HOMC       ", "homc_sharpe"),
        ("H-Vol      ", "hvol_sharpe"),
        ("H-Blend    ", "hblend_sharpe"),
    ]:
        capture = df[col] / df["oracle_sharpe"].replace(0, np.nan)
        print(
            f"{label}: mean={capture.mean() * 100:+6.1f}%  "
            f"median={capture.median() * 100:+6.1f}%"
        )

    # Head-to-head
    print()
    print(f"Head-to-head on Sharpe — {symbol}")
    print("-" * 60)

    def h2h(a: str, b: str) -> int:
        return int((df[f"{a}_sharpe"] > df[f"{b}_sharpe"]).sum())

    total = len(df)
    print(f"H-Blend beats Composite : {h2h('hblend','composite')}/{total}")
    print(f"H-Blend beats HOMC      : {h2h('hblend','homc')}/{total}")
    print(f"H-Blend beats H-Vol     : {h2h('hblend','hvol')}/{total}")
    print(f"H-Blend beats Buy & Hold: {h2h('hblend','bh')}/{total}")
    print(f"H-Vol   beats Composite : {h2h('hvol','composite')}/{total}")
    print(f"H-Vol   beats HOMC      : {h2h('hvol','homc')}/{total}")
    print(f"H-Vol   beats Buy & Hold: {h2h('hvol','bh')}/{total}")
    print(f"HOMC    beats Composite : {h2h('homc','composite')}/{total}")
    print(f"HOMC    beats Buy & Hold: {h2h('homc','bh')}/{total}")
    print(f"Composite beats B&H     : {h2h('composite','bh')}/{total}")
    print()
    print("Positive CAGR:")
    for label, col in [
        ("  Composite", "composite_cagr"),
        ("  HOMC     ", "homc_cagr"),
        ("  H-Vol    ", "hvol_cagr"),
        ("  H-Blend  ", "hblend_cagr"),
    ]:
        print(f"{label}: {int((df[col] > 0).sum())}/{total}")


def main() -> None:
    store = DataStore(SETTINGS.data.dir)

    all_dfs = []
    for symbol, start_ts, end_ts in SYMBOLS:
        df = _evaluate_symbol(store, symbol, start_ts, end_ts)
        all_dfs.append(df)
        _print_per_window(df, symbol)
        _print_aggregate(df, symbol)
        print()

    # Cross-symbol summary
    print()
    print("=" * 140)
    print("Cross-symbol summary: median Sharpe per strategy per asset")
    print("=" * 140)
    print(f"{'symbol':<12} {'B&H':>8} {'Composite':>11} {'HOMC':>8} {'H-Vol':>8} {'H-Blend':>10}")
    for df in all_dfs:
        sym = df["symbol"].iloc[0]
        print(
            f"{sym:<12} "
            f"{df['bh_sharpe'].median():>8.2f} "
            f"{df['composite_sharpe'].median():>11.2f} "
            f"{df['homc_sharpe'].median():>8.2f} "
            f"{df['hvol_sharpe'].median():>8.2f} "
            f"{df['hblend_sharpe'].median():>10.2f}"
        )


if __name__ == "__main__":
    main()
