"""Multi-asset portfolio experiment: BTC H-Vol hybrid + ^GSPC buy & hold.

Tests the "diversification lunch" hypothesis: BTC hybrid alone has median
Sharpe 2.15 on the 16-window random eval, S&P buy & hold has median 0.77.
BTC and S&P have historically low correlation (~0.1-0.2 daily on 2015-2024).
A mixed portfolio with a small SPX weight should produce a Sharpe higher
than either component alone — specifically, Sharpe gains from diversification
even though each component contributes less return in isolation.

Tests 7 allocation ratios at window-start constant-mix (no mid-window
rebalancing — equivalent to putting W_btc in BTC strategy and W_sp in
SPX B&H on day 1 and letting them drift to the end):

    100/0, 80/20, 60/40, 50/50, 40/60, 20/80, 0/100

Plus a ~daily-rebalanced constant-mix variant for comparison — this is
closer to "risk parity light" where the fixed weight is held but
rebalanced each day (accrues trading costs in reality but free in backtest).

Saves per-window raw results to scripts/data/btc_sp500_portfolio.parquet:

    columns: rebalance, btc_weight, sp_weight, window_idx, window_start,
             window_end, final_equity, cagr, sharpe, max_dd, btc_return,
             sp_return
    rows   : 7 weights × 2 rebalance modes × 16 windows = 224

All source data: 2018-01-01 → 2024-12-31, seed 42 window selection
matching the existing random_window_eval methodology.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

BTC_SYMBOL = "BTC-USD"
SP_SYMBOL = "^GSPC"
# 2015-01-01 start matches the BTC random_window_eval methodology so the
# window distribution is directly comparable. Both BTC and S&P have data
# back to 2015-01-02. Using 2018 as the start (original draft) pushed
# all 16 random windows into 2021-2024 because BTC's 1000-bar warmup
# ate most of the 2018-2020 history, biasing the result.
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")

# BTC hybrid default configuration (from Tier-0e/0f winner)
BTC_CFG = BacktestConfig(
    model_type="hybrid",
    train_window=1000,
    retrain_freq=21,
    n_states=5,
    order=5,
    return_bins=3,
    volatility_bins=3,
    vol_window=10,
    laplace_alpha=0.01,
    hybrid_routing_strategy="vol",
    hybrid_vol_quantile=0.70,
)

# Allocations: BTC weight, SP weight (must sum to 1)
WEIGHT_GRID = [
    (1.00, 0.00),
    (0.80, 0.20),
    (0.60, 0.40),
    (0.50, 0.50),
    (0.40, 0.60),
    (0.20, 0.80),
    (0.00, 1.00),
]


def _run_btc_strategy_on_window(
    prices: pd.DataFrame,
    window_start_date: pd.Timestamp,
    window_end_date: pd.Timestamp,
) -> pd.Series:
    """Run the BTC hybrid on one calendar-date window. Returns its equity
    curve with a *date-normalized* index (time component zeroed out) so
    it can be cleanly combined with S&P's date-indexed equity curve."""
    warmup_bars = BTC_CFG.train_window + BTC_CFG.vol_window + 5
    # Slice: (warmup_bars before window_start) to window_end
    slice_mask = (
        prices.index >= window_start_date - pd.Timedelta(days=warmup_bars * 2)
    ) & (prices.index <= window_end_date)
    engine_input = prices[slice_mask]
    if len(engine_input) < warmup_bars:
        return pd.Series(dtype=float)

    try:
        result = BacktestEngine(BTC_CFG).run(engine_input, symbol=BTC_SYMBOL)
    except Exception as e:
        print(f"    btc engine error: {e}")
        return pd.Series(dtype=float)

    eq = result.equity_curve.loc[result.equity_curve.index >= window_start_date]
    if eq.empty or eq.iloc[0] <= 0:
        return pd.Series(dtype=float)
    rebased = (eq / eq.iloc[0]) * 10_000.0
    rebased.index = rebased.index.normalize()
    return rebased


def _run_sp_bh_on_window(
    prices: pd.DataFrame,
    window_start_date: pd.Timestamp,
    window_end_date: pd.Timestamp,
) -> pd.Series:
    """S&P buy & hold equity curve for the given calendar-date window.
    Returns a date-normalized index."""
    mask = (prices.index >= window_start_date) & (prices.index <= window_end_date)
    window = prices[mask]
    if window.empty:
        return pd.Series(dtype=float)
    eq = (window["close"] / window["close"].iloc[0]) * 10_000.0
    eq.index = eq.index.normalize()
    return eq


def _combine_portfolio(
    btc_equity: pd.Series,
    sp_equity: pd.Series,
    w_btc: float,
    w_sp: float,
    rebalance: str,
) -> pd.Series:
    """Build a portfolio equity curve from two component equity curves.

    Uses BTC's date index as the master (BTC trades 7 days, S&P 5 days).
    On weekends, S&P return = 0 (market closed, position unchanged).

    rebalance="window": at window start allocate W_btc + W_sp, then let
        each drift independently. Portfolio value = W_btc * (btc_eq /
        btc_eq[0]) * 10_000 + W_sp * (sp_eq / sp_eq[0]) * 10_000.
    rebalance="daily": each day, reset weights to (W_btc, W_sp). Compute
        daily returns of each component, combine as the weighted sum,
        then compound. Equivalent to instantaneous rebalancing every day.
    """
    if btc_equity.empty or sp_equity.empty:
        return pd.Series(dtype=float)

    # Normalize both so they start at 1.0
    btc_norm = btc_equity / btc_equity.iloc[0]
    sp_norm = sp_equity / sp_equity.iloc[0]

    # Reindex S&P onto BTC's calendar with forward-fill. This treats
    # weekends/holidays as "position unchanged, no return" for S&P.
    sp_reindexed = sp_norm.reindex(btc_norm.index, method="ffill")
    # Before the first S&P date in the window, sp_reindexed is NaN —
    # fill with 1.0 (starting weight, no return yet).
    sp_reindexed = sp_reindexed.fillna(1.0)

    if rebalance == "window":
        return (w_btc * btc_norm + w_sp * sp_reindexed) * 10_000.0

    # daily rebalancing
    btc_returns = btc_norm.pct_change().fillna(0)
    sp_returns = sp_reindexed.pct_change().fillna(0)
    port_returns = w_btc * btc_returns + w_sp * sp_returns
    port_equity = (1.0 + port_returns).cumprod() * 10_000.0
    return port_equity


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    btc_prices = store.load(BTC_SYMBOL, "1d").sort_index()
    btc_prices = btc_prices.loc[(btc_prices.index >= START) & (btc_prices.index <= END)]
    sp_prices = store.load(SP_SYMBOL, "1d").sort_index()
    sp_prices = sp_prices.loc[(sp_prices.index >= START) & (sp_prices.index <= END)]

    print(f"BTC: {len(btc_prices)} bars ({btc_prices.index[0].date()} → {btc_prices.index[-1].date()})")
    print(f"SP : {len(sp_prices)} bars ({sp_prices.index[0].date()} → {sp_prices.index[-1].date()})")

    # Window selection uses BTC indices, but the window is defined by
    # CALENDAR DATES so we can slice S&P by the same date range even
    # though S&P has fewer bars per year (no weekends). BTC is the
    # binding warmup constraint (hybrid needs 1000 bars of history).
    vol_window = 10
    homc_train_window = 1000
    warmup_pad = 5
    six_months = 126
    n_windows = 16
    seed = 42

    min_start = homc_train_window + vol_window + warmup_pad
    max_start = len(btc_prices) - six_months - 1
    starts = draw_non_overlapping_starts(
        seed=seed,
        min_start=min_start,
        max_start=max_start,
        window_len=six_months,
        n_windows=n_windows,
    )

    print(f"Running {n_windows} random windows × {len(WEIGHT_GRID)} weights × 2 rebalance modes")
    print()

    rows: list[dict] = []
    for i, start_i in enumerate(starts, start=1):
        end_i = start_i + six_months
        window_start_ts = btc_prices.index[start_i].normalize()
        window_end_ts = btc_prices.index[end_i - 1].normalize()
        print(f"  window {i}/{n_windows}: {window_start_ts.date()} → {window_end_ts.date()}")

        btc_eq = _run_btc_strategy_on_window(btc_prices, window_start_ts, window_end_ts)
        sp_eq = _run_sp_bh_on_window(sp_prices, window_start_ts, window_end_ts)

        if btc_eq.empty:
            print("    skipping — BTC equity curve empty")
            continue
        if sp_eq.empty:
            print("    skipping — S&P equity curve empty")
            continue

        # Per-component metrics for reference
        btc_metrics = compute_metrics(
            btc_eq,
            [],
            risk_free_rate=historical_usd_rate("2018-2024"),
            periods_per_year=365.0,
        )
        sp_metrics = compute_metrics(
            sp_eq,
            [],
            risk_free_rate=historical_usd_rate("2018-2024"),
            periods_per_year=252.0,
        )

        for w_btc, w_sp in WEIGHT_GRID:
            for rebalance in ("window", "daily"):
                port_eq = _combine_portfolio(btc_eq, sp_eq, w_btc, w_sp, rebalance)
                if port_eq.empty:
                    continue
                m = compute_metrics(
                    port_eq,
                    [],
                    risk_free_rate=historical_usd_rate("2018-2024"),
                    periods_per_year=252.0,
                )
                rows.append({
                    "rebalance": rebalance,
                    "btc_weight": w_btc,
                    "sp_weight": w_sp,
                    "window_idx": i,
                    "window_start": str(window_start_ts.date()),
                    "window_end": str(window_end_ts.date()),
                    "final_equity": float(m.final_equity),
                    "cagr": float(m.cagr),
                    "sharpe": float(m.sharpe),
                    "max_dd": float(m.max_drawdown),
                    "btc_cagr": float(btc_metrics.cagr),
                    "btc_sharpe": float(btc_metrics.sharpe),
                    "sp_cagr": float(sp_metrics.cagr),
                    "sp_sharpe": float(sp_metrics.sharpe),
                })

    df = pd.DataFrame(rows)
    print()
    print(f"Collected {len(df)} rows")

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "btc_sp500_portfolio.parquet"
    df.to_parquet(data_path, index=False)
    print(f"Saved raw data to {data_path}")

    if df.empty:
        print("No successful windows — check the input data and script logic")
        return

    # Aggregate: median metrics per (rebalance, weight) combo
    agg = (
        df.groupby(["rebalance", "btc_weight", "sp_weight"])
        .agg(
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            mean_cagr=("cagr", "mean"),
            median_cagr=("cagr", "median"),
            mean_max_dd=("max_dd", "mean"),
            positive=("cagr", lambda s: int((s > 0).sum())),
            n=("sharpe", "count"),
        )
        .reset_index()
    )

    print()
    print("=" * 120)
    print("Aggregate results — 16 random 6-month windows, seed 42")
    print("=" * 120)
    for rebal in ["window", "daily"]:
        sub = agg[agg["rebalance"] == rebal].sort_values("btc_weight", ascending=False)
        print()
        print(f"Rebalance mode: {rebal}")
        print("-" * 120)
        print(
            f"{'BTC':>5} {'SP':>5}  {'mean Sh':>8} {'median Sh':>10} "
            f"{'mean CAGR':>11} {'median CAGR':>13} {'mean MDD':>10} {'pos/N':>8}"
        )
        for _, r in sub.iterrows():
            print(
                f"{r['btc_weight'] * 100:>4.0f}% {r['sp_weight'] * 100:>4.0f}%  "
                f"{r['mean_sharpe']:>8.2f} {r['median_sharpe']:>10.2f} "
                f"{r['mean_cagr'] * 100:>10.1f}% {r['median_cagr'] * 100:>12.1f}% "
                f"{r['mean_max_dd'] * 100:>9.1f}% "
                f"{int(r['positive']):>3d}/{int(r['n'])}"
            )

    # Headline: which combination has the best median Sharpe?
    print()
    print("=" * 120)
    print("Winner per rebalance mode by median Sharpe")
    print("=" * 120)
    for rebal in ["window", "daily"]:
        sub = agg[agg["rebalance"] == rebal]
        best = sub.loc[sub["median_sharpe"].idxmax()]
        btc_only = sub[(sub["btc_weight"] == 1.0)].iloc[0]
        sp_only = sub[(sub["sp_weight"] == 1.0)].iloc[0]
        print(
            f"  [{rebal}] best: {best['btc_weight'] * 100:.0f}/{best['sp_weight'] * 100:.0f} "
            f"(BTC/SP) median Sharpe {best['median_sharpe']:.2f}"
        )
        print(
            f"           BTC-only median Sharpe: {btc_only['median_sharpe']:.2f}  |  "
            f"SP-only median Sharpe: {sp_only['median_sharpe']:.2f}"
        )
        if best["median_sharpe"] > btc_only["median_sharpe"]:
            lift = best["median_sharpe"] - btc_only["median_sharpe"]
            print(
                f"           → Mixed portfolio beats BTC-only by {lift:+.2f} Sharpe"
            )
        elif best["median_sharpe"] > sp_only["median_sharpe"]:
            lift = best["median_sharpe"] - sp_only["median_sharpe"]
            print(
                f"           → Mixed portfolio beats SP-only by {lift:+.2f} Sharpe but not BTC"
            )
        else:
            print("           → Neither component is beaten by any mix")


if __name__ == "__main__":
    main()
