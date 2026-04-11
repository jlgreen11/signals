"""Macro feature experiment: can a VIX-based risk-off overlay improve
the BTC hybrid?

Hypothesis: VIX is a known leading indicator of cross-asset risk-off
regimes. BTC isn't directly measured by VIX but crypto does experience
contagion during equity panics (March 2020, Sept 2022). A simple
overlay that forces the BTC hybrid to FLAT when VIX is in its training-
distribution top quartile could cut drawdowns without giving up much
bull participation.

This script tests that specifically. It runs the production BTC hybrid
default, then ALSO runs a "vix-gated" variant that forces target=0
whenever VIX is above its 75th-percentile training threshold. Both on
the same 16 random 6-month windows at 4 seeds for robustness.

If the VIX overlay reduces drawdowns without hurting Sharpe, it's a
useful macro signal. If it hurts both, the VIX regime doesn't transfer
to BTC usefully.

Saves per-window results to scripts/data/btc_macro_vix_overlay.parquet.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from _window_sampler import draw_non_overlapping_starts

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

BTC_SYMBOL = "BTC-USD"
VIX_SYMBOL = "^VIX"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")
SEEDS = [42, 7, 100, 999]
N_WINDOWS = 16
SIX_MONTHS = 126
TRAIN_WINDOW = 1000
VOL_WINDOW = 10
WARMUP_PAD = 5


def _btc_cfg() -> BacktestConfig:
    return BacktestConfig(
        model_type="hybrid",
        train_window=TRAIN_WINDOW,
        retrain_freq=21,
        n_states=5,
        order=5,
        return_bins=3,
        volatility_bins=3,
        vol_window=VOL_WINDOW,
        laplace_alpha=0.01,
        hybrid_routing_strategy="vol",
        hybrid_vol_quantile=0.70,
    )


def _apply_vix_overlay(
    btc_equity: pd.Series,
    vix: pd.Series,
    window_start: pd.Timestamp,
    vix_threshold: float,
) -> pd.Series:
    """Given a backtest equity curve and VIX series, rebuild the equity
    curve assuming positions are forced to 0 on days when VIX exceeds
    the threshold. This is a post-hoc overlay — it doesn't change the
    signals, just kills position during risk-off days.

    Approximation: we estimate daily returns from the equity curve,
    zero out returns on risk-off days, and recompound. This loses some
    accuracy (the overlay should really be applied before the portfolio
    layer sizes), but it's a good first-order estimate.
    """
    btc_norm = btc_equity / btc_equity.iloc[0]
    btc_norm.index = btc_norm.index.normalize()
    btc_returns = btc_norm.pct_change().fillna(0)

    # Align VIX to BTC's dates with forward-fill for weekends
    vix_aligned = vix.copy()
    vix_aligned.index = vix_aligned.index.normalize()
    vix_on_btc = vix_aligned.reindex(btc_returns.index, method="ffill")

    # Risk-off mask
    risk_off = vix_on_btc > vix_threshold
    # Zero out returns on risk-off days (flat position, no move)
    gated_returns = btc_returns.where(~risk_off, 0.0)
    overlay_equity = (1.0 + gated_returns).cumprod() * 10_000.0
    return overlay_equity


def _run_window(
    btc_prices: pd.DataFrame,
    vix_prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    vix_threshold_quantile: float = 0.75,
) -> tuple[Metrics, Metrics]:
    """Returns (baseline_metrics, vix_overlay_metrics)."""
    cfg = _btc_cfg()
    slice_start = start_i - TRAIN_WINDOW - VOL_WINDOW - WARMUP_PAD
    if slice_start < 0:
        slice_start = 0
    engine_input = btc_prices.iloc[slice_start:end_i]
    eval_start_ts = btc_prices.index[start_i]

    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=BTC_SYMBOL)
    except Exception as e:
        print(f"    btc engine error: {e}")
        empty = compute_metrics(pd.Series(dtype=float), [])
        return empty, empty

    btc_eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if btc_eq.empty or btc_eq.iloc[0] <= 0:
        empty = compute_metrics(pd.Series(dtype=float), [])
        return empty, empty
    btc_eq = (btc_eq / btc_eq.iloc[0]) * 10_000.0
    baseline_metrics = compute_metrics(
        btc_eq,
        [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=365.0,
    )

    # Compute VIX threshold from the TRAINING window (no lookahead)
    training_end_ts = btc_prices.index[start_i - 1]
    training_start_ts = training_end_ts - pd.Timedelta(days=TRAIN_WINDOW * 2)
    vix_training = vix_prices.loc[
        (vix_prices.index >= training_start_ts) & (vix_prices.index <= training_end_ts)
    ]["close"]
    if vix_training.empty:
        return baseline_metrics, compute_metrics(pd.Series(dtype=float), [])
    vix_threshold = float(vix_training.quantile(vix_threshold_quantile))

    # Apply the overlay
    vix_series = vix_prices["close"]
    overlay_eq = _apply_vix_overlay(btc_eq, vix_series, eval_start_ts, vix_threshold)
    overlay_metrics = compute_metrics(
        overlay_eq,
        [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=365.0,
    )

    return baseline_metrics, overlay_metrics


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    btc_prices = store.load(BTC_SYMBOL, "1d").sort_index()
    btc_prices = btc_prices.loc[(btc_prices.index >= START) & (btc_prices.index <= END)]
    vix_prices = store.load(VIX_SYMBOL, "1d").sort_index()
    vix_prices = vix_prices.loc[(vix_prices.index >= START) & (vix_prices.index <= END)]
    print(f"BTC: {len(btc_prices)} bars  VIX: {len(vix_prices)} bars")

    min_start = TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = len(btc_prices) - SIX_MONTHS - 1

    rows: list[dict] = []
    t0 = time.time()
    for seed in SEEDS:
        starts = draw_non_overlapping_starts(
            seed=seed,
            min_start=min_start,
            max_start=max_start,
            window_len=SIX_MONTHS,
            n_windows=N_WINDOWS,
        )
        for win_idx, start_i in enumerate(starts):
            end_i = start_i + SIX_MONTHS
            baseline, overlay = _run_window(btc_prices, vix_prices, start_i, end_i)
            rows.append({
                "seed": seed,
                "window_idx": win_idx,
                "window_start": str(btc_prices.index[start_i].date()),
                "window_end": str(btc_prices.index[end_i - 1].date()),
                "baseline_sharpe": float(baseline.sharpe),
                "baseline_cagr": float(baseline.cagr),
                "baseline_max_dd": float(baseline.max_drawdown),
                "overlay_sharpe": float(overlay.sharpe),
                "overlay_cagr": float(overlay.cagr),
                "overlay_max_dd": float(overlay.max_drawdown),
            })
        elapsed = time.time() - t0
        sub = pd.DataFrame([r for r in rows if r["seed"] == seed])
        print(
            f"  seed {seed:>4d}: baseline median Sh "
            f"{sub['baseline_sharpe'].median():+5.2f}  overlay median Sh "
            f"{sub['overlay_sharpe'].median():+5.2f}  ({elapsed:4.0f}s)"
        )

    df = pd.DataFrame(rows)
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "btc_macro_vix_overlay.parquet"
    df.to_parquet(path, index=False)
    print(f"\nSaved: {path}")

    print()
    print("=" * 100)
    print("Cross-seed summary")
    print("=" * 100)
    print(f"{'seed':>6}  {'baseline Sh':>12}  {'overlay Sh':>11}  {'Δ Sh':>7}  "
          f"{'baseline MDD':>13}  {'overlay MDD':>12}  {'Δ MDD':>8}")
    for seed in SEEDS:
        sub = df[df["seed"] == seed]
        bl_sh = sub["baseline_sharpe"].median()
        ov_sh = sub["overlay_sharpe"].median()
        bl_mdd = sub["baseline_max_dd"].mean()
        ov_mdd = sub["overlay_max_dd"].mean()
        print(
            f"{seed:>6}  {bl_sh:>12.2f}  {ov_sh:>11.2f}  {ov_sh - bl_sh:+7.2f}  "
            f"{bl_mdd * 100:>12.1f}%  {ov_mdd * 100:>11.1f}%  "
            f"{(ov_mdd - bl_mdd) * 100:+7.1f}pp"
        )

    bl_avg = df.groupby("seed")["baseline_sharpe"].median().mean()
    ov_avg = df.groupby("seed")["overlay_sharpe"].median().mean()
    bl_mdd_avg = df.groupby("seed")["baseline_max_dd"].mean().mean()
    ov_mdd_avg = df.groupby("seed")["overlay_max_dd"].mean().mean()
    print()
    print(f"Average median-Sharpe across seeds: baseline {bl_avg:.2f}, overlay {ov_avg:.2f}")
    print(f"Average mean-MDD across seeds: baseline {bl_mdd_avg*100:+.1f}%, overlay {ov_mdd_avg*100:+.1f}%")

    if ov_avg > bl_avg:
        print(f"✓ VIX overlay lifts avg Sharpe by {ov_avg - bl_avg:+.2f}")
    else:
        print(f"✗ VIX overlay hurts avg Sharpe by {ov_avg - bl_avg:+.2f}")
    if ov_mdd_avg > bl_mdd_avg:
        print(f"✓ VIX overlay reduces avg MDD by {(ov_mdd_avg - bl_mdd_avg)*100:+.1f}pp")
    else:
        print(f"✗ VIX overlay deepens avg MDD by {(ov_mdd_avg - bl_mdd_avg)*100:+.1f}pp")


if __name__ == "__main__":
    main()
