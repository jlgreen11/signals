"""Multi-seed robustness check for the BTC deep sweep winners.

The single-seed sweep in scripts/btc_deep_sweep.py identified several
apparent winners:

  - Dim 5: buy=25, sell=-20 (and sell=-25) → median Sharpe 2.40
           vs current default sell=-35 at 2.15 (+0.25 improvement)
  - Dim 2: composite 5×5 / train_window=1000 / alpha=0.01 → 1.59
           vs current default composite 3×3/252/0.01 at 1.44 (+0.15)
  - Dim 3: H-Vol @ q=0.70 + max_long=1.25 → Sharpe 2.15 (same as
           max_long=1.0) but CAGR +216% vs +156% (+60pp)
  - Dim 4: retrain_freq=14 → 2.16 vs default 21 at 2.15 (+0.01, noise)

And the portfolio experiment found:
  - 40/60 BTC/SP with daily rebalancing → median Sharpe 2.44, the
    project record

Tier-1S established that single-seed results can be data-mining
artifacts (HOMC@order=6 on S&P looked great at seed 42 but failed 2/4
alternative seeds). This script re-runs all the "winners" plus the
baseline at seeds {7, 100, 999} and reports whether each holds up.

Configs under test:
  Baseline:  H-Vol @ q=0.70, buy=25, sell=-35, max_long=1.0, retrain=21
  New-A:     H-Vol @ q=0.70, buy=25, sell=-20 (threshold winner)
  New-B:     H-Vol @ q=0.70, buy=25, sell=-25 (threshold co-winner)
  New-C:     H-Vol @ q=0.70, buy=25, sell=-35, max_long=1.25 (leverage)
  New-D:     Composite 5×5, train_window=1000, alpha=0.01
  New-E:     H-Vol @ q=0.70 with retrain_freq=14
  New-F:     Combined: sell=-20 AND max_long=1.25

Plus portfolio robustness (separate path):
  Port-A:    40/60 BTC/SP daily rebalance
  Port-B:    50/50 BTC/SP daily rebalance
  Port-C:    60/40 BTC/SP daily rebalance

Saves all raw per-window results to
scripts/data/btc_deep_sweep_robustness.parquet.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.config import SETTINGS
from signals.data.storage import DataStore

BTC_SYMBOL = "BTC-USD"
SP_SYMBOL = "^GSPC"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")
SEEDS = [42, 7, 100, 999]  # 42 matches the original sweep
N_WINDOWS = 16
SIX_MONTHS = 126
VOL_WINDOW = 10
WARMUP_PAD = 5
TRAIN_WINDOW = 1000


@dataclass
class NamedConfig:
    name: str
    description: str
    cfg: BacktestConfig


def _build_configs() -> list[NamedConfig]:
    base_kwargs = dict(
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

    baseline = BacktestConfig(**base_kwargs)
    new_a = BacktestConfig(**{**base_kwargs, "sell_threshold_bps": -20.0})
    new_b = BacktestConfig(**{**base_kwargs, "sell_threshold_bps": -25.0})
    new_c = BacktestConfig(**{**base_kwargs, "max_long": 1.25})
    new_d = BacktestConfig(
        model_type="composite",
        train_window=TRAIN_WINDOW,
        retrain_freq=21,
        return_bins=5,
        volatility_bins=5,
        vol_window=VOL_WINDOW,
        laplace_alpha=0.01,
    )
    new_e = BacktestConfig(**{**base_kwargs, "retrain_freq": 14})
    new_f = BacktestConfig(
        **{**base_kwargs, "sell_threshold_bps": -20.0, "max_long": 1.25}
    )

    return [
        NamedConfig("baseline", "H-Vol @ q=0.70 production default", baseline),
        NamedConfig("new-A", "sell_bps=-20 (threshold winner)", new_a),
        NamedConfig("new-B", "sell_bps=-25 (threshold co-winner)", new_b),
        NamedConfig("new-C", "max_long=1.25 (leverage)", new_c),
        NamedConfig("new-D", "composite 5×5 / 1000-bar window", new_d),
        NamedConfig("new-E", "retrain_freq=14", new_e),
        NamedConfig("new-F", "sell=-20 + max_long=1.25 (combined)", new_f),
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
        result = BacktestEngine(cfg).run(engine_input, symbol=BTC_SYMBOL)
    except Exception as e:
        print(f"    engine error: {e}")
        return compute_metrics(pd.Series(dtype=float), [])
    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(eq_rebased, [])


def _window_starts_for_seed(prices: pd.DataFrame, seed: int) -> list[int]:
    min_start = TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = len(prices) - SIX_MONTHS - 1
    rng = random.Random(seed)
    return sorted(rng.sample(range(min_start, max_start), N_WINDOWS))


def _run_strategy_sweep(
    configs: list[NamedConfig], prices: pd.DataFrame
) -> list[dict]:
    rows: list[dict] = []
    total = len(configs) * len(SEEDS)
    i = 0
    t0 = time.time()
    for cfg in configs:
        for seed in SEEDS:
            i += 1
            starts = _window_starts_for_seed(prices, seed)
            sharpes: list[float] = []
            cagrs: list[float] = []
            for win_idx, start_i in enumerate(starts):
                end_i = start_i + SIX_MONTHS
                m = _run_on_window(cfg.cfg, prices, start_i, end_i)
                rows.append({
                    "kind": "strategy",
                    "config_name": cfg.name,
                    "description": cfg.description,
                    "seed": seed,
                    "window_idx": win_idx,
                    "window_start": str(prices.index[start_i].date()),
                    "window_end": str(prices.index[end_i - 1].date()),
                    "cagr": float(m.cagr),
                    "sharpe": float(m.sharpe),
                    "max_dd": float(m.max_drawdown),
                    "n_trades": int(m.n_trades),
                    "final_equity": float(m.final_equity),
                })
                sharpes.append(float(m.sharpe))
                cagrs.append(float(m.cagr))
            median_sh = float(np.median(sharpes))
            median_cagr = float(np.median(cagrs))
            pos = int(sum(1 for c in cagrs if c > 0))
            elapsed = time.time() - t0
            print(
                f"  [{i:2d}/{total}] {cfg.name:<10} seed={seed:>4d}  "
                f"median Sh {median_sh:+5.2f}  median CAGR {median_cagr * 100:+6.1f}%  "
                f"pos {pos:>2d}/{N_WINDOWS}  ({elapsed:4.0f}s)"
            )
    return rows


def _run_portfolio_sweep(
    btc_prices: pd.DataFrame, sp_prices: pd.DataFrame
) -> list[dict]:
    """Run the 40/60, 50/50, 60/40 daily-rebalance portfolios across
    all 4 seeds. The BTC strategy inside each portfolio is the baseline
    H-Vol @ q=0.70 — the same config that scored 2.44 at seed 42."""
    base_cfg = BacktestConfig(
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
    weights = [(0.40, 0.60), (0.50, 0.50), (0.60, 0.40)]

    rows: list[dict] = []
    total = len(weights) * len(SEEDS)
    idx = 0
    t0 = time.time()
    for w_btc, w_sp in weights:
        for seed in SEEDS:
            idx += 1
            starts = _window_starts_for_seed(btc_prices, seed)
            sharpes: list[float] = []
            for win_idx, start_i in enumerate(starts):
                end_i = start_i + SIX_MONTHS
                window_start_ts = btc_prices.index[start_i].normalize()
                window_end_ts = btc_prices.index[end_i - 1].normalize()

                # BTC strategy
                warmup_bars = base_cfg.train_window + base_cfg.vol_window + WARMUP_PAD
                slice_mask = (
                    btc_prices.index >= window_start_ts - pd.Timedelta(days=warmup_bars * 2)
                ) & (btc_prices.index <= window_end_ts)
                engine_input = btc_prices[slice_mask]
                try:
                    result = BacktestEngine(base_cfg).run(
                        engine_input, symbol=BTC_SYMBOL
                    )
                    btc_eq = result.equity_curve.loc[
                        result.equity_curve.index >= window_start_ts
                    ]
                    if btc_eq.empty or btc_eq.iloc[0] <= 0:
                        continue
                    btc_eq = (btc_eq / btc_eq.iloc[0]) * 10_000.0
                    btc_eq.index = btc_eq.index.normalize()
                except Exception as e:
                    print(f"    btc engine error: {e}")
                    continue

                # S&P B&H
                sp_mask = (sp_prices.index >= window_start_ts) & (
                    sp_prices.index <= window_end_ts
                )
                sp_window = sp_prices[sp_mask]
                if sp_window.empty:
                    continue
                sp_eq = (sp_window["close"] / sp_window["close"].iloc[0]) * 10_000.0
                sp_eq.index = sp_eq.index.normalize()

                # Combine: daily rebalance
                btc_norm = btc_eq / btc_eq.iloc[0]
                sp_reindexed = (sp_eq / sp_eq.iloc[0]).reindex(
                    btc_norm.index, method="ffill"
                ).fillna(1.0)
                btc_rets = btc_norm.pct_change().fillna(0)
                sp_rets = sp_reindexed.pct_change().fillna(0)
                port_rets = w_btc * btc_rets + w_sp * sp_rets
                port_eq = (1.0 + port_rets).cumprod() * 10_000.0
                m = compute_metrics(port_eq, [])

                rows.append({
                    "kind": "portfolio",
                    "config_name": f"port_{int(w_btc * 100):02d}_{int(w_sp * 100):02d}",
                    "description": f"{int(w_btc * 100)}/{int(w_sp * 100)} BTC/SP daily rebalance",
                    "seed": seed,
                    "window_idx": win_idx,
                    "window_start": str(window_start_ts.date()),
                    "window_end": str(window_end_ts.date()),
                    "cagr": float(m.cagr),
                    "sharpe": float(m.sharpe),
                    "max_dd": float(m.max_drawdown),
                    "n_trades": 0,
                    "final_equity": float(m.final_equity),
                })
                sharpes.append(float(m.sharpe))

            median_sh = float(np.median(sharpes)) if sharpes else float("nan")
            elapsed = time.time() - t0
            print(
                f"  [{idx:2d}/{total}] port_{int(w_btc*100)}_{int(w_sp*100)} "
                f"seed={seed:>4d}  median Sh {median_sh:+5.2f}  "
                f"n_valid={len(sharpes)}  ({elapsed:4.0f}s)"
            )
    return rows


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    btc_prices = store.load(BTC_SYMBOL, "1d").sort_index()
    btc_prices = btc_prices.loc[(btc_prices.index >= START) & (btc_prices.index <= END)]
    sp_prices = store.load(SP_SYMBOL, "1d").sort_index()
    sp_prices = sp_prices.loc[(sp_prices.index >= START) & (sp_prices.index <= END)]
    print(f"BTC: {len(btc_prices)} bars, SP: {len(sp_prices)} bars")
    print()

    configs = _build_configs()
    print(f"Running {len(configs)} BTC strategy configs × {len(SEEDS)} seeds × {N_WINDOWS} windows")
    print(f"Plus 3 portfolio configs × {len(SEEDS)} seeds × {N_WINDOWS} windows")
    print()

    print("=" * 80)
    print("Phase A — BTC strategy robustness")
    print("=" * 80)
    strategy_rows = _run_strategy_sweep(configs, btc_prices)

    print()
    print("=" * 80)
    print("Phase B — Portfolio robustness")
    print("=" * 80)
    portfolio_rows = _run_portfolio_sweep(btc_prices, sp_prices)

    all_rows = strategy_rows + portfolio_rows
    df = pd.DataFrame(all_rows)
    print()
    print(f"Collected {len(df)} rows total")

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "btc_deep_sweep_robustness.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved: {path}")

    # Aggregate per (config_name, seed)
    agg = (
        df.groupby(["kind", "config_name", "description", "seed"])
        .agg(
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            median_cagr=("cagr", "median"),
            mean_max_dd=("max_dd", "mean"),
            positive=("cagr", lambda s: int((s > 0).sum())),
            n=("sharpe", "count"),
        )
        .reset_index()
    )

    print()
    print("=" * 100)
    print("Robustness table — median Sharpe per config per seed")
    print("=" * 100)
    print(
        f"{'config':<12} {'description':<40}  "
        f"{'seed 42':>8} {'seed 7':>8} {'seed 100':>9} {'seed 999':>9}"
    )
    for cfg_name in agg["config_name"].unique():
        sub = agg[agg["config_name"] == cfg_name]
        desc = sub["description"].iloc[0]
        row = {int(r["seed"]): r["median_sharpe"] for _, r in sub.iterrows()}
        print(
            f"{cfg_name:<12} {desc:<40}  "
            f"{row.get(42, float('nan')):>8.2f} "
            f"{row.get(7, float('nan')):>8.2f} "
            f"{row.get(100, float('nan')):>9.2f} "
            f"{row.get(999, float('nan')):>9.2f}"
        )

    # Identify robust winners: beat baseline at all 4 seeds
    print()
    print("=" * 100)
    print("Robust winners (beat baseline median Sharpe at ALL 4 seeds)")
    print("=" * 100)

    def _get_medians(cfg_name: str) -> dict[int, float]:
        sub = agg[agg["config_name"] == cfg_name]
        return {int(r["seed"]): float(r["median_sharpe"]) for _, r in sub.iterrows()}

    baseline_medians = _get_medians("baseline")
    print(f"Baseline medians: {baseline_medians}")
    print()

    robust_winners: list[tuple[str, str, dict[int, float]]] = []
    for cfg_name in agg["config_name"].unique():
        if cfg_name == "baseline":
            continue
        medians = _get_medians(cfg_name)
        if all(medians.get(s, -np.inf) >= baseline_medians.get(s, np.inf) for s in SEEDS):
            desc = agg[agg["config_name"] == cfg_name]["description"].iloc[0]
            robust_winners.append((cfg_name, desc, medians))

    if robust_winners:
        print("✓ Configs that beat baseline at EVERY seed:")
        for name, desc, medians in robust_winners:
            avg_lift = np.mean(
                [medians[s] - baseline_medians[s] for s in SEEDS]
            )
            print(
                f"  {name:<12} {desc:<40}  avg Sharpe lift: {avg_lift:+.3f}  "
                f"seeds: {medians}"
            )
    else:
        print("✗ No config beats baseline at every seed. All apparent seed-42")
        print("  winners were data-mining artifacts — baseline remains the default.")


if __name__ == "__main__":
    main()
