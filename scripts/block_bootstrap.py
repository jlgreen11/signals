"""Tier-A3: Moving-block bootstrap for 95% CI on annualized Sharpe.

Addresses the SKEPTIC_REVIEW critique that the 16-window random evaluator
reports point estimates (median Sharpe ~2.15 for BTC H-Vol @ q=0.70) without
any confidence intervals. A moving-block bootstrap over each window's daily
returns preserves short-horizon autocorrelation inside the block while
resampling blocks with replacement — the standard tool for building CIs on
serially-correlated time-series statistics.

Procedure
---------
1. Pick BTC-USD H-Vol at q=0.70 with seed=42. Use the same 16 random
   6-month windows as `scripts/random_window_eval.py`.
2. For each window, run the BacktestEngine once to obtain a ~126-bar daily
   return series (equity_curve.pct_change()).
3. For B=1000 bootstrap replicates per window, draw blocks of length 21
   (one month) with replacement from the real daily-return series, glue
   them until length matches the real series, and compute annualized
   Sharpe on that synthetic return series. Block length 21 is a compromise
   between preserving autocorrelation and having enough block draws per
   replicate for variance in the bootstrap distribution.
4. Per window: report observed Sharpe, bootstrap mean, 2.5% / 97.5%
   percentiles.
5. Aggregate: build a bootstrap CI on the *median across windows* of the
   annualized Sharpe. For each of B replicates, compute one Sharpe per
   window using the same replicate index (i.e. the b-th replicate across
   all 16 windows contributes a single median), then take percentiles of
   those 1000 per-replicate medians. This gives a CI on the headline
   "median Sharpe 2.15" number that matches the agg reported in
   random_window_eval.py.
6. Persist the per-window rows to scripts/data/block_bootstrap.parquet.

Run as:
    python scripts/block_bootstrap.py              # B=1000, full
    python scripts/block_bootstrap.py --quick      # B=100, smoke test
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import sharpe_ratio
from signals.config import SETTINGS
from signals.data.storage import DataStore

SYMBOL = "BTC-USD"
START_TS = pd.Timestamp("2015-01-01", tz="UTC")
END_TS = pd.Timestamp("2024-12-31", tz="UTC")
SEED = 42
N_WINDOWS = 16
SIX_MONTHS = 126
BLOCK_LEN = 21                 # one month, preserves short-horizon autocorr
PERIODS_PER_YEAR = 365.0       # BTC trades daily incl. weekends
VOL_WINDOW = 10
HOMC_TRAIN_WINDOW = 1000
WARMUP_PAD = 5
HVOL_QUANTILE = 0.70           # Tier-0e BTC-optimal


def _hvol_config() -> BacktestConfig:
    return BacktestConfig(
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
        hybrid_vol_quantile=HVOL_QUANTILE,
    )


def _pick_window_starts(n_bars: int, n_windows: int, seed: int) -> list[int]:
    min_start = HOMC_TRAIN_WINDOW + VOL_WINDOW + WARMUP_PAD
    max_start = n_bars - SIX_MONTHS - 1
    if max_start - min_start < n_windows:
        raise ValueError(
            f"Not enough bars for {n_windows} {SIX_MONTHS}-bar windows "
            f"(have {n_bars}, need min_start={min_start}, max_start={max_start})"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(range(min_start, max_start), n_windows))


def _run_strategy_returns(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
) -> pd.Series:
    """Return the strategy's daily pct-return series restricted to the eval window.

    Mirrors `_run_strategy_on_window` in scripts/random_window_eval.py so the
    daily returns feeding the bootstrap match the headline evaluator exactly.
    """
    slice_start = start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]

    result = BacktestEngine(cfg).run(engine_input, symbol=symbol)
    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return pd.Series(dtype=float)
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    returns = eq_rebased.pct_change().dropna()
    return returns


def _moving_block_bootstrap(
    returns: np.ndarray,
    *,
    block_len: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Moving-block bootstrap: draw ceil(N/block_len) blocks of length
    `block_len` with replacement from all valid starting positions, then
    concatenate and truncate to N. Returns a (n_replicates, N) array of
    replicate return series.
    """
    n = len(returns)
    if n == 0:
        return np.zeros((n_replicates, 0), dtype=float)
    n_block_starts = max(1, n - block_len + 1)
    blocks_per_rep = int(np.ceil(n / block_len))
    # shape (n_replicates, blocks_per_rep)
    start_idx = rng.integers(0, n_block_starts, size=(n_replicates, blocks_per_rep))
    # build an (n_replicates, blocks_per_rep, block_len) index matrix of
    # positions into `returns`, then reshape to (n_replicates, blocks_per_rep*block_len)
    offsets = np.arange(block_len)
    # broadcasting: (n_replicates, blocks_per_rep, 1) + (block_len,) -> (.., block_len)
    idx = start_idx[:, :, None] + offsets[None, None, :]
    idx = np.clip(idx, 0, n - 1)                 # safe if block_len > n
    replicates = returns[idx]                    # (n_replicates, blocks_per_rep, block_len)
    replicates = replicates.reshape(n_replicates, -1)[:, :n]
    return replicates


def _annualized_sharpe(returns: np.ndarray, periods_per_year: float) -> np.ndarray:
    """Vectorized annualized Sharpe across a (R, N) replicate matrix."""
    if returns.ndim == 1:
        returns = returns[None, :]
    mean = returns.mean(axis=1)
    std = returns.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(std > 0, mean / std * np.sqrt(periods_per_year), 0.0)
    return sr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke-test mode: B=100 instead of 1000",
    )
    args = parser.parse_args()
    n_replicates = 100 if args.quick else 1000

    print(
        f"Block bootstrap — {SYMBOL} H-Vol @ q={HVOL_QUANTILE}, seed={SEED}, "
        f"{N_WINDOWS} windows, B={n_replicates}, block_len={BLOCK_LEN}"
    )

    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START_TS) & (prices.index <= END_TS)]
    if prices.empty:
        raise SystemExit(f"No data for {SYMBOL} in {START_TS.date()}..{END_TS.date()}")

    starts = _pick_window_starts(len(prices), N_WINDOWS, SEED)
    cfg = _hvol_config()

    # Per-bootstrap-index Sharpe matrix: rows = window idx, cols = replicate idx.
    # Used to build the aggregate-median CI at the end.
    sharpe_mat = np.full((N_WINDOWS, n_replicates), np.nan, dtype=float)
    rng = np.random.default_rng(seed=SEED)

    rows: list[dict] = []
    t0 = time.time()
    for i, start_i in enumerate(starts):
        end_i = start_i + SIX_MONTHS
        window_prices = prices.iloc[start_i:end_i]
        w_start = window_prices.index[0].date()
        w_end = window_prices.index[-1].date()
        print(f"  window {i + 1}/{N_WINDOWS}: {w_start} -> {w_end}", flush=True)

        returns = _run_strategy_returns(cfg, prices, start_i, end_i, SYMBOL)
        if returns.empty:
            print("    (empty equity curve — skipping)")
            rows.append({
                "window_idx": i,
                "start": w_start,
                "end": w_end,
                "n_bars": 0,
                "observed_sharpe": np.nan,
                "boot_mean_sharpe": np.nan,
                "boot_std_sharpe": np.nan,
                "ci_lo_2p5": np.nan,
                "ci_hi_97p5": np.nan,
            })
            continue

        r = returns.to_numpy(dtype=float)
        obs = float(sharpe_ratio(pd.Series(r), PERIODS_PER_YEAR))

        reps = _moving_block_bootstrap(
            r, block_len=BLOCK_LEN, n_replicates=n_replicates, rng=rng
        )
        rep_sharpes = _annualized_sharpe(reps, PERIODS_PER_YEAR)
        sharpe_mat[i, :] = rep_sharpes

        lo, hi = np.percentile(rep_sharpes, [2.5, 97.5])
        mean_sr = float(rep_sharpes.mean())
        std_sr = float(rep_sharpes.std(ddof=1))

        print(
            f"    observed={obs:+.3f}  boot_mean={mean_sr:+.3f}  "
            f"95%CI=[{lo:+.3f}, {hi:+.3f}]  (n={len(r)})"
        )
        rows.append({
            "window_idx": i,
            "start": w_start,
            "end": w_end,
            "n_bars": int(len(r)),
            "observed_sharpe": obs,
            "boot_mean_sharpe": mean_sr,
            "boot_std_sharpe": std_sr,
            "ci_lo_2p5": float(lo),
            "ci_hi_97p5": float(hi),
        })

    df = pd.DataFrame(rows)

    # Aggregate bootstrap CI on the MEDIAN-across-windows Sharpe.
    # For each replicate index b, take the median Sharpe across all 16 windows
    # for that same b; then take percentiles over those 1000 medians.
    valid = ~np.all(np.isnan(sharpe_mat), axis=1)
    if valid.any():
        per_rep_median = np.nanmedian(sharpe_mat[valid, :], axis=0)
        med_obs = float(np.nanmedian(df["observed_sharpe"].to_numpy()))
        med_lo, med_hi = np.percentile(per_rep_median, [2.5, 97.5])
        med_mean = float(per_rep_median.mean())
    else:
        per_rep_median = np.array([])
        med_obs = med_mean = med_lo = med_hi = float("nan")

    print()
    print("=" * 80)
    print(f"Aggregate — median-across-windows Sharpe CI ({SYMBOL} H-Vol q={HVOL_QUANTILE})")
    print("=" * 80)
    print(f"  observed median Sharpe     : {med_obs:+.3f}")
    print(f"  bootstrap mean-of-medians  : {med_mean:+.3f}")
    print(f"  95% CI (percentile method) : [{med_lo:+.3f}, {med_hi:+.3f}]")
    print(f"  (elapsed: {time.time() - t0:.1f}s, B={n_replicates})")

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "block_bootstrap.parquet"

    # Persist per-window rows plus an aggregate row so downstream can read one file.
    agg_row = {
        "window_idx": -1,
        "start": None,
        "end": None,
        "n_bars": int(df["n_bars"].sum()) if not df.empty else 0,
        "observed_sharpe": med_obs,
        "boot_mean_sharpe": med_mean,
        "boot_std_sharpe": float(per_rep_median.std(ddof=1)) if per_rep_median.size else float("nan"),
        "ci_lo_2p5": float(med_lo),
        "ci_hi_97p5": float(med_hi),
    }
    out = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
