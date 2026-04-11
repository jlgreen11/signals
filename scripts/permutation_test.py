"""Tier-B1: Monte-Carlo permutation test against a directional-info null.

Addresses the SKEPTIC_REVIEW critique that the headline Sharpe numbers
for BTC H-Vol @ q=0.70 have no null-hypothesis significance test. If the
strategy is extracting real directional information, shuffling the day-to-
day return ordering (thereby destroying the directional structure but
keeping the return distribution intact) should collapse its Sharpe. If the
shuffled Sharpes cluster near the real Sharpe, the model is not predicting
direction — it's just vol-sizing or riding a drift.

Null construction
-----------------
For each of the 16 BTC 6-month windows at seed=42:
  * Compute the *bar-internal* structure on the real eval-window prices:
      - log open->close return   lr_oc[t] = log(close[t] / open[t])
      - log close->open gap      lr_co[t] = log(open[t] / close[t-1])     (t>=1)
      - log close->close return  lr_cc[t] = lr_co[t] + lr_oc[t]
  * Under H0, shuffle the TOTAL log close-to-close returns lr_cc across
    t = 1..N-1 (the first bar is the anchor and stays put). Split each
    shuffled lr_cc back into (overnight gap, intraday) in proportion to
    a re-ordered lr_oc (we shuffle lr_oc with the SAME permutation so the
    bar-internal ratio |oc|/|cc| that determines trade P&L is preserved).
  * Reconstruct closes via cumulative sum of shuffled lr_cc starting from
    real close[0]. Reconstruct opens from shuffled lr_oc:
        open[t]  = close[t]  * exp(-shuffled_lr_oc[t])
    i.e. open[t]/close[t] = exp(-lr_oc[t]) — preserves the bar-internal
    open/close relationship exactly as the permuted bar's original bar.
  * Bars 1..N-1 are replaced. We keep the real high/low/volume columns
    for schema compatibility (the engine only uses open/close).
  * History BEFORE the eval window is left untouched so the Markov models
    train on the real past and only *predict* into shuffled bars — which
    is exactly the test: can the model's out-of-sample directional signal
    survive when the target is random?

The strategy runs on this frankenstein price series; we compute its
Sharpe on the eval-window equity curve. Repeat N=200 times per window.

Test statistic
--------------
One-sided p-value = (1 + # {null Sharpe >= real Sharpe}) / (N + 1).
The "+1" is a standard permutation-test correction so p is never zero.

Aggregation
-----------
Per-window p-values are combined via Fisher's method:
    X^2 = -2 * sum(log(p_i))  ~  chi-squared with 2*k degrees of freedom
where k = number of valid windows. The combined p is the chi-squared
survival function at X^2.

Run as:
    python scripts/permutation_test.py              # N=200, full
    python scripts/permutation_test.py --quick      # N=20, smoke test
"""

from __future__ import annotations

import argparse
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
PERIODS_PER_YEAR = 365.0
VOL_WINDOW = 10
HOMC_TRAIN_WINDOW = 1000
WARMUP_PAD = 5
HVOL_QUANTILE = 0.70


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
            f"(have {n_bars})"
        )
    from _window_sampler import draw_non_overlapping_starts
    return draw_non_overlapping_starts(
        seed=seed,
        min_start=min_start,
        max_start=max_start,
        window_len=SIX_MONTHS,
        n_windows=n_windows,
    )


def _run_strategy_sharpe(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
) -> float:
    """Run the engine on `prices` and return the Sharpe of the eval-window
    equity curve. `start_i`/`end_i` are indices into `prices` (which may have
    its eval-window bars replaced by a permutation). The slice handed to the
    engine still includes the upstream train window so models see real history.
    """
    slice_start = start_i - cfg.train_window - cfg.vol_window - WARMUP_PAD
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]

    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=symbol)
    except Exception as e:
        print(f"    engine error: {e}")
        return float("nan")

    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return float("nan")
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    returns = eq_rebased.pct_change().dropna()
    if len(returns) < 2:
        return float("nan")
    return float(sharpe_ratio(returns, PERIODS_PER_YEAR))


def _permuted_window_prices(
    window: pd.DataFrame,
    perm: np.ndarray,
) -> pd.DataFrame:
    """Return a copy of `window` with open/close on bars 1..N-1 replaced by
    a permutation of the real close-to-close log returns.

    Shuffles lr_cc (close-to-close) AND lr_oc (open-to-close) with the same
    permutation index. Reconstructs:
      close[t] = close[0] * exp(cumsum(shuffled_lr_cc[1..t]))
      open[t]  = close[t] * exp(-shuffled_lr_oc[t])
    This keeps each bar's internal open/close ratio intact (so trade P&L is
    drawn from the real distribution of intraday moves) while randomizing the
    time-ordering, which destroys any directional predictability.
    """
    n = len(window)
    if n < 3:
        return window.copy()

    close = window["close"].to_numpy(dtype=float)
    open_ = window["open"].to_numpy(dtype=float)

    # log returns on bars 1..N-1 (N-1 values).
    lr_cc = np.log(close[1:]) - np.log(close[:-1])              # (N-1,)
    lr_oc = np.log(close[1:]) - np.log(open_[1:])               # (N-1,)  == log(close/open) on that bar

    # Apply the same permutation to both series so the (open/close) ratio
    # stays attached to its original log close-to-close move.
    shuf_lr_cc = lr_cc[perm]
    shuf_lr_oc = lr_oc[perm]

    new_close = np.empty(n, dtype=float)
    new_open = np.empty(n, dtype=float)
    new_close[0] = close[0]
    new_open[0] = open_[0]
    # reconstruct close from cumsum of log returns
    new_close[1:] = close[0] * np.exp(np.cumsum(shuf_lr_cc))
    # reconstruct open from the bar-internal open/close ratio
    new_open[1:] = new_close[1:] * np.exp(-shuf_lr_oc)

    out = window.copy()
    out["close"] = new_close
    out["open"] = new_open
    # keep high/low consistent with the new (open, close) range so the
    # engine's other consumers (if any) don't see low > close / high < open.
    if "high" in out.columns:
        out["high"] = np.maximum(new_open, new_close)
    if "low" in out.columns:
        out["low"] = np.minimum(new_open, new_close)
    return out


def _fisher_combine(pvals: np.ndarray) -> tuple[float, float]:
    """Fisher's method: X^2 = -2*sum(log(p)) ~ chi^2_{2k}.

    Returns (chi2_statistic, combined_pvalue). Small p-values mean the
    combined evidence rejects the null.
    """
    clean = pvals[~np.isnan(pvals)]
    if len(clean) == 0:
        return float("nan"), float("nan")
    # Floor at a tiny positive number to avoid log(0).
    clean = np.clip(clean, 1e-12, 1.0)
    x2 = float(-2.0 * np.sum(np.log(clean)))
    from scipy.stats import chi2
    p_combined = float(chi2.sf(x2, df=2 * len(clean)))
    return x2, p_combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke-test mode: N=20 permutations instead of 200",
    )
    args = parser.parse_args()
    n_perms = 20 if args.quick else 200

    print(
        f"Permutation test — {SYMBOL} H-Vol @ q={HVOL_QUANTILE}, seed={SEED}, "
        f"{N_WINDOWS} windows, N={n_perms} permutations each"
    )

    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START_TS) & (prices.index <= END_TS)]
    if prices.empty:
        raise SystemExit(f"No data for {SYMBOL}")

    starts = _pick_window_starts(len(prices), N_WINDOWS, SEED)
    cfg = _hvol_config()
    rng = np.random.default_rng(seed=SEED)

    rows: list[dict] = []
    t0 = time.time()
    for i, start_i in enumerate(starts):
        end_i = start_i + SIX_MONTHS
        window = prices.iloc[start_i:end_i]
        w_start = window.index[0].date()
        w_end = window.index[-1].date()
        print(f"  window {i + 1}/{N_WINDOWS}: {w_start} -> {w_end}", flush=True)

        real_sr = _run_strategy_sharpe(cfg, prices, start_i, end_i, SYMBOL)
        if np.isnan(real_sr):
            print("    (skipping: real Sharpe is NaN)")
            rows.append({
                "window_idx": i,
                "start": w_start,
                "end": w_end,
                "real_sharpe": np.nan,
                "null_mean_sharpe": np.nan,
                "null_std_sharpe": np.nan,
                "null_q05": np.nan,
                "null_q50": np.nan,
                "null_q95": np.nan,
                "p_value_one_sided": np.nan,
                "n_perms": 0,
            })
            continue

        n_bars = len(window)
        null_sharpes = np.full(n_perms, np.nan, dtype=float)
        for b in range(n_perms):
            perm = rng.permutation(n_bars - 1)           # shuffle lr_cc[1..N-1]
            shuffled_window = _permuted_window_prices(window, perm)

            # Splice back into the full prices frame so the engine still sees
            # the real train window before eval-window start.
            shuffled_prices = prices.copy()
            shuffled_prices.iloc[start_i:end_i] = shuffled_window.values

            null_sharpes[b] = _run_strategy_sharpe(
                cfg, shuffled_prices, start_i, end_i, SYMBOL
            )

        valid = ~np.isnan(null_sharpes)
        if not valid.any():
            p_val = float("nan")
            nm = ns = nq05 = nq50 = nq95 = float("nan")
        else:
            vals = null_sharpes[valid]
            n_ge = int(np.sum(vals >= real_sr))
            # +1 correction (Phipson & Smyth 2010): p never zero.
            p_val = (1 + n_ge) / (1 + int(valid.sum()))
            nm = float(vals.mean())
            ns = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            nq05, nq50, nq95 = [float(x) for x in np.percentile(vals, [5, 50, 95])]

        print(
            f"    real={real_sr:+.3f}  null_mean={nm:+.3f}  null_std={ns:.3f}  "
            f"p={p_val:.4f}  (N_valid={int(valid.sum())}/{n_perms})"
        )
        rows.append({
            "window_idx": i,
            "start": w_start,
            "end": w_end,
            "real_sharpe": real_sr,
            "null_mean_sharpe": nm,
            "null_std_sharpe": ns,
            "null_q05": nq05,
            "null_q50": nq50,
            "null_q95": nq95,
            "p_value_one_sided": p_val,
            "n_perms": int(valid.sum()),
        })

    df = pd.DataFrame(rows)

    pvals = df["p_value_one_sided"].to_numpy(dtype=float)
    x2, fisher_p = _fisher_combine(pvals)

    print()
    print("=" * 80)
    print(f"Aggregate — {SYMBOL} H-Vol q={HVOL_QUANTILE}")
    print("=" * 80)
    print("Per-window p-values:")
    for r in df.to_dict("records"):
        print(f"  {r['start']} -> {r['end']}  p={r['p_value_one_sided']:.4f}")
    print()
    n_sig = int(np.sum(pvals < 0.05))
    print(f"  windows with p < 0.05  : {n_sig}/{len(pvals)}")
    print(f"  median p-value         : {np.nanmedian(pvals):.4f}")
    print(f"  Fisher combined X^2    : {x2:.3f}  (df={2 * int(np.sum(~np.isnan(pvals)))})")
    print(f"  Fisher combined p      : {fisher_p:.6g}")
    print(f"  (elapsed: {time.time() - t0:.1f}s, N={n_perms})")

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "permutation_test.parquet"

    agg_row = {
        "window_idx": -1,
        "start": None,
        "end": None,
        "real_sharpe": float(np.nanmedian(df["real_sharpe"].to_numpy())) if not df.empty else float("nan"),
        "null_mean_sharpe": float(np.nanmean(df["null_mean_sharpe"].to_numpy())) if not df.empty else float("nan"),
        "null_std_sharpe": float(np.nanmean(df["null_std_sharpe"].to_numpy())) if not df.empty else float("nan"),
        "null_q05": float("nan"),
        "null_q50": float("nan"),
        "null_q95": float("nan"),
        "p_value_one_sided": fisher_p,
        "n_perms": int(df["n_perms"].sum()) if not df.empty else 0,
    }
    out = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
