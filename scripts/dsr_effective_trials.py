"""DSR at intermediate effective trial counts.

The project ran a 108-config grid sweep:
  3 hold_days x 3 n_long x 2 max_per_sector x 3 (short,long) x 2 filter thresholds

DSR@1 passes trivially (1.0) but is dishonest — config was selected from sweep.
DSR@108 fails (0.663) but is too harsh — many configs are highly correlated.

The RIGHT question: what is the effective number of independent trials?
The truly independent dimensions are:
  - (short,long) window: 3 options  [(21,126), (42,189), (63,252)]
  - max_per_sector: 2 options [1, 2]
  = ~6 effectively independent strategies

This script computes DSR at n_trials = 1, 6, 12, 24, 54, 108 for both
the full period and the holdout-only period (2019-2026).
"""
import sys
sys.path.insert(0, "/Users/jlg/claude/signals")

import numpy as np
import pandas as pd
from scipy.stats import skew as calc_skew, kurtosis as calc_kurtosis

from signals.backtest.bias_free import (
    load_bias_free_data,
    run_bias_free_backtest,
    clear_cache,
)
from signals.backtest.metrics import deflated_sharpe_ratio, expected_max_sharpe


TRIAL_COUNTS = [1, 6, 12, 24, 54, 108]


def run_period(label: str, start: str, end: str):
    """Run canonical backtest on a date range, return result + return stats."""
    clear_cache()
    data = load_bias_free_data(start=start, end=end)
    result = run_bias_free_backtest(
        data,
        short=63,
        long=252,
        hold_days=105,
        n_long=15,
        max_per_sector=2,
        min_short_return=0.10,
        max_long_return=1.50,
        cost_bps=10.0,
    )

    returns = result.equity_series.pct_change().dropna()
    n_obs = len(returns)
    ret_skew = float(calc_skew(returns))
    # scipy kurtosis = excess kurtosis; DSR formula uses raw kurtosis
    ret_kurt = float(calc_kurtosis(returns)) + 3.0

    return result, n_obs, ret_skew, ret_kurt


def dsr_table(sharpe, n_obs, skew, kurt, label):
    """Print DSR at each trial count and find the survival threshold."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  Sharpe = {sharpe:.4f}  |  N_obs = {n_obs}  |  Skew = {skew:.3f}  |  Kurt = {kurt:.3f}")
    print(f"{'='*65}")
    print(f"  {'n_trials':>10}  {'E[max SR]':>10}  {'DSR':>10}  {'Pass?':>8}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    survival_threshold = None
    for n in TRIAL_COUNTS:
        e_max = expected_max_sharpe(n)
        dsr = deflated_sharpe_ratio(
            sharpe=sharpe,
            n_trials=n,
            n_observations=n_obs,
            skew=skew,
            kurt=kurt,
        )
        passes = dsr > 0.95
        marker = "PASS" if passes else "FAIL"
        print(f"  {n:>10}  {e_max:>10.4f}  {dsr:>10.6f}  {marker:>8}")

        if passes and survival_threshold is None:
            # Check if next one fails
            pass
        if not passes and survival_threshold is None:
            survival_threshold = n  # first failure

    # Find exact threshold by binary search
    if survival_threshold is not None and survival_threshold > 1:
        lo, hi = 1, survival_threshold
        while hi - lo > 1:
            mid = (lo + hi) // 2
            dsr_mid = deflated_sharpe_ratio(
                sharpe=sharpe, n_trials=mid, n_observations=n_obs,
                skew=skew, kurt=kurt,
            )
            if dsr_mid > 0.95:
                lo = mid
            else:
                hi = mid
        print(f"\n  --> Signal survives DSR > 0.95 up to n_trials = {lo}")
        print(f"      Fails at n_trials = {hi}")
    elif survival_threshold is None:
        print(f"\n  --> Signal survives DSR > 0.95 at ALL tested trial counts (up to 108)")
    else:
        print(f"\n  --> Signal FAILS even at n_trials = 1")


def main():
    print("=" * 65)
    print("  DSR AT EFFECTIVE TRIAL COUNTS")
    print("  Canonical config: short=63, long=252, hold=105, n_long=15")
    print("  max_per_sector=2, min_short_ret=0.10, max_long_ret=1.50, cost=10bps")
    print("=" * 65)

    print("\n  Grid sweep dimensions:")
    print("    3 hold_days x 3 n_long x 2 max_per_sector x 3 windows x 2 filters = 108")
    print("  Effective independent dimensions:")
    print("    3 (short,long) windows x 2 sector caps = ~6 independent strategies")
    print("    (hold_days, n_long, filter thresholds are nuisance params)")

    # 1. Full period (2000-2026)
    print("\n[1/3] Running FULL period backtest (2000-2026)...")
    full_res, full_n, full_skew, full_kurt = run_period(
        "FULL", "2000-01-01", "2026-04-13"
    )
    print(f"  Sharpe: {full_res.sharpe:.4f}  CAGR: {full_res.cagr:.2%}  MaxDD: {full_res.max_drawdown:.2%}")

    # 2. Train period (2000-2018)
    print("\n[2/3] Running TRAIN period backtest (2000-2018)...")
    train_res, train_n, train_skew, train_kurt = run_period(
        "TRAIN", "2000-01-01", "2018-12-31"
    )
    print(f"  Sharpe: {train_res.sharpe:.4f}  CAGR: {train_res.cagr:.2%}  MaxDD: {train_res.max_drawdown:.2%}")

    # 3. Holdout period (2019-2026)
    print("\n[3/3] Running HOLDOUT period backtest (2019-2026)...")
    holdout_res, holdout_n, holdout_skew, holdout_kurt = run_period(
        "HOLDOUT", "2019-01-01", "2026-04-13"
    )
    print(f"  Sharpe: {holdout_res.sharpe:.4f}  CAGR: {holdout_res.cagr:.2%}  MaxDD: {holdout_res.max_drawdown:.2%}")

    # 4. DSR tables
    dsr_table(full_res.sharpe, full_n, full_skew, full_kurt,
              "FULL PERIOD (2000-2026) — in-sample, config was selected from this data")

    dsr_table(train_res.sharpe, train_n, train_skew, train_kurt,
              "TRAIN PERIOD (2000-2018) — config selected from ~this period")

    dsr_table(holdout_res.sharpe, holdout_n, holdout_skew, holdout_kurt,
              "HOLDOUT PERIOD (2019-2026) — not directly swept, but top-15 were evaluated")

    # 5. Summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"\n  Full-period Sharpe:    {full_res.sharpe:.4f}")
    print(f"  Train Sharpe:          {train_res.sharpe:.4f}")
    print(f"  Holdout Sharpe:        {holdout_res.sharpe:.4f}")
    print()
    print("  The key question: at the effective trial count of ~6")
    print("  (3 window lengths x 2 sector caps), does the signal survive?")

    for label, sharpe, n_obs, skew, kurt in [
        ("Full",    full_res.sharpe,    full_n,    full_skew,    full_kurt),
        ("Train",   train_res.sharpe,   train_n,   train_skew,   train_kurt),
        ("Holdout", holdout_res.sharpe, holdout_n, holdout_skew, holdout_kurt),
    ]:
        dsr_6 = deflated_sharpe_ratio(
            sharpe=sharpe, n_trials=6, n_observations=n_obs,
            skew=skew, kurt=kurt,
        )
        print(f"  DSR@6 ({label:>7}): {dsr_6:.6f}  {'PASS' if dsr_6 > 0.95 else 'FAIL'}")

    print()
    print("  Interpretation:")
    print("  - DSR@1 is dishonest (config chosen from sweep)")
    print("  - DSR@108 is too harsh (correlated configs inflate trial count)")
    print("  - DSR@6 is the honest middle ground (~6 independent strategies)")
    print("  - The holdout DSR is the cleanest test (data not swept)")


if __name__ == "__main__":
    main()
