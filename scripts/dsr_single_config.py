"""Single-config Deflated Sharpe Ratio test for the canonical momentum strategy.

This is the most important number for the project: does the single canonical
config pass DSR with n_trials=1 (no multi-trial penalty)?
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


def run_and_report(label: str, start: str, end: str):
    """Run backtest on a date range and return result + return stats."""
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

    # Compute return series stats for DSR
    returns = result.equity_series.pct_change().dropna()
    n_obs = len(returns)
    ret_skew = float(calc_skew(returns))
    # scipy kurtosis returns EXCESS kurtosis; DSR formula uses raw kurtosis
    ret_kurt = float(calc_kurtosis(returns)) + 3.0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Period:        {start} to {end}")
    print(f"  Sharpe:        {result.sharpe:.4f}")
    print(f"  CAGR:          {result.cagr:.2%}")
    print(f"  Max Drawdown:  {result.max_drawdown:.2%}")
    print(f"  Final Equity:  ${result.final_equity:,.0f}")
    print(f"  Win Rate:      {result.win_rate:.1%}")
    print(f"  N Trades:      {result.n_trades}")
    print(f"  N Observations:{n_obs}")
    print(f"  Skewness:      {ret_skew:.4f}")
    print(f"  Kurtosis:      {ret_kurt:.4f}")

    return result, n_obs, ret_skew, ret_kurt


def main():
    print("DEFLATED SHARPE RATIO: SINGLE-CONFIG TEST")
    print("Canonical config: short=63, long=252, hold=105, n_long=15")
    print("max_per_sector=2, min_short_ret=0.10, max_long_ret=1.50, cost=10bps")

    # 1. Full period
    full_result, full_n, full_skew, full_kurt = run_and_report(
        "FULL PERIOD", "2000-01-01", "2026-04-13"
    )

    # 2. Train period
    train_result, train_n, train_skew, train_kurt = run_and_report(
        "TRAIN PERIOD (2000-2018)", "2000-01-01", "2018-12-31"
    )

    # 3. Holdout period
    holdout_result, holdout_n, holdout_skew, holdout_kurt = run_and_report(
        "HOLDOUT PERIOD (2019-2026)", "2019-01-01", "2026-04-13"
    )

    # 4. DSR calculations
    print("\n" + "=" * 60)
    print("  DEFLATED SHARPE RATIO RESULTS")
    print("=" * 60)

    # DSR@1 (single config, no multi-trial penalty)
    dsr_1 = deflated_sharpe_ratio(
        sharpe=full_result.sharpe,
        n_trials=1,
        n_observations=full_n,
        skew=full_skew,
        kurt=full_kurt,
    )
    e_max_1 = expected_max_sharpe(1)

    # DSR@108 (full sweep penalty)
    dsr_108 = deflated_sharpe_ratio(
        sharpe=full_result.sharpe,
        n_trials=108,
        n_observations=full_n,
        skew=full_skew,
        kurt=full_kurt,
    )
    e_max_108 = expected_max_sharpe(108)

    # DSR@1 on holdout only
    dsr_holdout = deflated_sharpe_ratio(
        sharpe=holdout_result.sharpe,
        n_trials=1,
        n_observations=holdout_n,
        skew=holdout_skew,
        kurt=holdout_kurt,
    )

    # DSR@1 on train only
    dsr_train = deflated_sharpe_ratio(
        sharpe=train_result.sharpe,
        n_trials=1,
        n_observations=train_n,
        skew=train_skew,
        kurt=train_kurt,
    )

    print(f"\n  Full period Sharpe:      {full_result.sharpe:.4f}")
    print(f"  Train Sharpe:            {train_result.sharpe:.4f}")
    print(f"  Holdout Sharpe:          {holdout_result.sharpe:.4f}")
    print()
    print(f"  E[max SR] @ n_trials=1:  {e_max_1:.4f}")
    print(f"  E[max SR] @ n_trials=108:{e_max_108:.4f}")
    print()
    print(f"  DSR @ n_trials=1  (full):    {dsr_1:.6f}")
    print(f"  DSR @ n_trials=108 (full):   {dsr_108:.6f}")
    print(f"  DSR @ n_trials=1  (train):   {dsr_train:.6f}")
    print(f"  DSR @ n_trials=1  (holdout): {dsr_holdout:.6f}")
    print()

    # Verdict
    print("  " + "-" * 40)
    if dsr_1 > 0.95:
        print(f"  PASS: DSR@1 = {dsr_1:.4f} > 0.95")
        print("  The canonical config's Sharpe is statistically")
        print("  significant as a SINGLE hypothesis test.")
    else:
        print(f"  FAIL: DSR@1 = {dsr_1:.4f} <= 0.95")
        print("  The canonical config's Sharpe is NOT statistically")
        print("  significant even as a single hypothesis test.")

    if dsr_108 > 0.95:
        print(f"\n  DSR@108 also PASSES ({dsr_108:.4f} > 0.95)")
    else:
        print(f"\n  DSR@108 FAILS ({dsr_108:.4f} <= 0.95) as expected")
        print("  (multi-trial penalty makes it harder to pass)")
    print("  " + "-" * 40)


if __name__ == "__main__":
    main()
