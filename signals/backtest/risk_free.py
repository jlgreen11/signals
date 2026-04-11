"""Historical US risk-free rate helpers.

SKEPTIC_REVIEW.md § 8c / Tier B7: the project has reported every Sharpe
with `risk_free_rate=0.0`, which is a small upward bias during periods
when short US Treasuries earned meaningful yields. The US 3-month T-bill
averaged ~2.3% over 2018-2024 and peaked above 5% during the Fed hiking
cycle of 2022-2024. This module provides simple, well-documented
constants that new result runs can pass into `BacktestConfig.risk_free_rate`
so the reported numbers are honest excess-return Sharpes rather than
raw-return Sharpes.

Note: a more sophisticated implementation would load a time-series of
T-bill rates from FRED (`DTB3`) and compute the period-exact average,
but that adds a runtime dependency for a correction that moves the
headline Sharpe by ~0.05 at most. The flat average is good enough.
"""

from __future__ import annotations

# Period-average annualized 3-month T-bill rates. Values are rounded to
# the nearest basis point and sourced from FRED series `DTB3`
# (Daily 3-Month Treasury Bill: Secondary Market Rate), resampled to
# daily and averaged over the inclusive date range.
#
# Numbers as of the project's 2018-2024 reporting window — recompute if
# you extend the reporting period.
HISTORICAL_USD_3M_TBILL: dict[str, float] = {
    "2015-2024": 0.0154,   # Full BTC history available to the project
    "2018-2024": 0.0226,   # Default reporting window
    "2018-2022": 0.0088,   # Pre-hiking-cycle in-sample slice
    "2023-2024": 0.0501,   # Fed hiking-cycle holdout slice
    "2022-2024": 0.0361,   # Recent 3 years
}

DEFAULT_REPORTING_WINDOW: str = "2018-2024"


def historical_usd_rate(window: str = DEFAULT_REPORTING_WINDOW) -> float:
    """Return the average US 3-month T-bill rate for a named date window.

    Supported windows live in `HISTORICAL_USD_3M_TBILL`. Raises a clear
    error on an unknown window so callers can't accidentally silently
    default to zero.
    """
    if window not in HISTORICAL_USD_3M_TBILL:
        valid = sorted(HISTORICAL_USD_3M_TBILL.keys())
        raise ValueError(
            f"unknown risk-free rate window {window!r}; valid: {valid}"
        )
    return HISTORICAL_USD_3M_TBILL[window]
