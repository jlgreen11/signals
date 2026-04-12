"""Earnings data fetcher for PEAD (Post-Earnings Announcement Drift) strategy.

Fetches historical earnings data using yfinance's get_earnings_dates() API.
Falls back to a YoY EPS growth heuristic when consensus estimates are
unavailable: compare this quarter's reported EPS to the same quarter one
year ago. This is less precise than a surprise vs. consensus but still
captures the directional information that drives the drift.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from signals.utils.logging import get_logger

log = get_logger(__name__)


def fetch_earnings_yfinance(
    tickers: list[str],
    start: str | datetime | pd.Timestamp | None = None,
    end: str | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch historical earnings data from yfinance for a list of tickers.

    Returns a DataFrame with columns:
      [ticker, report_date, actual_eps, estimated_eps, surprise, surprise_pct]

    If yfinance provides consensus estimates, surprise = actual - estimated.
    Otherwise, falls back to YoY EPS growth heuristic (see
    ``_yoy_eps_surprise``).

    Tickers for which no earnings data can be retrieved are silently skipped.
    """
    import yfinance as yf

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None

    all_rows: list[dict] = []

    for ticker in tickers:
        log.info("fetching earnings for %s", ticker)
        try:
            tk = yf.Ticker(ticker)
            # get_earnings_dates returns a DataFrame indexed by Earnings Date
            # with columns like 'EPS Estimate', 'Reported EPS', 'Surprise(%)'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                edf = tk.get_earnings_dates(limit=60)
            if edf is None or edf.empty:
                log.warning("no earnings dates for %s, trying quarterly_earnings", ticker)
                edf = _try_quarterly_earnings(tk, ticker)
                if edf is None or edf.empty:
                    log.warning("no earnings data at all for %s — skipping", ticker)
                    continue

            rows = _parse_earnings_dates(edf, ticker, start_ts, end_ts)
            if rows:
                all_rows.extend(rows)
            else:
                # Fall back to quarterly_earnings for YoY heuristic
                qdf = _try_quarterly_earnings(tk, ticker)
                if qdf is not None and not qdf.empty:
                    yoy_rows = _yoy_eps_surprise(qdf, ticker, start_ts, end_ts)
                    all_rows.extend(yoy_rows)
        except Exception:
            log.exception("failed to fetch earnings for %s — skipping", ticker)
            continue

    if not all_rows:
        return pd.DataFrame(
            columns=["ticker", "report_date", "actual_eps", "estimated_eps",
                      "surprise", "surprise_pct"]
        )

    df = pd.DataFrame(all_rows)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values(["ticker", "report_date"]).reset_index(drop=True)
    return df


def _parse_earnings_dates(
    edf: pd.DataFrame,
    ticker: str,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> list[dict]:
    """Parse yfinance get_earnings_dates() output into standardized rows.

    The DataFrame index is the earnings date. Columns include:
    - 'EPS Estimate' (consensus)
    - 'Reported EPS' (actual)
    - 'Surprise(%)' (yfinance-computed)
    """
    rows: list[dict] = []
    for dt, row in edf.iterrows():
        report_date = pd.Timestamp(dt)
        if report_date.tzinfo is not None:
            report_date = report_date.tz_localize(None)
        if start_ts is not None:
            cmp_start = start_ts.tz_localize(None) if start_ts.tzinfo else start_ts
            if report_date < cmp_start:
                continue
        if end_ts is not None:
            cmp_end = end_ts.tz_localize(None) if end_ts.tzinfo else end_ts
            if report_date > cmp_end:
                continue

        actual = row.get("Reported EPS")
        estimated = row.get("EPS Estimate")

        # Skip future earnings (no reported EPS yet)
        if actual is None or (isinstance(actual, float) and np.isnan(actual)):
            continue

        actual = float(actual)

        if estimated is not None and not (isinstance(estimated, float) and np.isnan(estimated)):
            estimated = float(estimated)
            surprise = actual - estimated
            if abs(estimated) > 1e-8:
                surprise_pct = surprise / abs(estimated) * 100.0
            else:
                surprise_pct = 0.0
        else:
            estimated = np.nan
            surprise = np.nan
            surprise_pct = np.nan

        rows.append({
            "ticker": ticker,
            "report_date": report_date,
            "actual_eps": actual,
            "estimated_eps": estimated,
            "surprise": surprise,
            "surprise_pct": surprise_pct,
        })
    return rows


def _try_quarterly_earnings(tk: object, ticker: str) -> pd.DataFrame | None:
    """Try to get quarterly earnings from yfinance Ticker object."""
    try:
        qe = getattr(tk, "quarterly_earnings", None)
        if qe is not None and not qe.empty:
            return qe
    except Exception:
        log.debug("quarterly_earnings not available for %s", ticker)
    return None


def _yoy_eps_surprise(
    qdf: pd.DataFrame,
    ticker: str,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> list[dict]:
    """YoY EPS growth heuristic: compare each quarter's EPS to same quarter
    one year ago. This is a rough proxy for earnings surprise when consensus
    estimates are unavailable.

    Parameters
    ----------
    qdf : pd.DataFrame
        Quarterly earnings from yfinance. Index is typically a period
        like '2024Q1' or a date. Columns include 'Earnings' or 'Revenue'.
    """
    rows: list[dict] = []
    if "Earnings" not in qdf.columns:
        return rows

    # Sort chronologically
    qdf = qdf.sort_index()
    earnings_vals = qdf["Earnings"].values
    n = len(earnings_vals)

    for i in range(4, n):
        # Compare to same quarter last year (4 quarters ago)
        current_eps = float(earnings_vals[i])
        prior_eps = float(earnings_vals[i - 4])

        # Approximate report_date from index
        try:
            report_date = pd.Timestamp(qdf.index[i])
        except Exception:
            continue

        if start_ts is not None and report_date < start_ts:
            continue
        if end_ts is not None and report_date > end_ts:
            continue

        surprise = current_eps - prior_eps
        if abs(prior_eps) > 1e-8:
            surprise_pct = surprise / abs(prior_eps) * 100.0
        else:
            surprise_pct = 0.0

        rows.append({
            "ticker": ticker,
            "report_date": report_date,
            "actual_eps": current_eps,
            "estimated_eps": prior_eps,  # prior year same quarter as "estimate"
            "surprise": surprise,
            "surprise_pct": surprise_pct,
        })
    return rows


def compute_surprise(
    actual_eps: float,
    estimated_eps: float,
) -> tuple[float, float]:
    """Compute earnings surprise and surprise percentage.

    Returns (surprise, surprise_pct) where surprise_pct is in percentage
    points (e.g. 10.0 means a 10% surprise).
    """
    surprise = actual_eps - estimated_eps
    if abs(estimated_eps) > 1e-8:
        surprise_pct = surprise / abs(estimated_eps) * 100.0
    else:
        surprise_pct = 0.0
    return surprise, surprise_pct
