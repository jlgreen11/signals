#!/usr/bin/env python3
"""Download expanded fundamental data for all tickers with price data.

Fetches GP/A (gross profitability), ROE, P/E, and P/B from yfinance.
Saves to data/fundamentals_expanded.parquet.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signals.backtest.bias_free import load_bias_free_data


def fetch_fundamentals(ticker: str) -> dict | None:
    """Fetch fundamental data for a single ticker. Returns dict or None."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        gross_profits = info.get("grossProfits")
        total_assets = info.get("totalAssets")

        # If not in info, try financials / balance sheet
        if gross_profits is None:
            try:
                fin = t.financials
                if fin is not None and not fin.empty:
                    for label in ["Gross Profit", "GrossProfit"]:
                        if label in fin.index:
                            val = fin.loc[label].dropna()
                            if len(val) > 0:
                                gross_profits = float(val.iloc[0])
                                break
            except Exception:
                pass

        if total_assets is None:
            try:
                bs = t.balance_sheet
                if bs is not None and not bs.empty:
                    for label in ["Total Assets", "TotalAssets"]:
                        if label in bs.index:
                            val = bs.loc[label].dropna()
                            if len(val) > 0:
                                total_assets = float(val.iloc[0])
                                break
            except Exception:
                pass

        gpa = None
        if gross_profits is not None and total_assets is not None and total_assets > 0:
            gpa = gross_profits / total_assets

        roe = info.get("returnOnEquity")
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")

        # Sanitize: yfinance sometimes returns string "Infinity" or similar
        def _to_float(v):
            if v is None:
                return None
            try:
                f = float(v)
                if not pd.notna(f) or f == float("inf") or f == float("-inf"):
                    return None
                return f
            except (ValueError, TypeError):
                return None

        return {
            "ticker": ticker,
            "gpa": _to_float(gpa),
            "roe": _to_float(roe),
            "pe": _to_float(pe),
            "pb": _to_float(pb),
        }
    except Exception as e:
        return {"ticker": ticker, "gpa": None, "roe": None, "pe": None, "pb": None}


def main():
    print("Loading bias-free data to get ticker list...")
    data = load_bias_free_data()
    tickers = sorted(data.tickers)
    print(f"Found {len(tickers)} tickers")

    now = datetime.now(timezone.utc).isoformat()
    results = []
    errors = 0
    t0 = time.time()

    for i, ticker in enumerate(tickers):
        row = fetch_fundamentals(ticker)
        if row is not None:
            row["fetched_at"] = now
            results.append(row)
        else:
            errors += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_gpa = sum(1 for r in results if r.get("gpa") is not None)
            print(
                f"  [{i+1}/{len(tickers)}] "
                f"{elapsed:.0f}s elapsed | "
                f"{n_gpa} with GP/A so far | "
                f"{errors} errors"
            )

        # Rate limit: small delay to avoid throttling
        time.sleep(0.15)

    df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "data" / "fundamentals_expanded.parquet"
    df.to_parquet(out_path, index=False)

    n_gpa = df["gpa"].notna().sum()
    n_roe = df["roe"].notna().sum()
    n_pe = df["pe"].notna().sum()
    n_pb = df["pb"].notna().sum()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"  GP/A: {n_gpa} non-null ({n_gpa/len(df)*100:.0f}%)")
    print(f"  ROE:  {n_roe} non-null ({n_roe/len(df)*100:.0f}%)")
    print(f"  P/E:  {n_pe} non-null ({n_pe/len(df)*100:.0f}%)")
    print(f"  P/B:  {n_pb} non-null ({n_pb/len(df)*100:.0f}%)")


if __name__ == "__main__":
    main()
