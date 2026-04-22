"""Download S&P 600 SmallCap price data for tickers not already in dataset."""

import os
import time
import requests
from io import StringIO
import pandas as pd
import yfinance as yf

RAW_DIR = "/Users/jlg/claude/signals/data/raw"

# Step 1: Get S&P 600 SmallCap constituents from Wikipedia
print("Fetching S&P 600 SmallCap constituent list from Wikipedia...")
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', headers=headers)
resp.raise_for_status()
tables = pd.read_html(StringIO(resp.text))
sp600 = tables[0]
# Yahoo uses hyphens where Wikipedia uses dots (e.g., BRK.B -> BRK-B)
tickers = sorted(sp600['Symbol'].str.strip().str.replace('.', '-', regex=False).tolist())
print(f"Found {len(tickers)} S&P 600 SmallCap tickers")

# Step 2: Check which we already have
existing = set()
for f in os.listdir(RAW_DIR):
    if f.endswith('_1d.parquet'):
        existing.add(f.replace('_1d.parquet', ''))

already_have = [t for t in tickers if t in existing]
need = [t for t in tickers if t not in existing]
print(f"Already have: {len(already_have)}")
print(f"Need to download: {len(need)}")

# Step 3: Download
downloaded = 0
failed = []
for i, ticker in enumerate(need):
    try:
        t = yf.Ticker(ticker)
        df = t.history(start="2000-01-01", end="2026-04-21", auto_adjust=False)

        if df.empty or len(df) < 100:
            failed.append((ticker, "insufficient data"))
            continue

        # Normalize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        # Select and rename to match existing format
        out = pd.DataFrame({
            'adj_close': df['adj_close'] if 'adj_close' in df.columns else df.get('adjclose', df['close']),
            'close': df['close'],
            'high': df['high'],
            'low': df['low'],
            'open': df['open'],
            'volume': df['volume'].astype('int64'),
        })
        out.index.name = 'Date'

        # Ensure timezone-aware UTC (match existing: datetime64[ms, UTC])
        if out.index.tz is None:
            out.index = out.index.tz_localize('UTC')
        else:
            out.index = out.index.tz_convert('UTC')

        # Match existing column structure (flat columns, no MultiIndex)
        out.columns = pd.MultiIndex.from_product([['Price'], out.columns])
        out.columns = out.columns.droplevel(0)

        path = os.path.join(RAW_DIR, f"{ticker}_1d.parquet")
        out.to_parquet(path)
        downloaded += 1

    except Exception as e:
        failed.append((ticker, str(e)[:80]))

    if (i + 1) % 25 == 0:
        print(f"  Progress: {i+1}/{len(need)} processed, {downloaded} downloaded, {len(failed)} failed")

    time.sleep(0.5)  # rate limit

# Step 4: Report
print(f"\n{'='*60}")
print(f"DOWNLOAD COMPLETE")
print(f"{'='*60}")
print(f"New tickers downloaded: {downloaded}")
print(f"Failed: {len(failed)}")
if failed:
    print(f"Failed tickers:")
    for t, reason in failed:
        print(f"  {t}: {reason}")

total_files = len([f for f in os.listdir(RAW_DIR) if f.endswith('_1d.parquet')])
print(f"\nTotal universe size: {total_files} tickers")

# Step 5: Verify bias-free data loads correctly
print("\nVerifying bias-free data loads...")
try:
    from signals.backtest.bias_free import load_bias_free_data, clear_cache
    clear_cache()
    data = load_bias_free_data()
    print(f"Total tickers in bias-free data: {len(data.tickers)}")
    print("Verification OK")
except Exception as e:
    print(f"Verification error: {e}")
