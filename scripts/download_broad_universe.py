"""Download broad US stock universe (~3000+ tickers) via NASDAQ screener API.

Source: https://api.nasdaq.com/api/screener/stocks
Provides ALL Nasdaq-traded and NYSE-listed US securities.

We filter for US common stocks with standard tickers (1-5 uppercase letters),
then download daily price history from yfinance going back to 2000-01-01.
"""

import os
import time
import re
import requests
import pandas as pd
import yfinance as yf

RAW_DIR = "/Users/jlg/claude/signals/data/raw"
START_DATE = "2000-01-01"
END_DATE = "2026-04-21"
MIN_TRADING_DAYS = 252  # skip tickers with < 1 year of data
BATCH_DELAY = 0.5       # seconds between individual downloads
PROGRESS_EVERY = 100    # print progress every N tickers

# ── Step 1: Get US stock list from NASDAQ screener API ──────────────────────
print("Fetching stock list from NASDAQ screener API...")
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Accept': 'application/json'
}
resp = requests.get(
    'https://api.nasdaq.com/api/screener/stocks?tableType=traded&download=true',
    headers=headers, timeout=30
)
resp.raise_for_status()
rows = resp.json()['data']['rows']
print(f"Raw securities in list: {len(rows)}")

# Filter: US-domiciled, standard ticker format (1-5 uppercase letters)
us_standard = [r for r in rows
               if r.get('country') == 'United States'
               and re.match(r'^[A-Z]{1,5}$', r['symbol'])]
print(f"US stocks with standard tickers: {len(us_standard)}")

tickers = sorted(set(r['symbol'] for r in us_standard))
print(f"Unique tickers: {len(tickers)}")

# ── Step 2: Check existing ──────────────────────────────────────────────────
existing = set()
for f in os.listdir(RAW_DIR):
    if f.endswith('_1d.parquet'):
        existing.add(f.replace('_1d.parquet', ''))

already_have = [t for t in tickers if t in existing]
need = [t for t in tickers if t not in existing]
print(f"Already have: {len(already_have)}")
print(f"Need to download: {len(need)}")

# ── Step 3: Download with rate limiting ─────────────────────────────────────
downloaded = 0
skipped_short = 0
failed = []

print(f"\nStarting download of {len(need)} tickers...")
print(f"Estimated time: {len(need) * BATCH_DELAY / 60:.0f}+ minutes\n")

for i, ticker in enumerate(need):
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=START_DATE, end=END_DATE, auto_adjust=False)

        if df.empty or len(df) < MIN_TRADING_DAYS:
            skipped_short += 1
            continue

        # Normalize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        # Find adj_close column
        if 'adj_close' in df.columns:
            adj_col = 'adj_close'
        else:
            adj_candidates = [c for c in df.columns if 'adj' in c.lower()]
            adj_col = adj_candidates[0] if adj_candidates else 'close'

        out = pd.DataFrame({
            'adj_close': df[adj_col],
            'close': df['close'],
            'high': df['high'],
            'low': df['low'],
            'open': df['open'],
            'volume': df['volume'].astype('int64'),
        })
        out.index.name = 'Date'

        # Ensure timezone-aware UTC
        if out.index.tz is None:
            out.index = out.index.tz_localize('UTC')
        else:
            out.index = out.index.tz_convert('UTC')

        # Match existing parquet column format
        out.columns = pd.MultiIndex.from_product([['Price'], out.columns])
        out.columns = out.columns.droplevel(0)

        path = os.path.join(RAW_DIR, f"{ticker}_1d.parquet")
        out.to_parquet(path)
        downloaded += 1

    except Exception as e:
        failed.append((ticker, str(e)[:100]))

    # Progress reporting
    if (i + 1) % PROGRESS_EVERY == 0:
        total_files = len(existing) + downloaded
        print(f"  [{i+1}/{len(need)}] downloaded={downloaded}, "
              f"skipped_short={skipped_short}, failed={len(failed)}, "
              f"total_universe={total_files}")

    time.sleep(BATCH_DELAY)

# ── Step 4: Report ──────────────────────────────────────────────────────────
total_files = len([f for f in os.listdir(RAW_DIR) if f.endswith('_1d.parquet')])

print(f"\n{'='*60}")
print(f"DOWNLOAD COMPLETE")
print(f"{'='*60}")
print(f"New tickers downloaded:  {downloaded}")
print(f"Skipped (< {MIN_TRADING_DAYS} days):   {skipped_short}")
print(f"Failed:                  {len(failed)}")
print(f"Total universe size:     {total_files} tickers")

if failed and len(failed) <= 50:
    print(f"\nFailed tickers:")
    for t, reason in failed:
        print(f"  {t}: {reason}")
elif failed:
    print(f"\nFirst 50 failed tickers:")
    for t, reason in failed[:50]:
        print(f"  {t}: {reason}")
