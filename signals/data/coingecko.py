"""CoinGecko data source — used as a Bitcoin fallback for yfinance.

Implements only what we need today: daily OHLCV-ish via market_chart/range.
Note: CoinGecko's free endpoint returns prices and volumes but not OHLC for arbitrary
ranges, so we synthesize OHLC from the price series (open=prev close, high/low ≈ close).
This is sufficient for return-based features but not high-fidelity OHLC.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from signals.data.base import DataSource
from signals.utils.logging import get_logger

log = get_logger(__name__)

_SYMBOL_TO_ID = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
}


class CoinGeckoSource(DataSource):
    name = "coingecko"
    BASE = "https://api.coingecko.com/api/v3"

    def fetch(
        self,
        symbol: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        coin_id = _SYMBOL_TO_ID.get(symbol)
        if coin_id is None:
            raise ValueError(f"CoinGecko mapping unknown for symbol {symbol!r}")
        if interval != "1d":
            raise NotImplementedError("CoinGeckoSource only supports interval='1d' today.")

        start_ts = _to_unix(start) if start else 0
        end_ts = _to_unix(end) if end else int(datetime.now(tz=timezone.utc).timestamp())

        url = f"{self.BASE}/coins/{coin_id}/market_chart/range"
        params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}
        log.info("coingecko.fetch %s -> %s", symbol, url)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        prices = payload.get("prices", [])
        volumes = payload.get("total_volumes", [])
        if not prices:
            return self._validate(pd.DataFrame())

        price_df = pd.DataFrame(prices, columns=["ts_ms", "close"])
        vol_df = pd.DataFrame(volumes, columns=["ts_ms", "volume"])
        df = price_df.merge(vol_df, on="ts_ms", how="left")
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.floor("D")
        df = df.drop_duplicates("ts", keep="last").set_index("ts").sort_index()

        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1)
        df["low"] = df[["open", "close"]].min(axis=1)
        df["volume"] = df["volume"].fillna(0.0)

        return self._validate(df[["open", "high", "low", "close", "volume"]])


def _to_unix(value: datetime | str) -> int:
    if isinstance(value, str):
        value = pd.to_datetime(value, utc=True).to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())
