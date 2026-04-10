"""Incremental data refresh orchestration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from signals.data.base import DataSource
from signals.data.storage import DataStore
from signals.utils.logging import get_logger

log = get_logger(__name__)


class DataPipeline:
    """Pulls data from a DataSource and persists to a DataStore."""

    def __init__(self, source: DataSource, store: DataStore):
        self.source = source
        self.store = store

    def fetch(
        self,
        symbol: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fresh fetch over the requested range; merges into the store."""
        df = self.source.fetch(symbol, start=start, end=end, interval=interval)
        if df.empty:
            log.warning("fetch returned empty for %s", symbol)
            return df
        added, merged = self.store.append(symbol, interval, df)
        self.store.log_fetch(
            symbol=symbol,
            interval=interval,
            source=self.source.name,
            rows_added=added,
            first_ts=df.index.min(),
            last_ts=df.index.max(),
        )
        log.info(
            "fetch %s %s rows_added=%d total_rows=%d",
            symbol,
            interval,
            added,
            len(merged),
        )
        return merged

    def refresh(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        """Append-only incremental refresh from last stored bar."""
        last = self.store.last_timestamp(symbol, interval)
        if last is None:
            log.info("refresh %s: no existing data, fetching last 5 years", symbol)
            start = datetime.now(tz=timezone.utc) - timedelta(days=365 * 5)
            return self.fetch(symbol, start=start, interval=interval)

        # Start fetching from the bar after the last stored one.
        start = (last + timedelta(days=1)).to_pydatetime()
        if start >= datetime.now(tz=timezone.utc):
            log.info("refresh %s: already up-to-date (last=%s)", symbol, last)
            return self.store.load(symbol, interval)
        return self.fetch(symbol, start=start, interval=interval)
