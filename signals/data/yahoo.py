"""Yahoo Finance data source via yfinance."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from signals.data.base import DataSource
from signals.utils.logging import get_logger

log = get_logger(__name__)


class YahooFinanceSource(DataSource):
    """Wraps yfinance.download() and normalizes output."""

    name = "yahoo"

    def fetch(
        self,
        symbol: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        import yfinance as yf

        log.info("yahoo.fetch symbol=%s start=%s end=%s interval=%s", symbol, start, end, interval)
        df = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
            actions=False,
            threads=False,
        )
        if df is None or df.empty:
            log.warning("yfinance returned no rows for %s", symbol)
            return self._validate(pd.DataFrame())

        # yfinance can return a MultiIndex on columns when downloading multiple tickers
        # — collapse to a single level if needed.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        return self._validate(df)
