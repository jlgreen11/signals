"""Abstract base class for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataSource(ABC):
    """Common interface for historical OHLCV data providers."""

    name: str

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return a UTC-indexed OHLCV DataFrame.

        Columns must be lowercase: open, high, low, close, volume.
        Index must be a DatetimeIndex with tz=UTC.
        """

    @staticmethod
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")
        return df[REQUIRED_COLUMNS].sort_index()
