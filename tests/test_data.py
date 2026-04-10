"""Tests for storage and pipeline."""

from __future__ import annotations

import pandas as pd

from signals.data.base import DataSource
from signals.data.pipeline import DataPipeline
from signals.data.storage import DataStore


class FakeSource(DataSource):
    name = "fake"

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fetch(self, symbol, start=None, end=None, interval="1d"):
        df = self.df
        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            df = df.loc[df.index >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")
            df = df.loc[df.index <= end_ts]
        return self._validate(df)


def test_store_round_trip(tmp_path, synthetic_prices):
    store = DataStore(tmp_path)
    store.save("TEST", "1d", synthetic_prices)
    loaded = store.load("TEST", "1d")
    assert len(loaded) == len(synthetic_prices)
    assert loaded.index.tz is not None
    assert (loaded["close"].values == synthetic_prices["close"].values).all()


def test_store_append_dedupes(tmp_path, synthetic_prices):
    store = DataStore(tmp_path)
    a = synthetic_prices.iloc[:300]
    b = synthetic_prices.iloc[200:]  # overlap
    store.append("TEST", "1d", a)
    added, merged = store.append("TEST", "1d", b)
    assert len(merged) == len(synthetic_prices)
    assert added == len(synthetic_prices) - 300


def test_pipeline_fetch_and_refresh(tmp_path, synthetic_prices):
    store = DataStore(tmp_path)
    src = FakeSource(synthetic_prices)
    pipe = DataPipeline(source=src, store=store)
    df1 = pipe.fetch("TEST", start="2018-01-01", end="2018-06-30", interval="1d")
    assert len(df1) > 0
    last = store.last_timestamp("TEST", "1d")
    assert last is not None
    df2 = pipe.refresh("TEST", interval="1d")
    assert len(df2) >= len(df1)


def test_list_datasets(tmp_path, synthetic_prices):
    store = DataStore(tmp_path)
    store.save("TEST", "1d", synthetic_prices)
    rows = store.list_datasets()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "TEST"
    assert rows[0]["rows"] == len(synthetic_prices)


def test_signal_record_and_fetch(tmp_path):
    store = DataStore(tmp_path)
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    store.record_signal("TEST", ts, "BUY", 0.7, 4, 0.005)
    df = store.recent_signals("TEST", days=10)
    assert len(df) == 1
    assert df.iloc[0]["signal"] == "BUY"
