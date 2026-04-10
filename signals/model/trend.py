"""Trend-following models for equity indices (and anything with a strong drift).

The Markov-chain backbone of the signals project (`composite`, `hmm`, `homc`,
`hybrid`) was designed for BTC's wild regime structure and was empirically
shown to underperform buy & hold on ^GSPC (see
`scripts/HOMC_TIER0E_BTC_SP500.md`). Equity indices with secular uptrends and
tight vol distributions need a different model class.

This module provides two classic trend-following filters:

  - `TrendFilter`        — long when close > MA(window), else flat
  - `DualMovingAverage`  — long when MA(fast) > MA(slow), else flat
                           (the canonical "golden cross / death cross")

Both conform to the same model interface as the Markov models so they plug
into `BacktestEngine` and `SignalGenerator` transparently:

  - `fit(observations, **kwargs)` — no-op (rule is deterministic), records
    the window lengths and marks fitted_ = True
  - `predict_state(observations)` — returns 0 (below MA / bearish) or
    1 (above MA / bullish)
  - `predict_next(state)` — deterministic one-hot
  - `state_returns_` — synthetic [-1.0, +1.0] so SignalGenerator's
    expected-return computation yields a saturated signal: above-MA → full
    long, below-MA → flat (or short if allow_short, which is off by default
    for equity indices)
  - `label(state)` — "below_ma" / "above_ma" (or "death_cross" / "golden_cross"
    for the dual variant)

The synthetic `state_returns_` values are deliberately large so that
`SignalGenerator`'s buy/sell thresholds (25 / -35 bps by default) always
trigger — a trend filter is a binary signal and should not be gated by
expected-return magnitude.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Sentinel values chosen so SignalGenerator's buy/sell thresholds always
# trigger. The SignalGenerator formula `expected = probs @ state_returns_`
# yields ±1.0 for these models — well above 25 bps (BUY) and below
# -35 bps (SELL → flat for long-only).
_STATE_RETURNS = np.array([-1.0, +1.0])


class TrendFilter:
    """Long when close > MA(window), flat otherwise.

    The canonical single-MA equity trend filter. Default window is 200
    days — the standard in trend-following literature (Faber 2007,
    "A Quantitative Approach to Tactical Asset Allocation").
    """

    def __init__(self, window: int = 200):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = int(window)
        self.n_states = 2
        self.state_returns_ = _STATE_RETURNS.copy()
        self.state_counts_: np.ndarray = np.zeros(2, dtype=np.int64)
        self.fitted_: bool = False

    def fit(
        self,
        observations: pd.DataFrame,
        **_ignored,
    ) -> TrendFilter:
        """No learning — the MA rule is deterministic. Records training-
        window state frequencies for introspection, then marks fitted."""
        close = observations["close"].dropna()
        if len(close) < self.window + 1:
            raise ValueError(
                f"Need >= {self.window + 1} bars to fit TrendFilter(window={self.window}); "
                f"got {len(close)}"
            )
        ma = close.rolling(window=self.window, min_periods=self.window).mean()
        above = (close > ma).dropna().astype(int)
        self.state_counts_ = np.array(
            [int((above == 0).sum()), int((above == 1).sum())], dtype=np.int64
        )
        self.fitted_ = True
        return self

    def predict_state(self, observations: pd.DataFrame) -> int:
        if not self.fitted_:
            raise RuntimeError("TrendFilter not fit")
        close = observations["close"].dropna()
        if len(close) < self.window:
            raise ValueError(
                f"Need >= {self.window} bars to decode state; got {len(close)}"
            )
        ma_value = float(close.iloc[-self.window :].mean())
        return 1 if float(close.iloc[-1]) > ma_value else 0

    def predict_next(self, state: int) -> np.ndarray:
        s = int(state)
        if s not in (0, 1):
            raise ValueError(f"state must be 0 or 1; got {s}")
        out = np.zeros(2)
        out[s] = 1.0
        return out

    def expected_next_return(self, state: int) -> float:
        return float(self.state_returns_[int(state)])

    def label(self, state) -> str:
        s = int(state)
        if s == 0:
            return "below_ma"
        if s == 1:
            return "above_ma"
        return f"state_{s}"

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "TrendFilter",
            "window": self.window,
            "state_counts": self.state_counts_.tolist(),
            "fitted": self.fitted_,
        }
        p.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path | str) -> TrendFilter:
        d = json.loads(Path(path).read_text())
        obj = cls(window=d["window"])
        obj.state_counts_ = np.array(d["state_counts"], dtype=np.int64)
        obj.fitted_ = bool(d["fitted"])
        return obj


class DualMovingAverage:
    """Golden cross / death cross: long when MA(fast) > MA(slow).

    Classic 50/200 dual-MA crossover. Smoother than a single-MA rule
    because both MAs have their own lag, but it misses the fastest moves.
    """

    def __init__(self, fast_window: int = 50, slow_window: int = 200):
        if fast_window < 2 or slow_window < 2:
            raise ValueError("windows must be >= 2")
        if fast_window >= slow_window:
            raise ValueError("fast_window must be < slow_window")
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self.n_states = 2
        self.state_returns_ = _STATE_RETURNS.copy()
        self.state_counts_: np.ndarray = np.zeros(2, dtype=np.int64)
        self.fitted_: bool = False

    def fit(
        self,
        observations: pd.DataFrame,
        **_ignored,
    ) -> DualMovingAverage:
        close = observations["close"].dropna()
        if len(close) < self.slow_window + 1:
            raise ValueError(
                f"Need >= {self.slow_window + 1} bars to fit "
                f"DualMovingAverage(slow={self.slow_window}); got {len(close)}"
            )
        fast = close.rolling(window=self.fast_window, min_periods=self.fast_window).mean()
        slow = close.rolling(window=self.slow_window, min_periods=self.slow_window).mean()
        cross = (fast > slow).dropna().astype(int)
        self.state_counts_ = np.array(
            [int((cross == 0).sum()), int((cross == 1).sum())], dtype=np.int64
        )
        self.fitted_ = True
        return self

    def predict_state(self, observations: pd.DataFrame) -> int:
        if not self.fitted_:
            raise RuntimeError("DualMovingAverage not fit")
        close = observations["close"].dropna()
        if len(close) < self.slow_window:
            raise ValueError(
                f"Need >= {self.slow_window} bars to decode state; got {len(close)}"
            )
        fast_ma = float(close.iloc[-self.fast_window :].mean())
        slow_ma = float(close.iloc[-self.slow_window :].mean())
        return 1 if fast_ma > slow_ma else 0

    def predict_next(self, state: int) -> np.ndarray:
        s = int(state)
        if s not in (0, 1):
            raise ValueError(f"state must be 0 or 1; got {s}")
        out = np.zeros(2)
        out[s] = 1.0
        return out

    def expected_next_return(self, state: int) -> float:
        return float(self.state_returns_[int(state)])

    def label(self, state) -> str:
        s = int(state)
        if s == 0:
            return "death_cross"
        if s == 1:
            return "golden_cross"
        return f"state_{s}"

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "DualMovingAverage",
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "state_counts": self.state_counts_.tolist(),
            "fitted": self.fitted_,
        }
        p.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path | str) -> DualMovingAverage:
        d = json.loads(Path(path).read_text())
        obj = cls(fast_window=d["fast_window"], slow_window=d["slow_window"])
        obj.state_counts_ = np.array(d["state_counts"], dtype=np.int64)
        obj.fitted_ = bool(d["fitted"])
        return obj
