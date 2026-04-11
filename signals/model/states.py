"""Quantile-based state discretization, used internally by HOMC."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute quantile edges, ensuring strictly increasing boundaries."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map values into bin indices [0, len(edges)-2]."""
    idx = np.digitize(values, edges[1:-1], right=False)
    return idx.astype(np.int64)


class StateEncoder(ABC):
    n_states: int

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> StateEncoder: ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.Series: ...

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).transform(df)

    @abstractmethod
    def label(self, state: int) -> str: ...


class QuantileStateEncoder(StateEncoder):
    """Bin a single feature (typically return_1d) into N quantile buckets."""

    def __init__(self, n_bins: int = 5, feature: str = "return_1d"):
        self.n_bins = n_bins
        self.n_states = n_bins
        self.feature = feature
        self.edges_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> QuantileStateEncoder:
        values = df[self.feature].dropna().to_numpy()
        if len(values) < self.n_bins:
            raise ValueError(
                f"Not enough samples ({len(values)}) to fit {self.n_bins} bins"
            )
        self.edges_ = _quantile_edges(values, self.n_bins)
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.edges_ is None:
            raise RuntimeError("encoder not fit")
        values = df[self.feature].to_numpy()
        states = _digitize(values, self.edges_)
        out = pd.Series(states, index=df.index, name="state", dtype="Int64")
        out[df[self.feature].isna()] = pd.NA
        return out

    def label(self, state: int) -> str:
        names = {
            0: "deep-bear",
            1: "bear",
            2: "neutral",
            3: "bull",
            4: "deep-bull",
        }
        if self.n_bins == 5:
            return names.get(int(state), f"q{state}")
        return f"q{state}/{self.n_bins}"


class CompositeStateEncoder(StateEncoder):
    """2D encoder: return bins × volatility bins → flat state index.

    The original (Phase 1) encoder. Empirically, splitting the state space
    along both axes captures regime shifts that a single-axis quantile encoder
    misses.
    """

    RETURN_LABELS_3 = ["bear", "neutral", "bull"]
    VOL_LABELS_3 = ["calm", "normal", "panic"]

    def __init__(
        self,
        return_bins: int = 3,
        volatility_bins: int = 3,
        return_feature: str = "return_1d",
        volatility_feature: str = "volatility",
    ):
        self.return_bins = return_bins
        self.volatility_bins = volatility_bins
        self.n_states = return_bins * volatility_bins
        self.return_feature = return_feature
        self.volatility_feature = volatility_feature
        self.return_edges_: np.ndarray | None = None
        self.vol_edges_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> CompositeStateEncoder:
        rets = df[self.return_feature].dropna().to_numpy()
        vols = df[self.volatility_feature].dropna().to_numpy()
        if len(rets) < self.return_bins or len(vols) < self.volatility_bins:
            raise ValueError("Insufficient samples to fit composite encoder")
        self.return_edges_ = _quantile_edges(rets, self.return_bins)
        self.vol_edges_ = _quantile_edges(vols, self.volatility_bins)
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.return_edges_ is None or self.vol_edges_ is None:
            raise RuntimeError("encoder not fit")
        ret = df[self.return_feature].to_numpy()
        vol = df[self.volatility_feature].to_numpy()
        ret_idx = _digitize(ret, self.return_edges_)
        vol_idx = _digitize(vol, self.vol_edges_)
        flat = ret_idx * self.volatility_bins + vol_idx
        out = pd.Series(flat, index=df.index, name="state", dtype="Int64")
        mask = df[self.return_feature].isna() | df[self.volatility_feature].isna()
        out[mask] = pd.NA
        return out

    def label(self, state: int) -> str:
        r = int(state) // self.volatility_bins
        v = int(state) % self.volatility_bins
        rl = (
            self.RETURN_LABELS_3[r]
            if self.return_bins == 3 and 0 <= r < 3
            else f"r{r}"
        )
        vl = (
            self.VOL_LABELS_3[v]
            if self.volatility_bins == 3 and 0 <= v < 3
            else f"v{v}"
        )
        return f"{rl}-{vl}"
