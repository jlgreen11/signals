"""Time-series momentum (trend-following) across multiple asset classes.

Academic basis: Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum";
AQR "A Century of Evidence on Trend-Following Investing". The key
insight is that trend-following works on *macro asset classes* traded via
futures (equity indices, bonds, commodities, currencies) — not on
individual stocks, where idiosyncratic noise dominates.

The signal is simple: for each asset, go long if the trailing return
over `lookback_days` is positive, otherwise stay flat. Optionally
scale position sizes by inverse realized volatility (risk parity)
so that each asset contributes roughly equal risk to the portfolio.

This module is intentionally *not* plugged into the Markov model
backbone (composite, hmm, homc) because the signal is deterministic
and the universe is multi-asset — it has its own backtest loop
rather than flowing through BacktestEngine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TimeSeriesMomentum:
    """Multi-asset time-series momentum strategy.

    Parameters
    ----------
    lookback_days : int
        Number of trading days for the momentum signal (e.g. 21, 63, 252).
    vol_window : int
        Rolling window for realized volatility estimation (used for
        risk-parity weighting and as a minimum warm-up requirement).
    risk_parity : bool
        If True, scale each asset's weight by inverse realized volatility
        so risk contributions are balanced. If False, equal-weight all
        assets that have a positive momentum signal.
    rebalance_freq : int
        Rebalance every N trading days. Between rebalances, weights
        drift with market prices (no intra-period adjustment).
    commission_bps : float
        Round-trip transaction cost per rebalance, in basis points.
    slippage_bps : float
        Slippage per rebalance, in basis points.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        vol_window: int = 63,
        risk_parity: bool = True,
        rebalance_freq: int = 21,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0,
    ):
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if vol_window < 2:
            raise ValueError("vol_window must be >= 2")
        if rebalance_freq < 1:
            raise ValueError("rebalance_freq must be >= 1")
        self.lookback_days = lookback_days
        self.vol_window = vol_window
        self.risk_parity = risk_parity
        self.rebalance_freq = rebalance_freq
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    @property
    def warmup_days(self) -> int:
        """Minimum number of history bars needed before the first signal."""
        return max(self.lookback_days, self.vol_window) + 1

    def signals(
        self,
        prices_dict: dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Compute target weights for each asset as of a given date.

        Returns a dict {symbol: weight}. Positive weight means long;
        zero means flat. Weights sum to 1.0 across assets with a
        positive momentum signal (or 0 if all signals are flat).
        """
        raw_weights: dict[str, float] = {}
        for sym, df in prices_dict.items():
            close = df["close"].loc[df.index <= as_of_date].dropna()
            if len(close) < self.warmup_days:
                raw_weights[sym] = 0.0
                continue

            # Momentum signal: sign of trailing return
            current_price = float(close.iloc[-1])
            lookback_price = float(close.iloc[-self.lookback_days])
            if lookback_price <= 0:
                raw_weights[sym] = 0.0
                continue
            trailing_return = current_price / lookback_price - 1.0
            if trailing_return <= 0:
                raw_weights[sym] = 0.0
                continue

            # Position sizing
            if self.risk_parity:
                returns = close.pct_change().dropna()
                vol = float(returns.iloc[-self.vol_window :].std())
                if vol <= 0 or np.isnan(vol):
                    raw_weights[sym] = 0.0
                    continue
                raw_weights[sym] = 1.0 / vol
            else:
                raw_weights[sym] = 1.0

        # Normalize to sum to 1
        total = sum(raw_weights.values())
        if total <= 0:
            return {sym: 0.0 for sym in prices_dict}
        return {sym: w / total for sym, w in raw_weights.items()}

    def backtest(
        self,
        prices_dict: dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
        initial_cash: float = 10_000.0,
    ) -> pd.Series:
        """Walk-forward backtest returning a portfolio equity curve.

        The equity curve is indexed by the common trading calendar
        (inner join of all assets' trading days within [start, end]).
        """
        # Build aligned close-price matrix on the common calendar
        closes: dict[str, pd.Series] = {}
        for sym, df in prices_dict.items():
            sl = df.loc[(df.index >= start) & (df.index <= end), "close"].dropna()
            if not sl.empty:
                closes[sym] = sl

        if not closes:
            return pd.Series(dtype=float)

        close_df = pd.DataFrame(closes).dropna(how="any")
        if len(close_df) < 2:
            return pd.Series(dtype=float)

        dates = close_df.index
        returns_df = close_df.pct_change().fillna(0.0)

        # Walk forward
        equity = np.empty(len(dates))
        equity[0] = initial_cash
        current_weights: dict[str, float] = {sym: 0.0 for sym in closes}
        prev_weights: dict[str, float] = {sym: 0.0 for sym in closes}
        cost_rate = (self.commission_bps + self.slippage_bps) / 10_000.0

        for i in range(1, len(dates)):
            dt = dates[i]

            # Rebalance check
            if i == 1 or (i - 1) % self.rebalance_freq == 0:
                # Build the full history up to today for signal computation
                history = {
                    sym: df.loc[df.index <= dt]
                    for sym, df in prices_dict.items()
                    if sym in closes
                }
                current_weights = self.signals(history, dt)

                # Transaction cost: proportional to weight turnover
                turnover = sum(
                    abs(current_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
                    for s in closes
                )
                cost = cost_rate * turnover
            else:
                cost = 0.0

            # Portfolio return for this day
            day_return = sum(
                current_weights.get(sym, 0.0) * float(returns_df.loc[dt, sym])
                for sym in closes
            )
            equity[i] = equity[i - 1] * (1.0 + day_return - cost)
            prev_weights = dict(current_weights)

        return pd.Series(equity, index=dates, name="equity")
