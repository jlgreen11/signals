"""Cross-sectional momentum model.

Implements the classic "12-1 month" momentum strategy from Jegadeesh &
Titman (1993): rank a universe of stocks by trailing return over a lookback
window (default 252 trading days), skip the most recent period (default 21
days) to avoid the short-term reversal effect, then go equal-weight long the
top-N winners.

This is a fundamentally different signal class from the project's existing
single-asset Markov / trend / vol models. Instead of predicting ONE stock's
future from its own history, it predicts WHICH stocks will outperform
relative to others.

Usage::

    from signals.model.momentum import CrossSectionalMomentum

    mom = CrossSectionalMomentum(lookback_days=252, skip_days=21, n_long=5)
    weights = mom.rank(prices_dict, as_of_date=pd.Timestamp("2024-01-02"))
    equity = mom.backtest(prices_dict, start="2019-04-01", end="2026-04-01")
"""

from __future__ import annotations

import pandas as pd


class CrossSectionalMomentum:
    """Cross-sectional momentum: rank stocks by trailing return, go long winners.

    Parameters
    ----------
    lookback_days : int
        Number of trading days for the trailing return calculation (default 252,
        roughly 12 months).
    skip_days : int
        Number of most-recent trading days to skip before the lookback window
        (default 21, roughly 1 month). This avoids the well-documented
        short-term reversal effect.
    n_long : int
        Number of top-ranked stocks to hold (equal-weight). Stocks outside the
        top-N get zero weight.
    rebalance_freq : int
        Number of trading days between portfolio rebalances (default 21,
        roughly monthly).
    commission_bps : float
        One-way commission in basis points, applied to each rebalance trade.
    slippage_bps : float
        One-way slippage in basis points, applied to each rebalance trade.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        n_long: int = 5,
        rebalance_freq: int = 21,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.n_long = n_long
        self.rebalance_freq = rebalance_freq
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def _required_history(self) -> int:
        """Minimum number of trading days a stock needs to be rankable."""
        return self.lookback_days + self.skip_days

    def rank(
        self,
        prices_dict: dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Return {symbol: weight} for the top-N stocks as of a given date.

        Stocks with insufficient history (< lookback_days + skip_days bars
        before as_of_date) are excluded from ranking. Weights sum to 1.0
        across the top-N; all other stocks get 0.0.
        """
        trailing_returns: dict[str, float] = {}

        for symbol, df in prices_dict.items():
            # Get prices up to as_of_date
            eligible = df.loc[df.index <= as_of_date, "close"]
            if len(eligible) < self._required_history():
                continue

            # "12-1 month" convention: return from t-lookback-skip to t-skip
            end_idx = len(eligible) - self.skip_days
            start_idx = end_idx - self.lookback_days

            if start_idx < 0 or end_idx < 1:
                continue

            p_start = eligible.iloc[start_idx]
            p_end = eligible.iloc[end_idx]

            if p_start <= 0:
                continue

            trailing_returns[symbol] = (p_end - p_start) / p_start

        if not trailing_returns:
            return {sym: 0.0 for sym in prices_dict}

        # Sort by trailing return descending, pick top N
        sorted_symbols = sorted(
            trailing_returns, key=trailing_returns.get, reverse=True  # type: ignore[arg-type]
        )

        n_long = min(self.n_long, len(sorted_symbols))
        weight = 1.0 / n_long if n_long > 0 else 0.0

        result: dict[str, float] = {}
        winners = set(sorted_symbols[:n_long])
        for sym in prices_dict:
            result[sym] = weight if sym in winners else 0.0

        return result

    def backtest(
        self,
        prices_dict: dict[str, pd.DataFrame],
        start: str,
        end: str,
        initial_cash: float = 10000.0,
    ) -> pd.Series:
        """Walk-forward backtest with periodic rebalancing to top-N winners.

        Returns an equity curve (pd.Series with DatetimeIndex).

        Transaction costs: commission_bps + slippage_bps applied to the
        absolute dollar value of each trade (buy or sell).
        """
        if isinstance(start, pd.Timestamp) and start.tzinfo is not None:
            start_ts = start
        else:
            start_ts = pd.Timestamp(start, tz="UTC")
        if isinstance(end, pd.Timestamp) and end.tzinfo is not None:
            end_ts = end
        else:
            end_ts = pd.Timestamp(end, tz="UTC")

        # Build a common date index from the union of all stocks
        all_dates: set[pd.Timestamp] = set()
        for df in prices_dict.values():
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            all_dates.update(df.index[mask])
        trading_dates = sorted(all_dates)

        if not trading_dates:
            return pd.Series(dtype=float)

        # Track holdings: {symbol: number_of_shares}
        holdings: dict[str, float] = {sym: 0.0 for sym in prices_dict}
        cash = initial_cash

        equity_points: list[tuple[pd.Timestamp, float]] = []
        bars_since_rebalance = self.rebalance_freq  # Force rebalance on first bar
        cost_rate = (self.commission_bps + self.slippage_bps) * 1e-4

        for date in trading_dates:
            # Get current prices for all symbols
            prices: dict[str, float] = {}
            for sym, df in prices_dict.items():
                if date in df.index:
                    prices[sym] = float(df.loc[date, "close"])

            # Check if it's time to rebalance
            bars_since_rebalance += 1
            if bars_since_rebalance >= self.rebalance_freq:
                new_weights = self.rank(prices_dict, as_of_date=date)

                # Compute current portfolio equity
                equity = cash
                for sym in holdings:
                    if sym in prices:
                        equity += holdings[sym] * prices[sym]

                if equity > 0:
                    # Sell everything, then buy new targets
                    # First: compute trades needed
                    for sym in prices_dict:
                        if sym not in prices:
                            continue
                        price = prices[sym]
                        current_value = holdings[sym] * price
                        target_value = new_weights.get(sym, 0.0) * equity
                        trade_value = abs(target_value - current_value)

                        if trade_value > 1e-6:
                            cost = trade_value * cost_rate
                            cash -= cost

                    # Recompute equity after costs
                    equity_after_costs = cash
                    for sym in holdings:
                        if sym in prices:
                            equity_after_costs += holdings[sym] * prices[sym]

                    # Now set holdings to target
                    for sym in prices_dict:
                        if sym not in prices:
                            holdings[sym] = 0.0
                            continue
                        price = prices[sym]
                        # Sell current
                        cash += holdings[sym] * price
                        holdings[sym] = 0.0

                    # Buy new targets from remaining cash (after costs)
                    for sym in prices_dict:
                        if sym not in prices:
                            continue
                        w = new_weights.get(sym, 0.0)
                        if w > 0:
                            target_value = w * equity_after_costs
                            holdings[sym] = target_value / prices[sym]
                            cash -= target_value

                bars_since_rebalance = 0

            # Mark equity
            equity = cash
            for sym in holdings:
                if sym in prices:
                    equity += holdings[sym] * prices[sym]
            equity_points.append((date, equity))

        if not equity_points:
            return pd.Series(dtype=float)

        ts, eq = zip(*equity_points, strict=True)
        return pd.Series(eq, index=pd.DatetimeIndex(ts), name="equity")
