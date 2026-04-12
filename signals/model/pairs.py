"""Statistical arbitrage / pairs trading strategy.

Unlike the directional models (Markov chains, trend filters, hybrid vol-routers),
pairs trading is MARKET NEUTRAL. It profits from the relative movement between
two cointegrated stocks reverting to the mean. You don't need to predict direction
-- just that the spread will close.

Algorithm:
  1. Pair discovery: test all (N choose 2) pairs for cointegration via the
     Engle-Granger test (statsmodels `coint()`). Keep pairs with p-value < threshold.
  2. For each cointegrated pair (A, B):
     - Compute hedge ratio via OLS: A = beta * B + alpha + epsilon
     - Spread = A - beta * B
     - Z-score = (spread - rolling_mean) / rolling_std
  3. Trading signals:
     - ENTRY: |Z| > entry_threshold (default 2.0)
       * Z > 2 => short A, long B (spread too wide, expect contraction)
       * Z < -2 => long A, short B
     - EXIT: |Z| < exit_threshold (default 0.5) or Z crosses zero
  4. Walk-forward: re-discover pairs every `lookback` days to avoid look-ahead bias.

Transaction costs: 5 bps commission + 5 bps slippage on each leg, applied at
entry and exit (4 total legs per round-trip).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint


@dataclass
class PairPosition:
    """Tracks an active pair trade."""

    stock_a: str
    stock_b: str
    hedge_ratio: float
    direction: int  # +1 = long spread (long A, short B), -1 = short spread
    entry_date: pd.Timestamp
    entry_spread: float
    capital_per_leg: float
    shares_a: float = 0.0
    shares_b: float = 0.0
    closed: bool = False
    exit_date: pd.Timestamp | None = None
    pnl: float = 0.0


@dataclass
class PairInfo:
    """Discovered cointegrated pair."""

    stock_a: str
    stock_b: str
    pvalue: float
    hedge_ratio: float
    intercept: float


@dataclass
class PairsBacktestResult:
    """Container for pairs trading backtest results."""

    equity_curve: pd.Series
    trades: list[PairPosition] = field(default_factory=list)
    pair_discovery_log: list[dict] = field(default_factory=list)


class PairsTrading:
    """Statistical arbitrage pairs trading strategy.

    Parameters
    ----------
    coint_pvalue : float
        Maximum p-value for the Engle-Granger cointegration test to accept a pair.
    entry_zscore : float
        Z-score threshold to enter a trade (absolute value).
    exit_zscore : float
        Z-score threshold to exit a trade (absolute value).
    lookback : int
        Rolling window length for pair discovery and Z-score computation.
    max_pairs : int
        Maximum number of active pairs to trade simultaneously.
    zscore_window : int
        Rolling window for Z-score mean/std computation. Defaults to 60 days.
    commission_bps : float
        Commission per leg in basis points.
    slippage_bps : float
        Slippage per leg in basis points.
    """

    def __init__(
        self,
        coint_pvalue: float = 0.05,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        lookback: int = 252,
        max_pairs: int = 5,
        zscore_window: int = 60,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0,
    ):
        if entry_zscore <= exit_zscore:
            raise ValueError("entry_zscore must be > exit_zscore")
        if lookback < 30:
            raise ValueError("lookback must be >= 30")
        self.coint_pvalue = coint_pvalue
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.lookback = lookback
        self.max_pairs = max_pairs
        self.zscore_window = zscore_window
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self._tc_rate = (commission_bps + slippage_bps) / 10_000.0

    def find_pairs(
        self,
        prices_dict: dict[str, pd.Series],
        as_of_date: pd.Timestamp | None = None,
    ) -> list[PairInfo]:
        """Find cointegrated pairs from a dict of {ticker: close_price_series}.

        If `as_of_date` is given, only data up to that date is used.
        Returns a list of PairInfo sorted by p-value (best first), capped at max_pairs.
        """
        tickers = sorted(prices_dict.keys())
        if len(tickers) < 2:
            return []

        # Align all price series to a common date index
        df = pd.DataFrame(prices_dict)
        if as_of_date is not None:
            df = df.loc[df.index <= as_of_date]
        # Use only the trailing `lookback` bars
        df = df.iloc[-self.lookback :]
        df = df.dropna(axis=1, how="any")  # drop tickers with NaNs in the window

        available = [t for t in tickers if t in df.columns]
        if len(available) < 2:
            return []

        found: list[PairInfo] = []
        for a, b in combinations(available, 2):
            series_a = df[a].values
            series_b = df[b].values

            # Engle-Granger cointegration test
            try:
                _, pvalue, _ = coint(series_a, series_b)
            except Exception:
                continue

            if pvalue > self.coint_pvalue:
                continue

            # OLS hedge ratio: A = beta * B + alpha
            hedge_ratio, intercept = self._ols_hedge_ratio(series_a, series_b)
            found.append(PairInfo(
                stock_a=a,
                stock_b=b,
                pvalue=float(pvalue),
                hedge_ratio=float(hedge_ratio),
                intercept=float(intercept),
            ))

        # Sort by p-value (strongest cointegration first), cap at max_pairs
        found.sort(key=lambda p: p.pvalue)
        return found[: self.max_pairs]

    @staticmethod
    def _ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
        """OLS regression y ~ beta * x + alpha. Returns (beta, alpha)."""
        x_const = add_constant(x)
        model = OLS(y, x_const).fit()
        alpha = float(model.params[0])
        beta = float(model.params[1])
        return beta, alpha

    def compute_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        hedge_ratio: float,
    ) -> pd.Series:
        """Spread = price_A - hedge_ratio * price_B."""
        return price_a - hedge_ratio * price_b

    def compute_zscore(self, spread: pd.Series) -> pd.Series:
        """Rolling Z-score of the spread."""
        roll_mean = spread.rolling(window=self.zscore_window, min_periods=self.zscore_window).mean()
        roll_std = spread.rolling(window=self.zscore_window, min_periods=self.zscore_window).std()
        # Avoid division by zero
        roll_std = roll_std.replace(0.0, np.nan)
        return (spread - roll_mean) / roll_std

    def backtest(
        self,
        prices_dict: dict[str, pd.Series],
        start: pd.Timestamp,
        end: pd.Timestamp,
        initial_cash: float = 10_000.0,
    ) -> PairsBacktestResult:
        """Walk-forward pairs trading backtest.

        Re-discovers pairs every `lookback` days. Trades Z-score entry/exit signals.
        Returns a PairsBacktestResult with equity curve and trade log.
        """
        # Build aligned price DataFrame
        df = pd.DataFrame(prices_dict).sort_index()
        df = df.loc[(df.index >= start) & (df.index <= end)]
        df = df.dropna(axis=1, how="any")

        if df.empty or len(df) < self.zscore_window + 10:
            return PairsBacktestResult(
                equity_curve=pd.Series(dtype=float),
                trades=[],
            )

        dates = df.index
        cash = initial_cash
        active_positions: list[PairPosition] = []
        all_trades: list[PairPosition] = []
        discovery_log: list[dict] = []

        # Track current pairs and when they were discovered
        current_pairs: list[PairInfo] = []
        last_discovery_idx = -self.lookback  # force immediate discovery

        equity_values = []
        equity_dates = []

        for i, date in enumerate(dates):
            # Re-discover pairs every `lookback` days
            if i - last_discovery_idx >= self.lookback or i == 0:
                # Close all active positions before re-discovery
                for pos in active_positions:
                    if not pos.closed:
                        cash += self._close_position(pos, df, date)
                        all_trades.append(pos)
                active_positions = []

                # Discover new pairs using data up to this date
                # We need lookback bars of HISTORICAL data, so use the full
                # prices_dict (not the trimmed df) to avoid look-ahead
                hist_dict = {
                    t: pd.DataFrame(prices_dict).loc[
                        pd.DataFrame(prices_dict).index <= date, t
                    ].dropna()
                    for t in df.columns
                }
                # Only include tickers with enough history
                hist_dict = {t: s for t, s in hist_dict.items() if len(s) >= self.lookback}

                current_pairs = self.find_pairs(hist_dict, as_of_date=date)
                last_discovery_idx = i
                discovery_log.append({
                    "date": date,
                    "n_pairs": len(current_pairs),
                    "pairs": [(p.stock_a, p.stock_b, p.pvalue) for p in current_pairs],
                })

            # For each discovered pair, check Z-score signals
            for pair_info in current_pairs:
                a, b = pair_info.stock_a, pair_info.stock_b
                if a not in df.columns or b not in df.columns:
                    continue

                # Get historical prices up to current date for Z-score
                hist_a = df.loc[df.index <= date, a]
                hist_b = df.loc[df.index <= date, b]
                if len(hist_a) < self.zscore_window + 1:
                    continue

                spread = self.compute_spread(hist_a, hist_b, pair_info.hedge_ratio)
                zscore = self.compute_zscore(spread)
                if zscore.empty or pd.isna(zscore.iloc[-1]):
                    continue
                z = float(zscore.iloc[-1])

                # Check if we already have a position in this pair
                existing = [
                    p for p in active_positions
                    if not p.closed and p.stock_a == a and p.stock_b == b
                ]

                if existing:
                    pos = existing[0]
                    # Exit signal: |z| < exit_threshold or z crosses zero relative to entry
                    should_exit = abs(z) < self.exit_zscore
                    # Also exit if z crossed zero (mean reversion complete)
                    if (pos.direction == -1 and z <= 0) or (
                        pos.direction == 1 and z >= 0
                    ):
                        should_exit = True

                    if should_exit:
                        cash += self._close_position(pos, df, date)
                        all_trades.append(pos)
                        active_positions = [
                            p for p in active_positions if not p.closed
                        ]
                else:
                    # Entry signal
                    if len(active_positions) >= self.max_pairs:
                        continue

                    if abs(z) > self.entry_zscore:
                        # Capital per pair: divide remaining investable cash equally
                        # among remaining capacity
                        n_available = self.max_pairs - len(active_positions)
                        capital_per_pair = cash / max(n_available, 1)
                        # Each pair has two legs, each gets half the capital
                        capital_per_leg = capital_per_pair / 2.0

                        price_a = float(df.loc[date, a])
                        price_b = float(df.loc[date, b])
                        if price_a <= 0 or price_b <= 0:
                            continue

                        # Direction: z > entry => short spread (short A, long B)
                        direction = -1 if z > self.entry_zscore else 1

                        shares_a = capital_per_leg / price_a
                        shares_b = capital_per_leg / price_b

                        # Entry transaction costs (both legs)
                        entry_cost = capital_per_leg * self._tc_rate * 2
                        cash -= entry_cost

                        pos = PairPosition(
                            stock_a=a,
                            stock_b=b,
                            hedge_ratio=pair_info.hedge_ratio,
                            direction=direction,
                            entry_date=date,
                            entry_spread=float(spread.iloc[-1]),
                            capital_per_leg=capital_per_leg,
                            shares_a=shares_a,
                            shares_b=shares_b,
                        )
                        # Reserve capital for the position
                        cash -= capital_per_pair
                        active_positions.append(pos)

            # Mark-to-market: cash + unrealized PnL of active positions
            mtm = cash
            for pos in active_positions:
                if pos.closed:
                    continue
                price_a = float(df.loc[date, pos.stock_a])
                price_b = float(df.loc[date, pos.stock_b])
                # Value of position A leg
                val_a = pos.shares_a * price_a
                # Value of position B leg
                val_b = pos.shares_b * price_b
                if pos.direction == 1:
                    # Long A, short B: value = capital_per_leg + (val_a - entry_a) - (val_b - entry_b)
                    # Simplified: total capital deployed + unrealized on both legs
                    mtm += val_a + (pos.capital_per_leg * 2 - val_b)
                else:
                    # Short A, long B: value = capital_per_leg + (entry_a - val_a) + (val_b - entry_b)
                    mtm += (pos.capital_per_leg * 2 - val_a) + val_b

            equity_values.append(mtm)
            equity_dates.append(date)

        # Close any remaining open positions at the end
        if dates.size > 0:
            final_date = dates[-1]
            for pos in active_positions:
                if not pos.closed:
                    cash += self._close_position(pos, df, final_date)
                    all_trades.append(pos)

        equity = pd.Series(equity_values, index=equity_dates, name="equity")

        return PairsBacktestResult(
            equity_curve=equity,
            trades=all_trades,
            pair_discovery_log=discovery_log,
        )

    def _close_position(
        self,
        pos: PairPosition,
        df: pd.DataFrame,
        date: pd.Timestamp,
    ) -> float:
        """Close a position. Returns the cash recovered (capital + PnL - costs)."""
        price_a = float(df.loc[date, pos.stock_a])
        price_b = float(df.loc[date, pos.stock_b])

        val_a = pos.shares_a * price_a
        val_b = pos.shares_b * price_b

        if pos.direction == 1:
            # Was long A, short B
            # PnL from A (long): val_a - capital_per_leg
            # PnL from B (short): capital_per_leg - val_b
            pnl = (val_a - pos.capital_per_leg) + (pos.capital_per_leg - val_b)
            cash_back = pos.capital_per_leg * 2 + pnl
        else:
            # Was short A, long B
            pnl = (pos.capital_per_leg - val_a) + (val_b - pos.capital_per_leg)
            cash_back = pos.capital_per_leg * 2 + pnl

        # Exit transaction costs (both legs)
        exit_cost = (val_a + val_b) * self._tc_rate
        cash_back -= exit_cost

        pos.closed = True
        pos.exit_date = date
        pos.pnl = pnl - exit_cost - (pos.capital_per_leg * self._tc_rate * 2)
        return cash_back
