"""Post-Earnings Announcement Drift (PEAD) strategy.

Academic background: after an earnings surprise, stocks tend to drift in
the direction of the surprise for 60+ trading days. This is the
second-strongest anomaly in the finance literature (after momentum).

This module implements a long-only PEAD strategy:
  - After a positive surprise exceeding the threshold: go long for hold_days.
  - After a negative surprise: stay flat (long-only version).
  - Position sizing: equal-weight across active signals, capped at max_positions.
  - Transaction costs: configurable entry/exit cost in basis points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PEADStrategy:
    """Post-Earnings Announcement Drift strategy.

    Parameters
    ----------
    surprise_threshold_pct : float
        Minimum |surprise_pct| to trigger a trade (in percentage points).
    hold_days : int
        Number of trading days to hold the position after the earnings event.
    max_positions : int
        Maximum number of concurrent positions.
    cost_bps : float
        Round-trip cost in basis points (applied as cost_bps/10000 on entry
        and again on exit, i.e. 5 bps + 5 bps = 10 bps round-trip).
    """

    surprise_threshold_pct: float = 5.0
    hold_days: int = 60
    max_positions: int = 5
    cost_bps: float = 5.0

    def generate_trades(
        self,
        earnings_df: pd.DataFrame,
        prices_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate trade entries from earnings data and price history.

        Parameters
        ----------
        earnings_df : pd.DataFrame
            Must have columns: [ticker, report_date, surprise_pct].
            surprise_pct is in percentage points (e.g. 10.0 = 10%).
        prices_dict : dict[str, pd.DataFrame]
            Mapping from ticker to OHLCV DataFrame with DatetimeIndex and
            a 'close' column.

        Returns
        -------
        pd.DataFrame
            Columns: [ticker, entry_date, exit_date, direction, weight,
                       entry_price, exit_price, gross_return, net_return]
        """
        if earnings_df.empty:
            return pd.DataFrame(
                columns=["ticker", "entry_date", "exit_date", "direction",
                          "weight", "entry_price", "exit_price",
                          "gross_return", "net_return"]
            )

        trades: list[dict] = []

        # Filter for positive surprises exceeding threshold (long-only)
        mask = earnings_df["surprise_pct"].notna() & (
            earnings_df["surprise_pct"] > self.surprise_threshold_pct
        )
        qualified = earnings_df.loc[mask].copy()
        qualified = qualified.sort_values("report_date")

        for _, row in qualified.iterrows():
            ticker = row["ticker"]
            report_date = pd.Timestamp(row["report_date"])

            if ticker not in prices_dict:
                continue
            prices = prices_dict[ticker]
            if prices.empty:
                continue

            # Normalize report_date to match price index timezone
            if prices.index.tz is not None and report_date.tzinfo is None:
                report_date = report_date.tz_localize(prices.index.tz)
            elif prices.index.tz is None and report_date.tzinfo is not None:
                report_date = report_date.tz_localize(None)

            # Entry: next trading day after earnings announcement
            future_dates = prices.index[prices.index > report_date]
            if len(future_dates) < 2:
                continue
            entry_date = future_dates[0]

            # Exit: hold_days trading days after entry
            entry_loc = prices.index.get_loc(entry_date)
            exit_loc = min(entry_loc + self.hold_days, len(prices) - 1)
            exit_date = prices.index[exit_loc]

            entry_price = float(prices.loc[entry_date, "close"])
            exit_price = float(prices.loc[exit_date, "close"])

            if entry_price <= 0:
                continue

            gross_return = (exit_price / entry_price) - 1.0
            # Apply costs on entry and exit
            cost_frac = self.cost_bps / 10_000.0
            net_return = gross_return - 2 * cost_frac  # entry + exit

            trades.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": "long",
                "weight": 1.0,  # Placeholder; actual weights set in backtest
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
            })

        df = pd.DataFrame(trades)
        if df.empty:
            return pd.DataFrame(
                columns=["ticker", "entry_date", "exit_date", "direction",
                          "weight", "entry_price", "exit_price",
                          "gross_return", "net_return"]
            )
        return df

    def backtest(
        self,
        earnings_df: pd.DataFrame,
        prices_dict: dict[str, pd.DataFrame],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        initial_cash: float = 10_000.0,
    ) -> pd.Series:
        """Walk-forward backtest returning a daily equity curve.

        The strategy allocates equal weight to each active PEAD signal,
        capped at ``max_positions``. Days with no active signals hold cash.

        Parameters
        ----------
        earnings_df : pd.DataFrame
            Earnings data with columns [ticker, report_date, surprise_pct].
        prices_dict : dict[str, pd.DataFrame]
            Per-ticker OHLCV data (must have 'close').
        start, end : str or Timestamp, optional
            Date range for the equity curve.
        initial_cash : float
            Starting capital.

        Returns
        -------
        pd.Series
            Daily equity curve indexed by date.
        """
        # Build the full trading calendar from the union of all price indices
        all_dates: set[pd.Timestamp] = set()
        for prices in prices_dict.values():
            all_dates.update(prices.index.tolist())

        if not all_dates:
            return pd.Series(dtype=float)

        calendar = sorted(all_dates)
        calendar = pd.DatetimeIndex(calendar)

        if start is not None:
            start_ts = pd.Timestamp(start)
            if calendar.tz is not None and start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize(calendar.tz)
            calendar = calendar[calendar >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            if calendar.tz is not None and end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize(calendar.tz)
            calendar = calendar[calendar <= end_ts]

        if len(calendar) == 0:
            return pd.Series(dtype=float)

        # Generate all trades
        trades_df = self.generate_trades(earnings_df, prices_dict)

        # Build per-day position map: for each day, which trades are active?
        # Active = entry_date <= day < exit_date
        equity = np.full(len(calendar), initial_cash, dtype=float)

        # Pre-compute daily returns per ticker for efficiency
        ticker_daily_ret: dict[str, pd.Series] = {}
        for ticker, prices in prices_dict.items():
            closes = prices["close"]
            ticker_daily_ret[ticker] = closes.pct_change().fillna(0.0)

        # Walk forward day by day
        cash = initial_cash
        # Track active positions: list of (ticker, entry_date, exit_date, alloc_amount)
        active_positions: list[dict] = []

        # Convert trades to list of dicts for iteration
        trade_list = trades_df.to_dict("records") if not trades_df.empty else []

        for d_idx, day in enumerate(calendar):
            # 1. Check for new entries today
            new_today = [
                t for t in trade_list
                if pd.Timestamp(t["entry_date"]) == day
            ]

            # 2. Check for exits today
            exits_today = []
            still_active = []
            for pos in active_positions:
                if day >= pd.Timestamp(pos["exit_date"]):
                    exits_today.append(pos)
                else:
                    still_active.append(pos)

            # Process exits: convert position value back to cash
            for pos in exits_today:
                ticker = pos["ticker"]
                # Get today's close price
                if ticker in prices_dict and day in prices_dict[ticker].index:
                    current_price = float(prices_dict[ticker].loc[day, "close"])
                    shares = pos["shares"]
                    exit_value = shares * current_price
                    # Apply exit cost
                    cost_frac = self.cost_bps / 10_000.0
                    exit_value *= (1.0 - cost_frac)
                    cash += exit_value

            active_positions = still_active

            # Process new entries
            if new_today:
                # How many slots available?
                available_slots = self.max_positions - len(active_positions)
                # Sort by surprise magnitude (highest first)
                new_today.sort(
                    key=lambda t: abs(t.get("gross_return", 0)), reverse=True
                )
                entries = new_today[:max(0, available_slots)]

                if entries and cash > 0:
                    alloc_per = cash / len(entries)
                    for t in entries:
                        ticker = t["ticker"]
                        entry_price = t["entry_price"]
                        # Apply entry cost
                        cost_frac = self.cost_bps / 10_000.0
                        effective_alloc = alloc_per * (1.0 - cost_frac)
                        shares = effective_alloc / entry_price
                        cash -= alloc_per
                        active_positions.append({
                            "ticker": ticker,
                            "entry_date": t["entry_date"],
                            "exit_date": t["exit_date"],
                            "shares": shares,
                            "entry_price": entry_price,
                        })

            # 3. Mark to market: equity = cash + sum of position values
            total_position_value = 0.0
            for pos in active_positions:
                ticker = pos["ticker"]
                if ticker in prices_dict and day in prices_dict[ticker].index:
                    current_price = float(prices_dict[ticker].loc[day, "close"])
                    total_position_value += pos["shares"] * current_price
                else:
                    # Use entry price if no price available today
                    total_position_value += pos["shares"] * pos["entry_price"]

            equity[d_idx] = cash + total_position_value

        return pd.Series(equity, index=calendar, name="equity")


@dataclass
class PEADTradeStats:
    """Summary statistics for a set of PEAD trades."""

    n_trades: int = 0
    n_winners: int = 0
    n_losers: int = 0
    win_rate: float = 0.0
    avg_gross_return: float = 0.0
    avg_net_return: float = 0.0
    median_net_return: float = 0.0
    total_net_return: float = 0.0


def summarize_trades(trades_df: pd.DataFrame) -> PEADTradeStats:
    """Compute summary statistics for PEAD trades."""
    if trades_df.empty or "net_return" not in trades_df.columns:
        return PEADTradeStats()

    n = len(trades_df)
    winners = (trades_df["net_return"] > 0).sum()
    losers = (trades_df["net_return"] <= 0).sum()

    return PEADTradeStats(
        n_trades=n,
        n_winners=int(winners),
        n_losers=int(losers),
        win_rate=float(winners / n) if n > 0 else 0.0,
        avg_gross_return=float(trades_df["gross_return"].mean()),
        avg_net_return=float(trades_df["net_return"].mean()),
        median_net_return=float(trades_df["net_return"].median()),
        total_net_return=float(trades_df["net_return"].sum()),
    )
