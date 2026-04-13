"""Canonical survivorship-bias-free backtest for early-breakout momentum.

This is the SINGLE source of truth for bias-free backtesting. All sweep
scripts, feature tests, and evaluations MUST use this function instead of
reimplementing the backtest logic. This prevents the implementation drift
that caused different scripts to report 8.7%, 11.8%, 13.3%, and 20.5%
CAGR for the same baseline config.

Usage::

    from signals.backtest.bias_free import run_bias_free_backtest, load_bias_free_data

    data = load_bias_free_data()
    result = run_bias_free_backtest(data)
    print(f"CAGR: {result.cagr:.1%}, Sharpe: {result.sharpe:.3f}")

    # With custom scoring function:
    def my_scorer(close_mat, row, col, short=21, long=126):
        '''Return a score for this stock, or None to skip.'''
        ...
    result = run_bias_free_backtest(data, score_fn=my_scorer)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from signals.backtest.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class BiasFreData:
    """All data needed for a bias-free backtest."""

    close_mat: np.ndarray  # (n_dates, n_tickers) close prices, NaN where missing
    tickers: list[str]  # ticker names, aligned to columns of close_mat
    ticker_to_idx: dict[str, int]
    trading_dates: list[pd.Timestamp]
    constituent_map: dict[str, list[str]]  # date_str -> [tickers]
    constituent_dates: list[str]  # sorted date strings
    sectors: dict[str, str]  # ticker -> GICS sector


@dataclass
class BacktestResult:
    """Results from a bias-free backtest."""

    sharpe: float
    cagr: float
    max_drawdown: float
    final_equity: float
    win_rate: float
    n_trades: int
    equity_series: pd.Series
    trade_returns: list[float]


# ---------------------------------------------------------------------------
# Data loader (cached)
# ---------------------------------------------------------------------------
_CACHED_DATA: BiasFreData | None = None

CACHE_DIR = Path("/tmp/sp500_price_cache")
CONSTITUENT_CSV = "/tmp/sp500/S&P 500 Historical Components & Changes(01-17-2026).csv"
SECTOR_CSV = "/tmp/sp500_with_sectors.csv"

LEGIT_HIGH = {
    "AAPL", "AMZN", "NVDA", "NFLX", "MNST", "DECK", "AXON", "SBAC",
    "LRCX", "FIX", "BRK.B", "GOOG", "GOOGL", "MSFT", "META", "TSLA",
    "AVGO", "COST", "UNH", "LLY", "NVO", "NVR", "SEB", "BKNG", "AZO",
    "CMG", "ORLY", "MTD", "MELI",
}


def load_bias_free_data(
    start: str = "2000-01-01",
    end: str = "2026-04-11",
) -> BiasFreData:
    """Load and cache all data for bias-free backtesting."""
    global _CACHED_DATA
    if _CACHED_DATA is not None:
        return _CACHED_DATA

    # Load prices
    prices_dict: dict[str, pd.DataFrame] = {}
    for f in CACHE_DIR.glob("*.parquet"):
        try:
            d = pd.read_parquet(f)
            if len(d) < 100:
                continue
            ticker = f.stem.replace("_", ".")
            if ticker not in LEGIT_HIGH:
                if d["close"].max() > 5000:
                    continue
                if d["close"].pct_change().abs().max() > 3.0:
                    continue
            prices_dict[ticker] = d
        except Exception:
            pass

    # Sectors
    sector_df = pd.read_csv(SECTOR_CSV)
    sectors = dict(zip(sector_df["Symbol"], sector_df["GICS Sector"], strict=True))

    # Constituents
    df_c = pd.read_csv(CONSTITUENT_CSV)
    df_c["date"] = pd.to_datetime(df_c["date"])
    cmap: dict[str, list[str]] = {}
    for _, row in df_c.iterrows():
        d = row["date"].strftime("%Y-%m-%d")
        cmap[d] = [t.strip() for t in row["tickers"].split(",") if t.strip()]
    cmap_dates = sorted(cmap.keys())

    # Trading dates
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    all_dates: set[pd.Timestamp] = set()
    for df in prices_dict.values():
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        all_dates.update(df.index[mask])
    trading_dates = sorted(all_dates)

    # Build close matrix
    tickers = sorted(prices_dict.keys())
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    mat = np.full((len(trading_dates), len(tickers)), np.nan)
    date_to_row = {d: i for i, d in enumerate(trading_dates)}
    for t, df in prices_dict.items():
        col = ticker_to_idx[t]
        for dt in df.index:
            if dt in date_to_row:
                mat[date_to_row[dt], col] = float(df.loc[dt, "close"])

    _CACHED_DATA = BiasFreData(
        close_mat=mat,
        tickers=tickers,
        ticker_to_idx=ticker_to_idx,
        trading_dates=trading_dates,
        constituent_map=cmap,
        constituent_dates=cmap_dates,
        sectors=sectors,
    )
    return _CACHED_DATA


# ---------------------------------------------------------------------------
# Constituent lookup
# ---------------------------------------------------------------------------
def _get_constituents(
    data: BiasFreData, date: pd.Timestamp
) -> list[str]:
    """Get SP500 constituents for a date (nearest prior in map)."""
    d_str = date.strftime("%Y-%m-%d")
    dates = data.constituent_dates
    lo, hi, best = 0, len(dates) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if dates[mid] <= d_str:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return data.constituent_map[dates[best]]


# ---------------------------------------------------------------------------
# Default scoring: early-breakout acceleration
# ---------------------------------------------------------------------------
def default_acceleration_score(
    close_mat: np.ndarray,
    row: int,
    col: int,
    short: int = 21,
    long: int = 126,
    min_short_return: float = 0.10,
    max_long_return: float = 1.50,
) -> float | None:
    """Compute acceleration score for a stock. Returns None to skip.

    Acceleration = short_return - (long_return / (long / short))

    This is the CANONICAL formula. Do not change without updating all
    documentation and re-running the full validation suite.
    """
    if row < long:
        return None

    p_now = close_mat[row, col]
    p_short = close_mat[row - short, col]
    p_long = close_mat[row - long, col]

    if np.isnan(p_now) or np.isnan(p_short) or np.isnan(p_long):
        return None
    if p_short <= 0 or p_long <= 0:
        return None

    ret_short = p_now / p_short - 1.0
    ret_long = p_now / p_long - 1.0

    # Minimum short-term return filter
    if ret_short <= min_short_return:
        return None

    # Moonshot filter (on LONG window, not 252-day)
    if ret_long > max_long_return:
        return None

    # Acceleration: short return vs long-term pace scaled to short window
    accel = ret_short - ret_long / (long / short)
    return accel


# ---------------------------------------------------------------------------
# Canonical backtest
# ---------------------------------------------------------------------------
def run_bias_free_backtest(
    data: BiasFreData,
    score_fn: Callable | None = None,
    short: int = 21,
    long: int = 126,
    hold_days: int = 105,
    n_long: int = 15,
    max_per_sector: int = 2,
    min_short_return: float = 0.10,
    max_long_return: float = 1.50,
    rebalance_freq: int = 21,
    initial_cash: float = 100_000.0,
    cost_bps: float = 10.0,
    weight_fn: Callable | None = None,
) -> BacktestResult:
    """Run the canonical survivorship-bias-free backtest.

    Parameters
    ----------
    data : BiasFreData
        Loaded via load_bias_free_data().
    score_fn : callable, optional
        Custom scoring function: (close_mat, row, col, short, long) -> float|None.
        If None, uses default_acceleration_score.
    short / long : int
        Lookback windows for acceleration signal.
    hold_days : int
        Fixed hold period in trading days.
    n_long : int
        Max number of positions.
    max_per_sector : int
        Max positions per GICS sector.
    min_short_return : float
        Minimum short-window return to enter.
    max_long_return : float
        Maximum long-window return (moonshot filter).
    rebalance_freq : int
        Days between entry checks (positions still exit on fixed schedule).
    initial_cash : float
        Starting capital.
    cost_bps : float
        Round-trip transaction cost in basis points.
    weight_fn : callable, optional
        Custom position weighting: (selected_cols, close_mat, row) -> dict[col, weight].
        If None, uses equal weight.

    Returns
    -------
    BacktestResult
    """
    if score_fn is None:
        def score_fn(cm: np.ndarray, r: int, c: int, s: int = short, lg: int = long) -> float | None:
            return default_acceleration_score(cm, r, c, s, lg, min_short_return, max_long_return)

    mat = data.close_mat
    cost_rate = cost_bps * 1e-4
    n_dates = len(data.trading_dates)

    # Holdings: col -> {ep: entry_price, sh: shares, entry_row: int, sec: str}
    holdings: dict[int, dict] = {}
    cash = initial_cash
    equity_points: list[float] = []
    trade_returns: list[float] = []
    bars_since_rebal = rebalance_freq  # trigger on first eligible bar

    for row in range(n_dates):
        # === STEP 1: Fixed-hold exits ===
        for col in list(holdings):
            if (row - holdings[col]["entry_row"]) >= hold_days:
                p = mat[row, col]
                if not np.isnan(p):
                    pnl = p / holdings[col]["ep"] - 1.0
                    cash += holdings[col]["sh"] * p * (1 - cost_rate)
                    trade_returns.append(pnl)
                del holdings[col]

        # === STEP 2: Deploy idle cash into existing holdings ===
        if holdings and cash > 100:
            n_held = len(holdings)
            per = cash / n_held
            for col in holdings:
                p = mat[row, col]
                if not np.isnan(p) and p > 0:
                    holdings[col]["sh"] += per / p
                    cash -= per

        # === STEP 3: Check for new entries (monthly) ===
        bars_since_rebal += 1
        if bars_since_rebal >= rebalance_freq and row >= long:
            # Get point-in-time constituents
            eligible_tickers = _get_constituents(data, data.trading_dates[row])
            eligible_cols = [
                data.ticker_to_idx[t]
                for t in eligible_tickers
                if t in data.ticker_to_idx
            ]

            # Score all eligible stocks
            candidates: list[tuple[int, float, str]] = []
            for col in eligible_cols:
                if col in holdings:
                    continue  # already held
                score = score_fn(mat, row, col, short, long)
                if score is None:
                    continue
                ticker = data.tickers[col]
                sector = data.sectors.get(ticker, "Unknown")
                candidates.append((col, score, sector))

            # Sort by score descending
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Apply sector cap
            n_slots = n_long - len(holdings)
            if n_slots > 0 and candidates:
                sector_count: dict[str, int] = {}
                for h in holdings.values():
                    s = h["sec"]
                    sector_count[s] = sector_count.get(s, 0) + 1

                selected: list[tuple[int, str]] = []
                for col, _score, sector in candidates:
                    if len(selected) >= n_slots:
                        break
                    if sector_count.get(sector, 0) >= max_per_sector:
                        continue
                    selected.append((col, sector))
                    sector_count[sector] = sector_count.get(sector, 0) + 1

                # Size and buy
                equity = cash
                for col, h in holdings.items():
                    p = mat[row, col]
                    if not np.isnan(p):
                        equity += h["sh"] * p

                if selected and equity > 0:
                    target_n = max(len(holdings) + len(selected), n_long)
                    per_pos = equity / target_n

                    for col, sector in selected:
                        p = mat[row, col]
                        if np.isnan(p) or p <= 0:
                            continue
                        cost = per_pos * (1 + cost_rate)
                        if cost <= cash:
                            holdings[col] = {
                                "ep": p,
                                "sh": per_pos / p,
                                "entry_row": row,
                                "sec": sector,
                            }
                            cash -= cost

                    # Deploy remaining cash
                    if holdings and cash > 100:
                        n_held = len(holdings)
                        per = cash / n_held
                        for col in holdings:
                            p = mat[row, col]
                            if not np.isnan(p) and p > 0:
                                holdings[col]["sh"] += per / p
                                cash -= per

            bars_since_rebal = 0

        # === STEP 4: Mark equity ===
        equity = cash
        for col, h in holdings.items():
            p = mat[row, col]
            if not np.isnan(p):
                equity += h["sh"] * p
        equity_points.append(equity)

    # Final liquidation for remaining positions
    for col in list(holdings):
        p = mat[n_dates - 1, col]
        if not np.isnan(p):
            pnl = p / holdings[col]["ep"] - 1.0
            trade_returns.append(pnl)

    # Compute metrics
    equity_s = pd.Series(
        equity_points,
        index=pd.DatetimeIndex(data.trading_dates[:len(equity_points)]),
    )
    m = compute_metrics(equity_s, trades=[], periods_per_year=252)
    tr = np.array(trade_returns)
    win_rate = float((tr > 0).mean()) if len(tr) > 0 else 0.0

    return BacktestResult(
        sharpe=m.sharpe,
        cagr=m.cagr,
        max_drawdown=m.max_drawdown,
        final_equity=m.final_equity,
        win_rate=win_rate,
        n_trades=len(tr),
        equity_series=equity_s,
        trade_returns=trade_returns,
    )
