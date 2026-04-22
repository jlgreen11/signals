"""Canonical survivorship-bias-free backtest for early-breakout momentum.

This is the SINGLE source of truth for bias-free backtesting. All sweep
scripts, feature tests, and evaluations MUST use this function instead of
reimplementing the backtest logic.

Data lives in project-local paths under data/ — NOT in /tmp.

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

To download data for the first time (or refresh):

    from signals.backtest.bias_free import download_sp500_data
    download_sp500_data()  # ~15 min, downloads 26 years for ~500 tickers
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from signals.backtest.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Project-local data paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SECTOR_CSV = DATA_DIR / "sp500_sectors.csv"
CONSTITUENT_CSV = DATA_DIR / "sp500_constituents.csv"

LEGIT_HIGH = {
    "AAPL", "AMZN", "NVDA", "NFLX", "MNST", "DECK", "AXON", "SBAC",
    "LRCX", "FIX", "BRK.B", "GOOG", "GOOGL", "MSFT", "META", "TSLA",
    "AVGO", "COST", "UNH", "LLY", "NVO", "NVR", "SEB", "BKNG", "AZO",
    "CMG", "ORLY", "MTD", "MELI",
}


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
# Data download
# ---------------------------------------------------------------------------
def download_sp500_data(
    start: str = "2000-01-01",
    end: str | None = None,
) -> None:
    """Download S&P 500 constituent prices and sector data.

    Saves to project-local data/ directory:
      - data/raw/{TICKER}_1d.parquet  (one per ticker)
      - data/sp500_sectors.csv
      - data/sp500_constituents.csv  (if available)

    Idempotent: skips tickers that already have data covering the range.
    """
    import yfinance as yf

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Get current S&P 500 tickers + sectors
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df_sp = pd.read_csv(url)
    df_sp.to_csv(SECTOR_CSV, index=False)
    tickers = df_sp["Symbol"].tolist()
    print(f"S&P 500: {len(tickers)} tickers, sectors saved to {SECTOR_CSV}")

    # 2. Download prices
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(start, tz="UTC")
    skipped = 0

    for i, ticker in enumerate(tickers):
        safe = ticker.replace("/", "_").replace("^", "")
        path = RAW_DIR / f"{safe}_1d.parquet"

        # Skip if already have data covering the range
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                if len(existing) > 100:
                    first = existing.index.min()
                    if hasattr(first, "tz") and first.tz is None:
                        first = first.tz_localize("UTC")
                    if first <= start_ts + pd.Timedelta(days=30):
                        skipped += 1
                        continue
            except Exception:
                pass

        try:
            df = yf.download(
                ticker, start=start, end=end, interval="1d",
                progress=False, auto_adjust=False, actions=False, threads=False,
            )
            if df is None or df.empty:
                print(f"  [{i+1}/{len(tickers)}] {ticker}: no data")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
            })
            df.to_parquet(path)
            print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(df)} rows", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(tickers)}] {ticker}: FAILED ({e})")

    if skipped:
        print(f"Skipped {skipped} tickers with existing data")
    print("Done.")


# ---------------------------------------------------------------------------
# Data loader (cached)
# ---------------------------------------------------------------------------
_CACHED_DATA: BiasFreData | None = None


def load_bias_free_data(
    start: str = "2000-01-01",
    end: str = "2026-04-13",
) -> BiasFreData:
    """Load and cache all data for bias-free backtesting.

    Reads from project-local data/raw/*.parquet and data/sp500_sectors.csv.
    """
    global _CACHED_DATA
    if _CACHED_DATA is not None:
        return _CACHED_DATA

    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*_1d.parquet")):
        raise FileNotFoundError(
            f"No price data in {RAW_DIR}. Run:\n"
            "  from signals.backtest.bias_free import download_sp500_data\n"
            "  download_sp500_data()"
        )

    # Load prices from DataStore-format parquet files.
    # Use adj_close (total return: split + dividend adjusted) not close
    # (split only). This matches SPY's auto_adjust=True baseline so the
    # comparison is apples-to-apples.
    prices_dict: dict[str, pd.DataFrame] = {}
    for f in RAW_DIR.glob("*_1d.parquet"):
        try:
            d = pd.read_parquet(f)
            if len(d) < 100:
                continue
            # Derive ticker from filename: AAPL_1d.parquet -> AAPL
            ticker = f.stem.rsplit("_", 1)[0].replace("_", ".")

            # Prefer adj_close (total return); fall back to close if absent
            if "adj_close" in d.columns:
                d = d.rename(columns={"close": "close_raw"})
                d["close"] = d["adj_close"]
            # Else keep existing "close" column as-is

            if ticker not in LEGIT_HIGH:
                if d["close"].max() > 5000:
                    continue
                if d["close"].pct_change().abs().max() > 3.0:
                    continue
            prices_dict[ticker] = d
        except Exception:
            pass

    if not prices_dict:
        raise FileNotFoundError(f"No valid price data found in {RAW_DIR}")

    # Sectors
    if not SECTOR_CSV.exists():
        raise FileNotFoundError(
            f"Sector CSV not found at {SECTOR_CSV}. Run download_sp500_data()."
        )
    sector_df = pd.read_csv(SECTOR_CSV)
    col = "GICS Sector" if "GICS Sector" in sector_df.columns else "Sector"
    sectors = dict(zip(sector_df["Symbol"], sector_df[col], strict=False))

    # Constituents — use if available, else fall back to current S&P 500
    if CONSTITUENT_CSV.exists():
        df_c = pd.read_csv(CONSTITUENT_CSV)
        df_c["date"] = pd.to_datetime(df_c["date"])
        cmap: dict[str, list[str]] = {}
        for _, row in df_c.iterrows():
            d = row["date"].strftime("%Y-%m-%d")
            cmap[d] = [t.strip() for t in row["tickers"].split(",") if t.strip()]
    else:
        # Fall back: treat current S&P 500 as the constituents for all dates
        all_tickers = list(sectors.keys())
        cmap = {"2000-01-01": all_tickers}

    cmap_dates = sorted(cmap.keys())

    # Trading dates — only include dates where a majority of tickers have data.
    # Normalize all timestamps to midnight UTC to avoid double-counting when
    # different data sources use different intraday offsets (e.g., 00:00 vs 04:00).
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    date_counts: dict[pd.Timestamp, int] = {}
    for t, df in prices_dict.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        # Normalize to midnight UTC
        df.index = df.index.normalize()
        prices_dict[t] = df
        for dt in df.index:
            if start_ts <= dt <= end_ts and dt.weekday() < 5:
                date_counts[dt] = date_counts.get(dt, 0) + 1
    # A real trading day should have data for at least 50 tickers
    trading_dates = sorted(d for d, c in date_counts.items() if c >= 50)

    # Build close matrix
    tickers = sorted(prices_dict.keys())
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    mat = np.full((len(trading_dates), len(tickers)), np.nan)
    date_to_row = {d: i for i, d in enumerate(trading_dates)}
    for t, df in prices_dict.items():
        col_idx = ticker_to_idx[t]
        for dt in df.index:
            if dt in date_to_row:
                mat[date_to_row[dt], col_idx] = float(df.loc[dt, "close"])

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


def clear_cache() -> None:
    """Clear the in-memory data cache (forces reload on next call)."""
    global _CACHED_DATA
    _CACHED_DATA = None


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

    The defaults here (short=21, long=126) are STANDALONE defaults.
    The canonical backtest in ``run_bias_free_backtest`` passes
    short=63, long=252 (from grid sweep), which override these.
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
# Full-universe helper
# ---------------------------------------------------------------------------
def _get_dead_tickers(data: BiasFreData) -> set[str]:
    """Identify tickers whose data ends before the last 10 bars.

    A ticker is "dead" if its last non-NaN price is more than 10 bars
    before the end of the close matrix. These are delisted or otherwise
    stale tickers that should be excluded from the full-universe mode.
    """
    n_dates = data.close_mat.shape[0]
    cutoff = n_dates - 10
    dead: set[str] = set()
    for col, ticker in enumerate(data.tickers):
        col_data = data.close_mat[:, col]
        valid_mask = ~np.isnan(col_data)
        if not valid_mask.any():
            dead.add(ticker)
            continue
        last_valid = int(np.max(np.nonzero(valid_mask)))
        if last_valid < cutoff:
            dead.add(ticker)
    return dead


# ---------------------------------------------------------------------------
# Canonical backtest
# ---------------------------------------------------------------------------
def run_bias_free_backtest(
    data: BiasFreData,
    score_fn: Callable | None = None,
    short: int = 63,
    long: int = 252,
    hold_days: int = 105,  # canonical from grid sweep (42/63/84/105/126 tested)
    n_long: int = 15,
    max_per_sector: int = 2,
    min_short_return: float = 0.10,
    max_long_return: float = 1.50,
    rebalance_freq: int = 21,
    initial_cash: float = 100_000.0,
    cost_bps: float = 10.0,
    weight_fn: Callable | None = None,
    use_full_universe: bool = False,
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
        Lookback windows for acceleration signal. Defaults 63/252 are
        the canonical values from the grid sweep. These override the
        standalone defaults in ``default_acceleration_score`` (21/126).
    hold_days : int
        Fixed hold period in trading days. Default 105 is canonical
        (from sweep over 42/63/84/105/126).
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
    use_full_universe : bool
        When True, consider ALL tickers with valid price data at each
        rebalance date instead of only point-in-time S&P 500 constituents.
        Dead tickers (last valid price >10 bars before end of data) are
        excluded. This eliminates survivorship bias from the constituent
        list itself — the scoring function still sees the same data, but
        the eligible set is broader, which typically improves diversification
        and allows the strategy to catch momentum in stocks that were not
        yet (or no longer) in the index. Default False preserves the
        original constituent-based behavior.

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

    # Pre-compute dead tickers once for full-universe mode
    if use_full_universe:
        dead_tickers = _get_dead_tickers(data)
        alive_cols = [
            data.ticker_to_idx[t]
            for t in data.tickers
            if t not in dead_tickers
        ]

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
            # Get eligible tickers: full universe or point-in-time constituents
            if use_full_universe:
                eligible_cols = [
                    c for c in alive_cols
                    if not np.isnan(mat[row, c])
                ]
            else:
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
