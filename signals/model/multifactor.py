"""Multi-factor stock scoring model: momentum + value + quality + volatility filter.

Combines cross-sectional momentum (Jegadeesh & Titman 1993) with fundamental
factors (value via trailing P/E, quality via ROE) and a volatility filter to
diversify away from concentrated high-vol sector bets.

Each factor is converted to a percentile rank (0-100) across the eligible
universe. The composite score is a weighted blend:

    composite = w_mom * momentum_rank + w_val * value_rank + w_qual * quality_rank

Stocks in the top quartile of 63-day realized volatility are optionally
excluded before ranking, reducing concentration in high-vol names without
affecting the ranking of remaining stocks.

Usage::

    from signals.model.multifactor import MultiFactor

    mf = MultiFactor(momentum_weight=0.4, value_weight=0.3, quality_weight=0.3)
    fundamentals = mf.fetch_fundamentals(tickers)
    weights = mf.rank(prices_dict, fundamentals, as_of_date=pd.Timestamp("2026-04-01"))
    equity = mf.backtest(prices_dict, fundamentals, start="2022-04-01", end="2026-04-01")
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from signals.utils.logging import get_logger

log = get_logger(__name__)


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Convert a series to percentile ranks (0-100), NaN stays NaN."""
    valid = series.dropna()
    if len(valid) == 0:
        return series * 0.0
    ranked = valid.rank(method="average", pct=True) * 100.0
    return ranked.reindex(series.index)


def zscore_cross_section(values: dict[str, float]) -> dict[str, float]:
    """Z-score normalize a cross-section of factor values.

    Ported from Vibe-Trading's multi-factor signal engine. Preserves
    magnitude information: a stock at 2-sigma momentum gets weighted more
    than one at 1-sigma, unlike percentile ranking which treats 51st and
    99th percentile with equal spacing.

    Args:
        values: {ticker: raw_factor_value}. NaN values are set to 0.0.

    Returns:
        {ticker: z_score} with mean ~0 and std ~1 across the cross-section.
    """
    vals = [v for v in values.values() if not np.isnan(v)]
    if len(vals) < 2:
        return {k: 0.0 for k in values}
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    if std < 1e-12:
        return {k: 0.0 for k in values}
    return {
        k: (v - mean) / std if not np.isnan(v) else 0.0
        for k, v in values.items()
    }


class MultiFactor:
    """Composite stock scorer: momentum + value + quality + volatility filter.

    Ranks stocks on a blended score and picks the top N.

    Factors:
      - Momentum (default 40%): 12-1 month return
      - Value (default 30%): inverse P/E ratio (lower P/E = higher value score)
      - Quality (default 30%): ROE (higher = better)
      - Volatility filter: exclude stocks in the top quartile of 63-day vol

    Parameters
    ----------
    momentum_weight : float
        Weight for the momentum factor in [0, 1].
    value_weight : float
        Weight for the value factor in [0, 1].
    quality_weight : float
        Weight for the quality factor in [0, 1].
    volume_weight : float
        Weight for the volume ratio factor in [0, 1]. Default 0.0 (opt-in).
        When > 0, the volume ratio (today_volume / 20d_mean_volume) is added
        as a fourth factor. Higher volume ratio = higher conviction.
    n_long : int
        Number of top-ranked stocks to hold (equal-weight).
    vol_filter_quantile : float
        Stocks above this quantile of 63-day realized vol are excluded.
        Set to 1.0 to disable the filter.
    lookback_days : int
        Trading days for momentum return calculation.
    skip_days : int
        Recent trading days to skip (short-term reversal avoidance).
    rebalance_freq : int
        Trading days between rebalances.
    commission_bps : float
        One-way commission in basis points per rebalance.
    slippage_bps : float
        One-way slippage in basis points per rebalance.
    scoring_method : str
        "percentile" (default) or "zscore". Z-score preserves magnitude:
        a stock at 2-sigma momentum gets more weight than one at 1-sigma.
    """

    def __init__(
        self,
        momentum_weight: float = 0.40,
        value_weight: float = 0.30,
        quality_weight: float = 0.30,
        volume_weight: float = 0.0,
        n_long: int = 10,
        vol_filter_quantile: float = 0.75,
        lookback_days: int = 252,
        skip_days: int = 21,
        rebalance_freq: int = 21,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0,
        scoring_method: str = "percentile",
    ) -> None:
        total = momentum_weight + value_weight + quality_weight + volume_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Factor weights must sum to 1.0, got {total:.4f} "
                f"(mom={momentum_weight}, val={value_weight}, "
                f"qual={quality_weight}, vol={volume_weight})"
            )
        if scoring_method not in ("percentile", "zscore"):
            raise ValueError(
                f"scoring_method must be 'percentile' or 'zscore', got {scoring_method!r}"
            )
        self.momentum_weight = momentum_weight
        self.value_weight = value_weight
        self.quality_weight = quality_weight
        self.volume_weight = volume_weight
        self.n_long = n_long
        self.vol_filter_quantile = vol_filter_quantile
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.rebalance_freq = rebalance_freq
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.scoring_method = scoring_method

    def _required_history(self) -> int:
        """Minimum trading days a stock needs to be rankable."""
        return self.lookback_days + self.skip_days

    def fetch_fundamentals(
        self,
        tickers: list[str],
        cache_path: str | Path = "data/fundamentals.parquet",
        max_age_days: int = 7,
    ) -> pd.DataFrame:
        """Fetch P/E and ROE for all tickers via yfinance.

        Returns DataFrame with columns [ticker, pe_ratio, roe, fetched_at].
        Caches results to parquet; re-fetches if cache is older than max_age_days.
        """
        cache = Path(cache_path)

        # Check cache freshness
        if cache.exists():
            cached = pd.read_parquet(cache)
            if "fetched_at" in cached.columns and len(cached) > 0:
                last_fetch = pd.Timestamp(cached["fetched_at"].iloc[0])
                age = pd.Timestamp.now() - last_fetch
                if age.days < max_age_days:
                    log.info(
                        "Using cached fundamentals (%d tickers, %.1f days old)",
                        len(cached),
                        age.total_seconds() / 86400,
                    )
                    return cached

        import yfinance as yf

        rows: list[dict] = []
        now_str = pd.Timestamp.now().isoformat()
        n = len(tickers)

        for i, ticker in enumerate(tickers):
            if (i + 1) % 50 == 0 or i == 0:
                log.info("Fetching fundamentals: %d/%d (%s)", i + 1, n, ticker)
            try:
                info = yf.Ticker(ticker).info
                pe = info.get("trailingPE")
                roe = info.get("returnOnEquity")
                rows.append({
                    "ticker": ticker,
                    "pe_ratio": float(pe) if pe is not None else None,
                    "roe": float(roe) if roe is not None else None,
                    "fetched_at": now_str,
                })
            except Exception as e:
                log.warning("Failed to fetch %s: %s", ticker, e)
                rows.append({
                    "ticker": ticker,
                    "pe_ratio": None,
                    "roe": None,
                    "fetched_at": now_str,
                })
            # Brief pause to avoid rate limiting
            if (i + 1) % 10 == 0:
                time.sleep(0.5)

        df = pd.DataFrame(rows)

        # Save cache
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        log.info("Saved fundamentals cache to %s (%d rows)", cache, len(df))

        return df

    def score(
        self,
        prices_dict: dict[str, pd.DataFrame],
        fundamentals: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Compute composite score for each stock.

        Returns DataFrame with columns:
            ticker, momentum_rank, value_rank, quality_rank,
            composite_score, vol_63d, included
        """
        records: list[dict] = []
        fund_map: dict[str, dict] = {}
        if not fundamentals.empty:
            for _, row in fundamentals.iterrows():
                fund_map[row["ticker"]] = {
                    "pe_ratio": row.get("pe_ratio"),
                    "roe": row.get("roe"),
                }

        # Step 1: Compute raw momentum returns and 63-day volatility
        for symbol, df in prices_dict.items():
            eligible = df.loc[df.index <= as_of_date, "close"]
            if len(eligible) < self._required_history():
                continue

            # Momentum: 12-1 month return
            end_idx = len(eligible) - self.skip_days
            start_idx = end_idx - self.lookback_days
            if start_idx < 0 or end_idx < 1:
                continue
            p_start = eligible.iloc[start_idx]
            p_end = eligible.iloc[end_idx]
            if p_start <= 0:
                continue
            mom_return = (p_end - p_start) / p_start

            # 63-day realized volatility (annualized)
            recent = eligible.iloc[-63:] if len(eligible) >= 63 else eligible
            daily_rets = recent.pct_change().dropna()
            vol_63d = float(daily_rets.std() * np.sqrt(252)) if len(daily_rets) > 1 else 0.0

            # Fundamentals
            fund = fund_map.get(symbol, {})
            pe = fund.get("pe_ratio")
            roe = fund.get("roe")

            # Volume ratio: today's volume / 20-day mean volume
            vol_ratio = np.nan
            if self.volume_weight > 0 and "volume" in df.columns:
                vol_data = df.loc[df.index <= as_of_date, "volume"]
                if len(vol_data) >= 20:
                    today_vol = vol_data.iloc[-1]
                    mean_vol = vol_data.iloc[-20:].mean()
                    if mean_vol > 0:
                        vol_ratio = float(today_vol / mean_vol)

            records.append({
                "ticker": symbol,
                "momentum_raw": mom_return,
                "pe_ratio": pe,
                "roe": roe,
                "vol_63d": vol_63d,
                "volume_ratio": vol_ratio,
            })

        if not records:
            return pd.DataFrame(
                columns=[
                    "ticker", "momentum_rank", "value_rank", "quality_rank",
                    "composite_score", "vol_63d", "included",
                ]
            )

        df_scores = pd.DataFrame(records).set_index("ticker")

        # Step 2: Apply volatility filter
        vol_threshold = df_scores["vol_63d"].quantile(self.vol_filter_quantile)
        df_scores["included"] = df_scores["vol_63d"] <= vol_threshold

        # Step 3: Score factors among INCLUDED stocks only
        included = df_scores[df_scores["included"]].copy()

        if self.scoring_method == "zscore":
            # Z-score method: preserves magnitude information
            mom_zs = zscore_cross_section(included["momentum_raw"].to_dict())
            included["momentum_rank"] = included.index.map(
                lambda t, m=mom_zs: m.get(t, 0.0)
            )

            pe_valid = included["pe_ratio"].copy()
            pe_valid[pe_valid <= 0] = np.nan
            # Negate P/E so lower P/E = higher z-score
            neg_pe = (-pe_valid.fillna(pe_valid.max() * 10 if pe_valid.notna().sum() > 0 else 0))
            val_zs = zscore_cross_section(neg_pe.to_dict())
            included["value_rank"] = included.index.map(
                lambda t, m=val_zs: m.get(t, 0.0)
            )

            roe_series = included["roe"].copy()
            median_roe = roe_series.median() if roe_series.notna().sum() > 0 else 0.0
            roe_filled = roe_series.fillna(median_roe)
            qual_zs = zscore_cross_section(roe_filled.to_dict())
            included["quality_rank"] = included.index.map(
                lambda t, m=qual_zs: m.get(t, 0.0)
            )

            # Volume ratio z-score (if enabled)
            if self.volume_weight > 0 and "volume_ratio" in included.columns:
                vr_filled = included["volume_ratio"].fillna(1.0)
                vr_zs = zscore_cross_section(vr_filled.to_dict())
                included["volume_rank"] = included.index.map(
                    lambda t, m=vr_zs: m.get(t, 0.0)
                )
            else:
                included["volume_rank"] = 0.0

        else:
            # Percentile method (original default)
            # Momentum rank: higher return = higher rank
            included["momentum_rank"] = _percentile_rank(included["momentum_raw"])

            # Value rank: lower P/E = higher value score
            pe_valid = included["pe_ratio"].copy()
            pe_valid[pe_valid <= 0] = np.nan
            if pe_valid.notna().sum() > 0:
                included["value_rank"] = _percentile_rank(
                    -pe_valid.fillna(pe_valid.max() * 10)
                )
            else:
                included["value_rank"] = 50.0

            # Quality rank: higher ROE = higher rank
            roe_series = included["roe"].copy()
            if roe_series.notna().sum() > 0:
                median_roe = roe_series.median()
                roe_filled = roe_series.fillna(median_roe)
                included["quality_rank"] = _percentile_rank(roe_filled)
            else:
                included["quality_rank"] = 50.0

            # Volume ratio rank (if enabled)
            if self.volume_weight > 0 and "volume_ratio" in included.columns:
                vr_series = included["volume_ratio"].fillna(1.0)
                included["volume_rank"] = _percentile_rank(vr_series)
            else:
                included["volume_rank"] = 50.0  # neutral when not used

        # Step 4: Composite score
        included["composite_score"] = (
            self.momentum_weight * included["momentum_rank"]
            + self.value_weight * included["value_rank"]
            + self.quality_weight * included["quality_rank"]
            + self.volume_weight * included["volume_rank"]
        )

        # Merge back with excluded stocks
        df_scores = df_scores.join(
            included[["momentum_rank", "value_rank", "quality_rank", "composite_score"]],
            how="left",
        )

        return df_scores.reset_index()

    def rank(
        self,
        prices_dict: dict[str, pd.DataFrame],
        fundamentals: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Return {ticker: weight} for top-N stocks by composite score.

        Same interface shape as CrossSectionalMomentum.rank(), returning
        a dict with every ticker in prices_dict (0.0 for non-selected).
        """
        scored = self.score(prices_dict, fundamentals, as_of_date)

        if scored.empty:
            return {sym: 0.0 for sym in prices_dict}

        # Filter to included stocks with valid composite scores
        eligible = scored[scored["included"] & scored["composite_score"].notna()]

        if eligible.empty:
            return {sym: 0.0 for sym in prices_dict}

        # Sort by composite score descending, pick top N
        top = eligible.nlargest(min(self.n_long, len(eligible)), "composite_score")
        n_selected = len(top)
        weight = 1.0 / n_selected if n_selected > 0 else 0.0

        winners = set(top["ticker"])
        return {sym: (weight if sym in winners else 0.0) for sym in prices_dict}

    def backtest(
        self,
        prices_dict: dict[str, pd.DataFrame],
        fundamentals: pd.DataFrame,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        initial_cash: float = 10000.0,
    ) -> pd.Series:
        """Walk-forward backtest with monthly rebalancing.

        Returns an equity curve (pd.Series with DatetimeIndex).
        Transaction costs are applied identically to CrossSectionalMomentum.
        """
        if isinstance(start, pd.Timestamp) and start.tzinfo is not None:
            start_ts = start
        else:
            start_ts = pd.Timestamp(start, tz="UTC")
        if isinstance(end, pd.Timestamp) and end.tzinfo is not None:
            end_ts = end
        else:
            end_ts = pd.Timestamp(end, tz="UTC")

        # Build common date index
        all_dates: set[pd.Timestamp] = set()
        for df in prices_dict.values():
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            all_dates.update(df.index[mask])
        trading_dates = sorted(all_dates)

        if not trading_dates:
            return pd.Series(dtype=float)

        holdings: dict[str, float] = {sym: 0.0 for sym in prices_dict}
        cash = initial_cash

        equity_points: list[tuple[pd.Timestamp, float]] = []
        bars_since_rebalance = self.rebalance_freq  # Force rebalance on first bar
        cost_rate = (self.commission_bps + self.slippage_bps) * 1e-4

        for date in trading_dates:
            prices: dict[str, float] = {}
            for sym, df in prices_dict.items():
                if date in df.index:
                    prices[sym] = float(df.loc[date, "close"])

            bars_since_rebalance += 1
            if bars_since_rebalance >= self.rebalance_freq:
                new_weights = self.rank(prices_dict, fundamentals, as_of_date=date)

                equity = cash
                for sym in holdings:
                    if sym in prices:
                        equity += holdings[sym] * prices[sym]

                if equity > 0:
                    # Compute transaction costs
                    for sym in prices_dict:
                        if sym not in prices:
                            continue
                        price = prices[sym]
                        current_value = holdings[sym] * price
                        target_value = new_weights.get(sym, 0.0) * equity
                        trade_value = abs(target_value - current_value)
                        if trade_value > 1e-6:
                            cash -= trade_value * cost_rate

                    # Recompute equity after costs
                    equity_after_costs = cash
                    for sym in holdings:
                        if sym in prices:
                            equity_after_costs += holdings[sym] * prices[sym]

                    # Liquidate all holdings
                    for sym in prices_dict:
                        if sym not in prices:
                            holdings[sym] = 0.0
                            continue
                        cash += holdings[sym] * prices[sym]
                        holdings[sym] = 0.0

                    # Buy new targets
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
