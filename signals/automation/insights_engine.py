"""Daily insights engine: runs all models, stores signals, blends, and reports."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from signals.automation.cash_overlay import CashOverlay
from signals.automation.signal_store import SignalStore
from signals.model.momentum import CrossSectionalMomentum
from signals.model.multifactor import MultiFactor
from signals.model.news_filter import NewsFilter
from signals.model.pead import PEADStrategy
from signals.model.tsmom import TimeSeriesMomentum
from signals.utils.logging import get_logger

log = get_logger(__name__)

# Default tickers for momentum (top 20 liquid US large caps)
DEFAULT_MOMENTUM_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AVGO", "JPM", "UNH",
    "V", "MA", "HD", "PG", "JNJ",
    "COST", "ABBV", "CRM", "AMD", "NFLX",
]


def _load_full_universe_tickers() -> list[str]:
    """Load all tickers with price data from data/raw/.

    The full-universe evaluation (2026-04-21) showed that expanding from
    S&P 500 only (~500 tickers) to all available stocks (1100+) lifts
    Sharpe from 0.659 to 1.075. This function discovers all available
    tickers by scanning the raw data directory.
    """
    from pathlib import Path

    raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    if not raw_dir.exists():
        return DEFAULT_MOMENTUM_TICKERS
    tickers = []
    for f in raw_dir.glob("*_1d.parquet"):
        ticker = f.stem.replace("_1d", "")
        # Skip non-equity tickers (ETFs, crypto, indices)
        if ticker.endswith("-USD") or ticker.startswith("^"):
            continue
        if ticker in ("SPY", "EFA", "TLT", "IEF", "GLD", "USO", "DBA",
                       "VIX", "TNX", "UUP", "MDY", "IWM"):
            continue
        tickers.append(ticker)
    return sorted(tickers) if tickers else DEFAULT_MOMENTUM_TICKERS

# Default TSMOM asset classes (ETF proxies)
DEFAULT_TSMOM_TICKERS = [
    "SPY",       # US equities
    "EFA",       # International equities
    "TLT",       # Long-term treasuries
    "IEF",       # Intermediate treasuries
    "GLD",       # Gold
    "USO",       # Oil
    "DBA",       # Agriculture
    "BTC-USD",   # Bitcoin
]


class InsightsEngine:
    """Automated daily signal generation and reporting.

    Runs each model (momentum, TSMOM, PEAD), stores signals in the SignalStore,
    blends via CashOverlay, and produces a human-readable report.
    """

    def __init__(
        self,
        signal_store: SignalStore,
        cash_overlay: CashOverlay,
        data_store=None,
        tickers: list[str] | None = None,
        tsmom_tickers: list[str] | None = None,
        momentum_model: CrossSectionalMomentum | None = None,
        tsmom_model: TimeSeriesMomentum | None = None,
        pead_model: PEADStrategy | None = None,
        use_multifactor: bool = False,
        multifactor_model: MultiFactor | None = None,
        fundamentals: pd.DataFrame | None = None,
        news_filter: NewsFilter | None = None,
        sectors: dict[str, str] | None = None,
    ) -> None:
        self.signal_store = signal_store
        self.cash_overlay = cash_overlay
        self.data_store = data_store
        self.tickers = tickers or _load_full_universe_tickers()
        self.tsmom_tickers = tsmom_tickers or DEFAULT_TSMOM_TICKERS
        self.sectors = sectors
        self.momentum_model = momentum_model or CrossSectionalMomentum(
            lookback_days=252, skip_days=21, n_long=15, rebalance_freq=21,
            mode="early_breakout", max_per_sector=2, short_lookback=63,
        )
        self.tsmom_model = tsmom_model or TimeSeriesMomentum(
            lookback_days=252, vol_window=63, risk_parity=True, rebalance_freq=21,
        )
        self.pead_model = pead_model or PEADStrategy(
            surprise_threshold_pct=5.0, hold_days=60, max_positions=5,
        )
        self.use_multifactor = use_multifactor
        self.multifactor_model = multifactor_model or MultiFactor(
            momentum_weight=0.40, value_weight=0.30, quality_weight=0.30,
            n_long=5, vol_filter_quantile=0.75,
        )
        self.fundamentals = fundamentals if fundamentals is not None else pd.DataFrame()
        self.news_filter = news_filter or NewsFilter(
            lookback_days=7, max_risk_score=3,
        )
        self._last_report: dict | None = None

    def _load_prices(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Load price data for a list of tickers from the data store."""
        prices_dict: dict[str, pd.DataFrame] = {}
        if self.data_store is None:
            return prices_dict
        for ticker in tickers:
            try:
                df = self.data_store.load(ticker, "1d")
                if not df.empty:
                    prices_dict[ticker] = df
            except Exception as e:
                log.warning("Failed to load %s: %s", ticker, e)
        return prices_dict

    def _refresh_prices(self, tickers: list[str]) -> None:
        """Attempt to refresh price data (best-effort, no failure on error).

        Only refreshes tickers whose latest data is more than 1 trading day
        old, to avoid hammering Yahoo for 500+ tickers every run.
        """
        if self.data_store is None:
            return
        try:
            from signals.data.pipeline import DataPipeline
            from signals.data.yahoo import YahooFinanceSource
            pipeline = DataPipeline(source=YahooFinanceSource(), store=self.data_store)

            stale: list[str] = []
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
            for ticker in tickers:
                last = self.data_store.last_timestamp(ticker, "1d")
                if last is None or last < cutoff:
                    stale.append(ticker)

            if stale:
                log.info("Refreshing %d/%d stale tickers", len(stale), len(tickers))
            for ticker in stale:
                try:
                    pipeline.refresh(ticker, interval="1d")
                except Exception as e:
                    log.warning("Refresh failed for %s: %s", ticker, e)
        except ImportError:
            log.info("Data pipeline not available, skipping refresh")

    def run_momentum(
        self, prices_dict: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """Run cross-sectional momentum (or multifactor) and return target weights."""
        if not prices_dict:
            return {}
        # Use the latest date across all tickers
        latest_dates = []
        for df in prices_dict.values():
            if not df.empty:
                latest_dates.append(df.index.max())
        if not latest_dates:
            return {}
        as_of = max(latest_dates)

        if self.use_multifactor:
            # Load fundamentals on first use if not provided
            if self.fundamentals.empty:
                log.info("Fetching fundamentals for multifactor model...")
                self.fundamentals = self.multifactor_model.fetch_fundamentals(
                    list(prices_dict.keys())
                )
            weights = self.multifactor_model.rank(
                prices_dict, self.fundamentals, as_of_date=as_of,
            )
        else:
            weights = self.momentum_model.rank(
                prices_dict, as_of_date=as_of, sectors=self.sectors,
            )

        return {t: w for t, w in weights.items() if w > 0}

    def run_tsmom(
        self, prices_dict: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """Run TSMOM and return target weights."""
        if not prices_dict:
            return {}
        latest_dates = []
        for df in prices_dict.values():
            if not df.empty:
                latest_dates.append(df.index.max())
        if not latest_dates:
            return {}
        as_of = max(latest_dates)
        weights = self.tsmom_model.signals(prices_dict, as_of_date=as_of)
        return {t: w for t, w in weights.items() if w > 0}

    def run_pead(
        self,
        prices_dict: dict[str, pd.DataFrame],
        earnings_df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Run PEAD check and return target weights for active positions.

        If no earnings_df is provided, returns empty (no earnings events
        to react to today). In production, the caller should provide recent
        earnings data from an external source.
        """
        if earnings_df is None or earnings_df.empty:
            return {}
        trades = self.pead_model.generate_trades(earnings_df, prices_dict)
        if trades.empty:
            return {}
        # Convert active trades to equal-weight allocation
        active_tickers = trades["ticker"].unique().tolist()
        if not active_tickers:
            return {}
        weight_per = 1.0 / len(active_tickers)
        return {t: weight_per for t in active_tickers}

    def run_daily(self, earnings_df: pd.DataFrame | None = None) -> dict:
        """Execute all models for today, store signals, return report dict.

        Steps:
        1. Refresh price data for all tickers (best-effort)
        2. Run momentum ranking
        3. Run TSMOM signals
        4. Run PEAD check
        5. Store all signals in SignalStore
        6. Blend via CashOverlay
        7. Generate human-readable report
        8. Return report dict with all details
        """
        now = datetime.now(tz=UTC)
        all_tickers = list(set(self.tickers + self.tsmom_tickers))

        # Step 1: Refresh
        self._refresh_prices(all_tickers)

        # Step 2: Load prices
        momentum_prices = self._load_prices(self.tickers)
        tsmom_prices = self._load_prices(self.tsmom_tickers)

        # Step 3: Run models
        momentum_targets = self.run_momentum(momentum_prices)

        # Step 3b: Apply news filter to momentum/multifactor picks
        news_results: list[dict] = []
        if momentum_targets:
            try:
                filtered_targets = self.news_filter.filter_signals(momentum_targets)
                # Collect news check results for the report
                for ticker in momentum_targets:
                    if momentum_targets[ticker] > 0:
                        try:
                            news_results.append(self.news_filter.check_ticker(ticker))
                        except Exception as e:
                            log.warning("News check failed for %s: %s", ticker, e)
                momentum_targets = filtered_targets
            except Exception as e:
                log.warning("News filter failed, using unfiltered targets: %s", e)

        tsmom_targets = self.run_tsmom(tsmom_prices)
        pead_targets = self.run_pead(momentum_prices, earnings_df)

        # Step 4: Store signals
        for ticker, weight in momentum_targets.items():
            signal = "BUY" if weight > 0 else "FLAT"
            self.signal_store.record_signal(
                model="momentum",
                ticker=ticker,
                signal=signal,
                weight=weight,
                confidence=0.7,  # Momentum is historically reliable
                metadata={"lookback": self.momentum_model.lookback_days},
            )

        for ticker, weight in tsmom_targets.items():
            signal = "BUY" if weight > 0 else "FLAT"
            self.signal_store.record_signal(
                model="tsmom",
                ticker=ticker,
                signal=signal,
                weight=weight,
                confidence=0.6,
                metadata={"lookback": self.tsmom_model.lookback_days},
            )

        for ticker, weight in pead_targets.items():
            self.signal_store.record_signal(
                model="pead",
                ticker=ticker,
                signal="BUY",
                weight=weight,
                confidence=0.65,
            )

        # Step 5: Store portfolio targets
        model_targets = {
            "momentum": momentum_targets,
            "tsmom": tsmom_targets,
            "pead": pead_targets,
        }

        invested_pct_mom = sum(momentum_targets.values())
        invested_pct_tsmom = sum(tsmom_targets.values())
        invested_pct_pead = sum(pead_targets.values())

        self.signal_store.record_portfolio_target(
            "momentum", momentum_targets, 1.0 - invested_pct_mom
        )
        self.signal_store.record_portfolio_target(
            "tsmom", tsmom_targets, 1.0 - invested_pct_tsmom
        )
        self.signal_store.record_portfolio_target(
            "pead", pead_targets, 1.0 - invested_pct_pead
        )

        # Step 6: Blend
        blended = self.cash_overlay.blend(model_targets)

        # Step 7: Build report dict
        report = {
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "momentum_targets": momentum_targets,
            "tsmom_targets": tsmom_targets,
            "pead_targets": pead_targets,
            "blended_allocation": blended,
            "model_targets": model_targets,
            "n_momentum_signals": len(momentum_targets),
            "n_tsmom_signals": len(tsmom_targets),
            "n_pead_signals": len(pead_targets),
            "total_invested": sum(
                v for k, v in blended.items() if k != "_CASH"
            ),
            "cash": blended.get("_CASH", 0.0),
            "use_multifactor": self.use_multifactor,
            "news_filter_results": news_results,
        }

        report["report_text"] = self.generate_report(
            model_targets, blended
        )
        self._last_report = report
        return report

    def generate_report(
        self,
        model_signals: dict[str, dict[str, float]],
        blended: dict[str, float],
    ) -> str:
        """Format a human-readable daily insights report."""
        now = datetime.now(tz=UTC)
        lines = [
            "=" * 60,
            f"  DAILY INSIGHTS REPORT  --  {now.strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
            "",
        ]

        # Momentum section
        model_label = "MultiFactor" if self.use_multifactor else "Momentum"
        mom = model_signals.get("momentum", {})
        lines.append(f"[{model_label}] Top {len(mom)} stocks (50% model weight):")
        if mom:
            for ticker in sorted(mom, key=mom.get, reverse=True):
                lines.append(f"  {ticker:8s}  weight={mom[ticker]:.2f}  -> BUY")
        else:
            lines.append("  (no data available)")
        lines.append("")

        # TSMOM section
        tsmom = model_signals.get("tsmom", {})
        lines.append(f"[TSMOM] {len(tsmom)} assets with positive trend (30% model weight):")
        if tsmom:
            for ticker in sorted(tsmom, key=tsmom.get, reverse=True):
                lines.append(f"  {ticker:8s}  weight={tsmom[ticker]:.2f}  -> BUY")
        else:
            lines.append("  (no trend signals or no data)")
        lines.append("")

        # PEAD section
        pead = model_signals.get("pead", {})
        lines.append(f"[PEAD] {len(pead)} active earnings drift positions (20% model weight):")
        if pead:
            for ticker in sorted(pead, key=pead.get, reverse=True):
                lines.append(f"  {ticker:8s}  weight={pead[ticker]:.2f}  -> BUY")
        else:
            lines.append("  (no qualifying earnings events)")
        lines.append("")

        # Blended portfolio
        lines.append("-" * 60)
        lines.append("BLENDED PORTFOLIO TARGET:")
        lines.append("-" * 60)
        total_invested = sum(v for k, v in blended.items() if k != "_CASH")
        cash = blended.get("_CASH", 0.0)
        total = total_invested + cash

        for ticker in sorted(k for k in blended if k != "_CASH"):
            amt = blended[ticker]
            pct = amt / total * 100 if total > 0 else 0
            lines.append(f"  {ticker:8s}  ${amt:>10,.2f}  ({pct:5.1f}%)")

        lines.append(f"  {'CASH':8s}  ${cash:>10,.2f}  ({cash / total * 100 if total > 0 else 0:5.1f}%)")
        lines.append(f"  {'TOTAL':8s}  ${total:>10,.2f}")
        lines.append(
            f"  Gross exposure: {total_invested / total * 100 if total > 0 else 0:.1f}%"
        )
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def run_weekly(self, earnings_df: pd.DataFrame | None = None) -> dict:
        """Weekly deep analysis: run daily + summary statistics."""
        report = self.run_daily(earnings_df)

        # Add weekly summary: signal counts from the store over last 7 days
        recent = self.signal_store.get_latest_signals(n=500)
        if not recent.empty and "timestamp" in recent.columns:
            cutoff = (
                datetime.now(tz=UTC) - pd.Timedelta(days=7)
            ).isoformat()
            week_signals = recent[recent["timestamp"] >= cutoff]
            report["weekly_signal_count"] = len(week_signals)
            report["weekly_models_active"] = (
                week_signals["model"].nunique() if not week_signals.empty else 0
            )
        else:
            report["weekly_signal_count"] = 0
            report["weekly_models_active"] = 0

        report["report_type"] = "weekly"
        return report
