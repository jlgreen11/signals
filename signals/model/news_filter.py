"""News sentiment filter: post-signal check for event-driven risks.

After momentum or multifactor identifies trade candidates, this filter
searches recent news for each ticker and flags risks using keyword matching.
No LLMs -- keeps it simple and free.

Risk categories:
  - Geopolitical: war, sanctions, tariffs, embargoes
  - Regulatory: SEC actions, FDA rejections, antitrust, lawsuits
  - One-off catalysts: M&A, earnings restatements, bankruptcies
  - Sector shocks: bans, shortages, crises, halts

Uses yfinance .news (free, no API key) as the data source.

Usage::

    from signals.model.news_filter import NewsFilter

    nf = NewsFilter()
    result = nf.check_ticker("NVDA")
    filtered = nf.filter_signals({"NVDA": 0.5, "AAPL": 0.5})
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from signals.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_RISK_KEYWORDS: dict[str, list[str]] = {
    "geopolitical": [
        "war", "sanctions", "tariff", "embargo", "invasion", "missile",
        "conflict", "military", "geopolitical",
    ],
    "regulatory": [
        "SEC", "FDA reject", "antitrust", "investigation", "lawsuit",
        "fine", "subpoena", "indictment", "probe", "regulatory",
    ],
    "one_off": [
        "acquisition", "merger", "buyout", "restatement", "bankruptcy",
        "recall", "takeover", "spin-off", "divestiture",
    ],
    "sector_shock": [
        "ban", "shortage", "crisis", "crash", "collapse", "halt",
        "shutdown", "blackout", "disruption",
    ],
}


class NewsFilter:
    """Post-signal news check to identify event-driven risks.

    Parameters
    ----------
    lookback_days : int
        How many days of recent news to check (default 7).
    risk_keywords : dict
        Mapping of category name to list of keywords to search for.
    max_risk_score : int
        Threshold for flagging a stock. >= max_risk_score triggers CAUTION,
        >= caution_threshold * 5/3 triggers SKIP.
    """

    def __init__(
        self,
        lookback_days: int = 7,
        risk_keywords: dict[str, list[str]] | None = None,
        max_risk_score: int = 3,
    ) -> None:
        self.lookback_days = lookback_days
        self.risk_keywords = risk_keywords or DEFAULT_RISK_KEYWORDS
        self.max_risk_score = max_risk_score

    def _fetch_news(self, ticker: str) -> list[dict]:
        """Fetch recent news for a ticker via yfinance.

        Returns a list of dicts with at least a 'title' key.
        Handles missing/empty data gracefully.
        """
        try:
            import yfinance as yf

            yt = yf.Ticker(ticker)
            news = yt.news or []

            cutoff = datetime.now(tz=UTC) - timedelta(days=self.lookback_days)
            results: list[dict] = []
            for article in news:
                title = article.get("title", "")
                # yfinance news items may have 'providerPublishTime' (unix ts)
                pub_time = article.get("providerPublishTime")
                if pub_time is not None:
                    try:
                        pub_dt = datetime.fromtimestamp(pub_time, tz=UTC)
                        if pub_dt < cutoff:
                            continue
                    except (TypeError, ValueError, OSError):
                        pass  # If we can't parse, include it
                if title:
                    results.append({"title": title})
            return results

        except Exception as e:
            log.warning("Failed to fetch news for %s: %s", ticker, e)
            return []

    def _score_headlines(self, headlines: list[str]) -> tuple[int, list[str]]:
        """Score a list of headlines for risk keywords.

        Returns (total_hits, list_of_triggered_categories).
        """
        total_hits = 0
        triggered: set[str] = set()

        text_blob = " ".join(headlines).lower()

        for category, keywords in self.risk_keywords.items():
            for kw in keywords:
                # Case-insensitive check; but preserve case for acronyms like SEC
                if kw.lower() in text_blob:
                    total_hits += 1
                    triggered.add(category)

        return total_hits, sorted(triggered)

    def check_ticker(self, ticker: str) -> dict:
        """Check recent news for a single ticker.

        Returns a dict with:
            ticker, n_articles, risk_score, risk_categories,
            flagged, headlines, recommendation
        """
        articles = self._fetch_news(ticker)
        headlines = [a["title"] for a in articles]
        risk_score, risk_categories = self._score_headlines(headlines)

        # Determine recommendation
        skip_threshold = max(self.max_risk_score + 2, 5)
        if risk_score >= skip_threshold:
            recommendation = "SKIP"
        elif risk_score >= self.max_risk_score:
            recommendation = "CAUTION"
        else:
            recommendation = "PROCEED"

        return {
            "ticker": ticker,
            "n_articles": len(articles),
            "risk_score": risk_score,
            "risk_categories": risk_categories,
            "flagged": risk_score >= self.max_risk_score,
            "headlines": headlines,
            "recommendation": recommendation,
        }

    def check_ticker_from_headlines(
        self, ticker: str, headlines: list[str]
    ) -> dict:
        """Check a ticker using pre-supplied headlines (for testing/offline use).

        Same return shape as check_ticker() but skips the yfinance fetch.
        """
        risk_score, risk_categories = self._score_headlines(headlines)

        skip_threshold = max(self.max_risk_score + 2, 5)
        if risk_score >= skip_threshold:
            recommendation = "SKIP"
        elif risk_score >= self.max_risk_score:
            recommendation = "CAUTION"
        else:
            recommendation = "PROCEED"

        return {
            "ticker": ticker,
            "n_articles": len(headlines),
            "risk_score": risk_score,
            "risk_categories": risk_categories,
            "flagged": risk_score >= self.max_risk_score,
            "headlines": headlines,
            "recommendation": recommendation,
        }

    def check_portfolio(self, tickers: list[str]) -> pd.DataFrame:
        """Check all tickers, return a summary DataFrame.

        Columns: ticker, n_articles, risk_score, risk_categories,
                 flagged, recommendation
        """
        results = []
        for ticker in tickers:
            result = self.check_ticker(ticker)
            results.append({
                "ticker": result["ticker"],
                "n_articles": result["n_articles"],
                "risk_score": result["risk_score"],
                "risk_categories": ", ".join(result["risk_categories"]),
                "flagged": result["flagged"],
                "recommendation": result["recommendation"],
            })

        return pd.DataFrame(results) if results else pd.DataFrame(
            columns=["ticker", "n_articles", "risk_score", "risk_categories",
                     "flagged", "recommendation"]
        )

    def filter_signals(
        self,
        signals: dict[str, float],
        use_headlines: dict[str, list[str]] | None = None,
    ) -> dict[str, float]:
        """Filter a {ticker: weight} dict by news risk.

        Removes SKIP tickers entirely and halves CAUTION ticker weights.
        Redistributes removed weight to remaining tickers proportionally.

        Parameters
        ----------
        signals : dict
            {ticker: weight} from momentum/multifactor ranking.
        use_headlines : dict, optional
            If provided, use these headlines instead of fetching from yfinance.
            Useful for testing and backtesting.

        Returns
        -------
        dict
            Filtered {ticker: weight} with weights re-normalized to sum to 1.0.
        """
        if not signals:
            return {}

        adjusted: dict[str, float] = {}

        for ticker, weight in signals.items():
            if weight <= 0:
                continue

            if use_headlines is not None:
                headlines = use_headlines.get(ticker, [])
                result = self.check_ticker_from_headlines(ticker, headlines)
            else:
                result = self.check_ticker(ticker)

            rec = result["recommendation"]
            if rec == "SKIP":
                log.info(
                    "NewsFilter: SKIP %s (risk_score=%d, categories=%s)",
                    ticker, result["risk_score"], result["risk_categories"],
                )
                continue
            elif rec == "CAUTION":
                log.info(
                    "NewsFilter: CAUTION %s (risk_score=%d) — halving weight",
                    ticker, result["risk_score"],
                )
                adjusted[ticker] = weight * 0.5
            else:
                adjusted[ticker] = weight

        # Re-normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            return {t: w / total for t, w in adjusted.items()}
        return {}
