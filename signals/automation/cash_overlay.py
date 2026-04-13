"""Cash overlay manager: blends multi-model signals into a single portfolio."""

from __future__ import annotations

from dataclasses import dataclass, field

# Default model weights reflecting backtest Sharpe ranking
DEFAULT_MODEL_WEIGHTS = {
    "momentum": 0.50,
    "tsmom": 0.30,
    "pead": 0.20,
}


@dataclass
class CashOverlay:
    """Blends signals from multiple models into a single portfolio target.

    Parameters
    ----------
    total_capital : float
        Total paper trading capital in USD.
    model_weights : dict
        Model-level allocation weights, e.g. {'momentum': 0.50, 'tsmom': 0.30, 'pead': 0.20}.
        Must sum to <= 1.0. The remainder goes to cash.
    max_position_pct : float
        Maximum allocation to any single ticker (default 0.25).
    cash_reserve_pct : float
        Minimum cash reserve (default 0.05).
    max_gross_exposure : float
        Maximum total long exposure as fraction of capital (default 0.95).
    """

    total_capital: float = 100_000.0
    model_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MODEL_WEIGHTS))
    max_position_pct: float = 0.25
    cash_reserve_pct: float = 0.0
    max_gross_exposure: float = 1.0

    def blend(
        self, model_targets: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Combine targets from multiple models into a single {ticker: dollar_amount} allocation.

        Parameters
        ----------
        model_targets : dict
            Per-model target weights: {'momentum': {'AMD': 0.20, ...}, 'tsmom': {...}}.
            Each inner dict maps ticker -> weight (0-1), where weights within a model
            should sum to <= 1.0.

        Returns
        -------
        dict[str, float]
            Blended allocation in dollar amounts: {'AMD': 15000, ..., '_CASH': 5000}.
        """
        # Step 1: Compute raw dollar allocation per ticker
        raw_alloc: dict[str, float] = {}

        for model_name, ticker_weights in model_targets.items():
            model_weight = self.model_weights.get(model_name, 0.0)
            model_capital = self.total_capital * model_weight

            for ticker, weight in ticker_weights.items():
                if weight <= 0:
                    continue
                dollar_amount = model_capital * weight
                raw_alloc[ticker] = raw_alloc.get(ticker, 0.0) + dollar_amount

        # Step 2: Enforce per-position cap
        max_position_dollars = self.total_capital * self.max_position_pct
        capped_alloc: dict[str, float] = {}
        for ticker, amount in raw_alloc.items():
            capped_alloc[ticker] = min(amount, max_position_dollars)

        # Step 3: Enforce max gross exposure
        max_exposure_dollars = self.total_capital * self.max_gross_exposure
        total_allocated = sum(capped_alloc.values())
        if total_allocated > max_exposure_dollars and total_allocated > 0:
            scale = max_exposure_dollars / total_allocated
            capped_alloc = {t: v * scale for t, v in capped_alloc.items()}
            total_allocated = sum(capped_alloc.values())

        # Step 4: Enforce minimum cash reserve
        min_cash = self.total_capital * self.cash_reserve_pct
        max_investable = self.total_capital - min_cash
        if total_allocated > max_investable and total_allocated > 0:
            scale = max_investable / total_allocated
            capped_alloc = {t: v * scale for t, v in capped_alloc.items()}
            total_allocated = sum(capped_alloc.values())

        # Step 5: Compute cash remainder
        cash = self.total_capital - total_allocated
        capped_alloc["_CASH"] = cash

        return capped_alloc

    def rebalance_orders(
        self,
        current_positions: dict[str, float],
        target_positions: dict[str, float],
        prices: dict[str, float] | None = None,
    ) -> list[dict]:
        """Compute the trades needed to move from current to target.

        Parameters
        ----------
        current_positions : dict
            Current holdings as {ticker: dollar_value}. Exclude '_CASH'.
        target_positions : dict
            Target allocation as {ticker: dollar_value}. May include '_CASH'.
        prices : dict, optional
            Current prices for share calculation {ticker: price_per_share}.

        Returns
        -------
        list[dict]
            List of trade orders:
            {'ticker': str, 'action': 'BUY'|'SELL', 'notional': float, 'shares': float}
        """
        orders: list[dict] = []
        all_tickers = set(current_positions.keys()) | set(target_positions.keys())
        all_tickers.discard("_CASH")

        for ticker in sorted(all_tickers):
            current = current_positions.get(ticker, 0.0)
            target = target_positions.get(ticker, 0.0)
            diff = target - current

            if abs(diff) < 1.0:  # Less than $1, skip
                continue

            price = (prices or {}).get(ticker, 0.0)
            shares = abs(diff) / price if price > 0 else 0.0

            orders.append({
                "ticker": ticker,
                "action": "BUY" if diff > 0 else "SELL",
                "notional": abs(diff),
                "shares": round(shares, 4),
            })

        return orders

    def summary(self, blended: dict[str, float] | None = None) -> str:
        """Human-readable summary of the cash overlay configuration and current allocation."""
        lines = [
            "=== Cash Overlay Configuration ===",
            f"Total capital:      ${self.total_capital:,.2f}",
            f"Max position:       {self.max_position_pct:.0%}",
            f"Cash reserve:       {self.cash_reserve_pct:.0%}",
            f"Max gross exposure: {self.max_gross_exposure:.0%}",
            "",
            "Model weights:",
        ]
        for model, weight in sorted(self.model_weights.items()):
            lines.append(f"  {model:12s}  {weight:.0%}  (${self.total_capital * weight:,.0f})")

        if blended:
            lines.append("")
            lines.append("=== Blended Allocation ===")
            total_invested = sum(v for k, v in blended.items() if k != "_CASH")
            cash = blended.get("_CASH", 0.0)
            for ticker in sorted(k for k in blended if k != "_CASH"):
                amt = blended[ticker]
                pct = amt / self.total_capital * 100
                lines.append(f"  {ticker:12s}  ${amt:>10,.2f}  ({pct:5.1f}%)")
            lines.append(f"  {'_CASH':12s}  ${cash:>10,.2f}  ({cash / self.total_capital * 100:5.1f}%)")
            lines.append(f"  {'TOTAL':12s}  ${self.total_capital:>10,.2f}")
            lines.append(f"  Gross exposure:  {total_invested / self.total_capital:.1%}")

        return "\n".join(lines)
