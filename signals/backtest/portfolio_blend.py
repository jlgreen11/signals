"""PortfolioCombiner — weighted blend of multiple asset strategies.

Productionization of the research finding in `scripts/BTC_SP500_PORTFOLIO_RESULTS.md`:
a 40/60 BTC/SP constant-mix portfolio with daily rebalancing produced
median Sharpe 2.44 on the seed-42 random-window eval and averaged
Sharpe +16% over BTC-alone across 4 seeds. The research script that
produced that result was a one-off; this module is the reusable
abstraction.

Usage:

    from signals.backtest.engine import BacktestConfig, BacktestEngine
    from signals.backtest.portfolio_blend import PortfolioCombiner, PortfolioAllocation

    # Step 1: define component allocations
    btc_cfg = BacktestConfig(model_type="hybrid", ...)
    sp_cfg = None  # None means "buy and hold"

    allocations = [
        PortfolioAllocation(symbol="BTC-USD", cfg=btc_cfg, weight=0.4),
        PortfolioAllocation(symbol="^GSPC", cfg=sp_cfg, weight=0.6),
    ]

    # Step 2: combine per-component equity curves
    combiner = PortfolioCombiner(
        allocations=allocations,
        rebalance="daily",  # or "window"
    )
    port_eq = combiner.combine(
        component_equities={"BTC-USD": btc_eq, "^GSPC": sp_eq},
    )

The component equity curves are assumed to already be computed. This
module does NOT run the backtest engine — it's a pure post-hoc combiner.
That's the simplest shape that covers the research finding.

For a full end-to-end portfolio backtest (where the engine runs each
component, then the combiner blends), use `run_portfolio_backtest()`.

## Rebalancing modes

- `"window"`: allocate W_btc and W_sp at window start, let them drift
  independently. No mid-window rebalancing. Equivalent to buying at t=0
  and holding to t=T.

- `"daily"`: compute each day's component return, combine as
  `w_btc × btc_ret + w_sp × sp_ret`, compound. Equivalent to rebalancing
  to target weights every bar. Free in backtest (no trading costs
  modeled); in reality would accrue ~5 bps per rebalance per leg.

- `"threshold"`: rebalance only when any weight has drifted more than
  `threshold_pct` from target. More tax-efficient than daily in
  production. Not yet implemented (scheduled as a follow-up).

## Date alignment

Different assets trade on different calendars (crypto 7d/week, equities
5d/week). The combiner uses a master index (the asset with the most
trading days, typically BTC for crypto-equity mixes) and forward-fills
the lower-frequency assets on their off-days. Off-day returns for the
forward-filled asset are 0 (position unchanged, no mark-to-market move).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from signals.backtest.engine import BacktestConfig

RebalanceMode = Literal["window", "daily"]


@dataclass
class PortfolioAllocation:
    """One component of a multi-asset portfolio.

    - symbol: the data store symbol (e.g. "BTC-USD", "^GSPC")
    - cfg: the BacktestConfig to run that component, or None for
      buy-and-hold (simpler — no engine run, just price-relative
      equity curve)
    - weight: allocation weight. All weights in a portfolio must sum to 1.
    """

    symbol: str
    cfg: BacktestConfig | None
    weight: float


class PortfolioCombiner:
    """Combine multiple component equity curves into a portfolio
    equity curve at a fixed constant-mix allocation.

    This class handles the mechanical "given these per-asset equity
    curves, compute the portfolio equity curve" problem. It does NOT
    run the backtest engine; that's the caller's responsibility
    (either via the helper function `run_portfolio_backtest()` below
    or by calling `BacktestEngine.run()` yourself and passing the
    results here).
    """

    def __init__(
        self,
        allocations: list[PortfolioAllocation],
        rebalance: RebalanceMode = "daily",
        initial_cash: float = 10_000.0,
    ):
        if not allocations:
            raise ValueError("at least one allocation required")
        total = sum(a.weight for a in allocations)
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"weights must sum to 1.0; got {total:.6f}"
            )
        if rebalance not in ("window", "daily"):
            raise ValueError(
                f"rebalance must be 'window' or 'daily'; got {rebalance!r}"
            )
        self.allocations = list(allocations)
        self.rebalance = rebalance
        self.initial_cash = float(initial_cash)

    def combine(
        self,
        component_equities: dict[str, pd.Series],
    ) -> pd.Series:
        """Combine per-symbol equity curves into a portfolio equity curve.

        Each input Series should be the equity of the component as a
        function of time, starting from any positive value (will be
        rebased). Indices must be DatetimeIndex. Asset-specific
        calendars are handled via date normalization + forward-fill.

        Returns a single portfolio equity Series on the master calendar
        (the union of all component calendars), rebased to
        `self.initial_cash` at the first timestamp where all components
        have a value.
        """
        if not component_equities:
            return pd.Series(dtype=float)

        # Normalize indices to dates (strip any intraday time component)
        normalized: dict[str, pd.Series] = {}
        for sym, eq in component_equities.items():
            if eq.empty:
                return pd.Series(dtype=float)
            copy = eq.copy()
            copy.index = copy.index.normalize()
            normalized[sym] = copy

        # Master calendar: union of all component calendars. Take the
        # first date where ALL components have seen a value (so the
        # forward-fill is grounded).
        all_dates = pd.DatetimeIndex([])
        for eq in normalized.values():
            all_dates = all_dates.union(eq.index)
        all_dates = all_dates.sort_values()

        # Reindex each onto the union, forward-fill to handle missing
        # trading days for equities (weekends etc).
        reindexed: dict[str, pd.Series] = {}
        for sym, eq in normalized.items():
            series = eq.reindex(all_dates, method="ffill")
            # Before the first real value, fill with the initial price
            # (position hasn't been taken yet — equity is flat at start).
            first_valid = series.first_valid_index()
            if first_valid is None:
                return pd.Series(dtype=float)
            series = series.loc[first_valid:]
            reindexed[sym] = series

        # Align all reindexed series onto the intersection of their
        # "from-first-valid" ranges so everyone starts at the same date.
        master_start = max(s.index[0] for s in reindexed.values())
        reindexed = {
            sym: s.loc[master_start:] for sym, s in reindexed.items()
        }
        master_index = reindexed[self.allocations[0].symbol].index

        # Normalize each to start at 1.0
        normed = {
            sym: (s / s.iloc[0]) for sym, s in reindexed.items()
        }
        # Reindex any with shorter index onto the master (shouldn't
        # happen after the above alignment, but be safe)
        normed = {
            sym: s.reindex(master_index, method="ffill").fillna(1.0)
            for sym, s in normed.items()
        }

        weight_map = {a.symbol: a.weight for a in self.allocations}

        if self.rebalance == "window":
            port_normed = sum(
                weight_map[sym] * series
                for sym, series in normed.items()
            )
            result = port_normed * self.initial_cash
        else:
            # daily rebalancing
            port_returns = pd.Series(0.0, index=master_index)
            for sym, series in normed.items():
                returns = series.pct_change().fillna(0.0)
                port_returns = port_returns + weight_map[sym] * returns
            result = (1.0 + port_returns).cumprod() * self.initial_cash

        # Ensure the result has a proper DatetimeIndex (the intermediate
        # union/reindex operations can downgrade to a plain Index, which
        # breaks downstream code like compute_metrics._annualization_factor).
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.DatetimeIndex(result.index)
        return result


def run_portfolio_backtest(
    allocations: list[PortfolioAllocation],
    prices_by_symbol: dict[str, pd.DataFrame],
    rebalance: RebalanceMode = "daily",
    initial_cash: float = 10_000.0,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """End-to-end portfolio backtest: run each component's strategy
    through the BacktestEngine (or buy-and-hold if cfg is None), then
    combine them with PortfolioCombiner.

    `prices_by_symbol` should map each allocation's symbol to a prices
    DataFrame (e.g. from `DataStore.load`). start and end trim the
    evaluation period; the backtest engine still uses its own training
    window (`cfg.train_window`) before `start`.

    Returns a single portfolio equity Series.
    """
    from signals.backtest.engine import BacktestEngine

    component_equities: dict[str, pd.Series] = {}

    for allocation in allocations:
        sym = allocation.symbol
        if sym not in prices_by_symbol:
            raise ValueError(f"missing prices for symbol {sym!r}")
        prices = prices_by_symbol[sym].sort_index()
        if start is not None:
            prices = prices.loc[prices.index >= start]
        if end is not None:
            prices = prices.loc[prices.index <= end]
        if prices.empty:
            raise ValueError(f"no price data for {sym} in the requested range")

        if allocation.cfg is None:
            # Buy and hold
            eq = (prices["close"] / prices["close"].iloc[0]) * initial_cash
        else:
            result = BacktestEngine(allocation.cfg).run(prices, symbol=sym)
            eq = result.equity_curve
            if eq.empty:
                raise ValueError(f"empty equity curve for {sym}")

        component_equities[sym] = eq

    combiner = PortfolioCombiner(
        allocations=allocations,
        rebalance=rebalance,
        initial_cash=initial_cash,
    )
    return combiner.combine(component_equities)


# ============================================================
# Helpful constants — the validated default portfolio
# ============================================================


def default_btc_sp_allocation() -> list[PortfolioAllocation]:
    """The 40/60 BTC/SP portfolio that averaged +16% Sharpe over
    BTC-alone across 4 seeds in the Tier-2 robustness check.

    BTC uses the production H-Vol hybrid default; SP uses buy & hold.

    Use like:
        from signals.backtest.portfolio_blend import (
            default_btc_sp_allocation,
            run_portfolio_backtest,
        )
        port_eq = run_portfolio_backtest(
            allocations=default_btc_sp_allocation(),
            prices_by_symbol={"BTC-USD": btc_df, "^GSPC": sp_df},
            rebalance="daily",
        )
    """
    btc_cfg = BacktestConfig(
        model_type="hybrid",
        train_window=1000,
        retrain_freq=21,
        n_states=5,
        order=5,
        return_bins=3,
        volatility_bins=3,
        vol_window=10,
        laplace_alpha=0.01,
        hybrid_routing_strategy="vol",
        hybrid_vol_quantile=0.70,
    )
    return [
        PortfolioAllocation(symbol="BTC-USD", cfg=btc_cfg, weight=0.4),
        PortfolioAllocation(symbol="^GSPC", cfg=None, weight=0.6),
    ]
