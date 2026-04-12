"""Compare equal-weight vs 4 portfolio optimizers on momentum top-10.

Loads SP500 prices from the DataStore, runs CrossSectionalMomentum to get
the top-10 picks, then applies each optimizer and runs a 4-year backtest
(2022-04 -> 2026-04) with monthly rebalancing.

Usage:
    python scripts/optimizer_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.backtest.metrics import compute_metrics
from signals.backtest.optimizers import (
    equal_volatility_weights,
    max_diversification_weights,
    mean_variance_weights,
    risk_parity_weights,
)
from signals.data.storage import DataStore
from signals.model.momentum import CrossSectionalMomentum


def equal_weight(
    prices_dict: dict[str, pd.DataFrame],
    selected_tickers: list[str],
    lookback: int = 60,
) -> dict[str, float]:
    """Baseline: equal weight across selected tickers."""
    n = len(selected_tickers)
    if n == 0:
        return {}
    w = 1.0 / n
    return {t: w for t in selected_tickers}


OPTIMIZERS = {
    "Equal-Weight": equal_weight,
    "Risk-Parity": risk_parity_weights,
    "Mean-Variance": mean_variance_weights,
    "Max-Diversification": max_diversification_weights,
    "Equal-Volatility": equal_volatility_weights,
}

START = "2022-04-01"
END = "2026-04-01"
N_LONG = 10
REBALANCE_FREQ = 21  # ~monthly
INITIAL_CASH = 10_000.0
COMMISSION_BPS = 5.0
SLIPPAGE_BPS = 5.0


def run_backtest_with_optimizer(
    prices_dict: dict[str, pd.DataFrame],
    optimizer_fn,
    start: str = START,
    end: str = END,
) -> pd.Series:
    """Walk-forward backtest: momentum selects, optimizer weights."""
    mom = CrossSectionalMomentum(
        lookback_days=252,
        skip_days=21,
        n_long=N_LONG,
        rebalance_freq=REBALANCE_FREQ,
    )

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    all_dates: set[pd.Timestamp] = set()
    for df in prices_dict.values():
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        all_dates.update(df.index[mask])
    trading_dates = sorted(all_dates)

    if not trading_dates:
        return pd.Series(dtype=float)

    holdings: dict[str, float] = {sym: 0.0 for sym in prices_dict}
    cash = INITIAL_CASH
    cost_rate = (COMMISSION_BPS + SLIPPAGE_BPS) * 1e-4
    equity_points: list[tuple[pd.Timestamp, float]] = []
    bars_since_rebalance = REBALANCE_FREQ

    for date in trading_dates:
        prices: dict[str, float] = {}
        for sym, df in prices_dict.items():
            if date in df.index:
                prices[sym] = float(df.loc[date, "close"])

        bars_since_rebalance += 1
        if bars_since_rebalance >= REBALANCE_FREQ:
            # Step 1: Momentum selects top-N
            raw_weights = mom.rank(prices_dict, as_of_date=date)
            selected = [t for t, w in raw_weights.items() if w > 0]

            if selected:
                # Step 2: Optimizer re-weights the selected tickers
                opt_weights = optimizer_fn(prices_dict, selected)

                # Merge: non-selected get 0
                new_weights = {sym: 0.0 for sym in prices_dict}
                for t, w in opt_weights.items():
                    new_weights[t] = w
            else:
                new_weights = {sym: 0.0 for sym in prices_dict}

            # Compute equity
            equity = cash
            for sym in holdings:
                if sym in prices:
                    equity += holdings[sym] * prices[sym]

            if equity > 0:
                # Transaction costs
                for sym in prices_dict:
                    if sym not in prices:
                        continue
                    price = prices[sym]
                    current_value = holdings[sym] * price
                    target_value = new_weights.get(sym, 0.0) * equity
                    trade_value = abs(target_value - current_value)
                    if trade_value > 1e-6:
                        cash -= trade_value * cost_rate

                equity_after_costs = cash
                for sym in holdings:
                    if sym in prices:
                        equity_after_costs += holdings[sym] * prices[sym]

                # Liquidate
                for sym in prices_dict:
                    if sym not in prices:
                        holdings[sym] = 0.0
                        continue
                    cash += holdings[sym] * prices[sym]
                    holdings[sym] = 0.0

                # Buy
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


def main() -> None:
    """Run the comparison and print results."""
    print("Loading SP500 prices from DataStore...")
    from signals.config import SETTINGS
    store = DataStore(SETTINGS.data.dir)

    # Load all SP500 tickers
    import pandas as pd
    sp500 = pd.read_csv("/tmp/sp500_with_sectors.csv")
    tickers = sp500["Symbol"].tolist()
    prices_dict = {}
    for t in tickers:
        try:
            df = store.load(t, "1d").sort_index()
            df.index = df.index.normalize()
            if len(df) >= 500:
                prices_dict[t] = df
        except Exception:
            pass
    print(f"Loaded {len(prices_dict)} tickers")

    results: list[dict] = []

    for name, opt_fn in OPTIMIZERS.items():
        print(f"\nRunning {name}...")
        equity = run_backtest_with_optimizer(prices_dict, opt_fn)

        if equity.empty:
            print(f"  {name}: no data")
            continue

        metrics = compute_metrics(equity, trades=[], periods_per_year=252)
        results.append({
            "Optimizer": name,
            "Sharpe": f"{metrics.sharpe:.3f}",
            "CAGR": f"{metrics.cagr:.2%}",
            "Max DD": f"{metrics.max_drawdown:.2%}",
            "Final Equity": f"${metrics.final_equity:,.0f}",
        })
        print(f"  Sharpe={metrics.sharpe:.3f}  CAGR={metrics.cagr:.2%}  "
              f"MaxDD={metrics.max_drawdown:.2%}")

    # Print table
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print("OPTIMIZER COMPARISON: Momentum Top-10, 2022-04 -> 2026-04")
        print("=" * 70)
        print(df.to_string(index=False))

        # Save markdown
        out_path = Path(__file__).parent / "OPTIMIZER_EVAL.md"
        with open(out_path, "w") as f:
            f.write("# Optimizer Comparison: Momentum Top-10\n\n")
            f.write(f"Period: {START} to {END} | Rebalance: monthly | "
                    f"Universe: {len(prices_dict)} SP500 stocks\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
