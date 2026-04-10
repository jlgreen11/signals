"""S&P 500 HOMC order sweep — the "Markov memory example dataset".

Produces a per-order summary of how higher-order Markov chains behave on
^GSPC daily closes. For each order in {1, 2, 3, 4, 5, 6, 7, 8, 9}:

  1. Fit HOMC on the same 16 random 6-month windows as the other S&P
     evaluations (seed 42, train_window=1000, 5 quantile states)
  2. Record transition-table stats: distinct k-tuples observed, median
     support (how many times each k-tuple was seen), most-frequent
     k-tuple and its support count
  3. Record the random-window eval metrics: median Sharpe, median CAGR,
     mean max DD, positive CAGR windows
  4. Compare to buy & hold

This is the "example dataset" of Markov memory on S&P. It answers the
question: "Does ANY memory order work on S&P, or does the whole model
class fail equally at all depths?"

The analogous BTC experiment was Tier-0 (order=7 broken) → Tier-0a
(order=5/w=1000 surprising) → Tier-0f (order-3 through order-5 tested
in the random-window eval). On BTC the optimal was order=5 with a hybrid
router. S&P may have a different answer — or confirm that no order works.

Note on data: uses the most-recent 1 fit from each retrain cycle to
summarize the transition-table stats. The stats are a snapshot of one
typical fit during the eval, not an aggregate across all 240+ fits —
but they're representative of what the model "sees" at each order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.backtest.engine import BacktestConfig, BacktestEngine
from signals.backtest.metrics import Metrics, compute_metrics
from signals.config import SETTINGS
from signals.data.storage import DataStore
from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility

SYMBOL = "^GSPC"
START = pd.Timestamp("2015-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31", tz="UTC")

ORDER_GRID = list(range(1, 10))  # 1 through 9 inclusive
N_STATES = 5
TRAIN_WINDOW = 1000
VOL_WINDOW = 10


@dataclass
class OrderRow:
    order: int
    # Transition-table stats from a representative training window
    distinct_ktuples: int
    median_support: float
    max_support: int
    most_frequent_history: str
    most_frequent_support: int
    # Random-window eval metrics
    mean_sharpe: float
    median_sharpe: float
    mean_cagr: float
    median_cagr: float
    mean_max_dd: float
    positive_windows: int
    n_windows: int


def _run_on_window(
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    start_i: int,
    end_i: int,
    symbol: str,
) -> Metrics:
    warmup_pad = 5
    slice_start = start_i - cfg.train_window - cfg.vol_window - warmup_pad
    if slice_start < 0:
        slice_start = 0
    engine_input = prices.iloc[slice_start:end_i]
    eval_start_ts = prices.index[start_i]

    try:
        result = BacktestEngine(cfg).run(engine_input, symbol=symbol)
    except Exception as e:
        print(f"  engine error order={cfg.order}: {e}")
        return compute_metrics(pd.Series(dtype=float), [])

    eq = result.equity_curve.loc[result.equity_curve.index >= eval_start_ts]
    if eq.empty or eq.iloc[0] <= 0:
        return compute_metrics(pd.Series(dtype=float), [])
    eq_rebased = (eq / eq.iloc[0]) * cfg.initial_cash
    return compute_metrics(eq_rebased, [])


def _snapshot_transition_stats(
    order: int, features: pd.DataFrame
) -> tuple[int, float, int, str, int]:
    """Fit a single HOMC on the supplied features and return transition
    table stats: (distinct_ktuples, median_support, max_support,
    most_frequent_label, most_frequent_support).

    The HigherOrderMarkovChain stores normalized transition probabilities,
    not raw counts. To recover support, we re-walk the encoded training
    series and count k-tuple occurrences directly.
    """
    from signals.model.states import QuantileStateEncoder

    encoder = QuantileStateEncoder(n_bins=N_STATES, feature="return_1d")
    clean = features[["return_1d"]].dropna()
    if len(clean) <= order + 5:
        return 0, 0.0, 0, "<insufficient data>", 0
    encoded = encoder.fit_transform(clean).dropna().astype(int)
    states = encoded.to_numpy()

    counts: dict[tuple[int, ...], int] = {}
    for i in range(order, len(states)):
        key = tuple(int(x) for x in states[i - order : i])
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return 0, 0.0, 0, "<empty>", 0

    supports = np.array(list(counts.values()), dtype=np.int64)
    distinct = int(len(counts))
    median_support = float(np.median(supports))
    max_support = int(supports.max())
    most_frequent_key = max(counts, key=lambda k: counts[k])
    most_frequent_support = int(counts[most_frequent_key])

    # Format the k-tuple using bin labels
    def _bin_name(idx: int) -> str:
        names = {0: "q0", 1: "q1", 2: "q2", 3: "q3", 4: "q4"}
        return names.get(idx, f"q{idx}")

    label = "→".join(_bin_name(x) for x in most_frequent_key)
    return distinct, median_support, max_support, label, most_frequent_support


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    prices = store.load(SYMBOL, "1d").sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
    print(f"{SYMBOL}: {len(prices)} bars ({prices.index[0].date()} → {prices.index[-1].date()})")

    # Build features on the full price series for the transition-stats
    # snapshot. Use a representative training window (the most-recent
    # TRAIN_WINDOW bars) as a stand-in for what the HOMC typically sees.
    feats_full = pd.DataFrame(index=prices.index)
    feats_full["return_1d"] = log_returns(prices["close"])
    feats_full["volatility_20d"] = rolling_volatility(feats_full["return_1d"], window=VOL_WINDOW)
    feats_full = feats_full.dropna()
    representative_window = feats_full.iloc[-TRAIN_WINDOW:]

    # Random-window setup: same 16 windows as the rest of the S&P eval suite
    warmup_pad = 5
    six_months = 126
    n_windows = 16
    seed = 42
    min_start = TRAIN_WINDOW + VOL_WINDOW + warmup_pad
    max_start = len(prices) - six_months - 1
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(min_start, max_start), n_windows))

    print(f"Sweeping HOMC orders {ORDER_GRID[0]}..{ORDER_GRID[-1]} on {n_windows} random windows")
    print()

    # Compute B&H baseline once across the same 16 windows
    bh_sharpes: list[float] = []
    bh_cagrs: list[float] = []
    for start_i in starts:
        end_i = start_i + six_months
        eval_window = prices.iloc[start_i:end_i]
        bh_eq = (eval_window["close"] / eval_window["close"].iloc[0]) * 10_000.0
        m_bh = compute_metrics(bh_eq, [])
        bh_sharpes.append(m_bh.sharpe)
        bh_cagrs.append(m_bh.cagr)
    bh_median_sharpe = float(np.median(bh_sharpes))
    bh_median_cagr = float(np.median(bh_cagrs))

    print(f"Buy & hold baseline: median Sharpe {bh_median_sharpe:.2f}, median CAGR {bh_median_cagr * 100:+.1f}%")
    print()

    results: list[OrderRow] = []
    for order in ORDER_GRID:
        # Transition stats snapshot from the representative window
        distinct, median_sup, max_sup, mf_label, mf_support = _snapshot_transition_stats(
            order, representative_window
        )

        # Random-window eval
        cfg = BacktestConfig(
            model_type="homc",
            train_window=TRAIN_WINDOW,
            retrain_freq=21,
            n_states=N_STATES,
            order=order,
            vol_window=VOL_WINDOW,
            laplace_alpha=1.0,
        )
        sharpes: list[float] = []
        cagrs: list[float] = []
        mdds: list[float] = []
        for start_i in starts:
            end_i = start_i + six_months
            m = _run_on_window(cfg, prices, start_i, end_i, SYMBOL)
            sharpes.append(m.sharpe)
            cagrs.append(m.cagr)
            mdds.append(m.max_drawdown)

        row = OrderRow(
            order=order,
            distinct_ktuples=distinct,
            median_support=median_sup,
            max_support=max_sup,
            most_frequent_history=mf_label,
            most_frequent_support=mf_support,
            mean_sharpe=float(np.mean(sharpes)),
            median_sharpe=float(np.median(sharpes)),
            mean_cagr=float(np.mean(cagrs)),
            median_cagr=float(np.median(cagrs)),
            mean_max_dd=float(np.mean(mdds)),
            positive_windows=int(sum(1 for c in cagrs if c > 0)),
            n_windows=n_windows,
        )
        results.append(row)

        print(
            f"  order={order} | "
            f"distinct k-tuples={row.distinct_ktuples:4d} (of {N_STATES ** order} possible) | "
            f"median support={row.median_support:5.1f} | "
            f"mean Sh={row.mean_sharpe:5.2f} | "
            f"median Sh={row.median_sharpe:5.2f} | "
            f"median CAGR={row.median_cagr * 100:+6.1f}% | "
            f"pos={row.positive_windows}/{n_windows}"
        )

    print()
    print("=" * 140)
    print(f"HOMC order sweep — {SYMBOL}")
    print("=" * 140)
    print(
        f"{'order':>5} {'distinct':>9} {'possible':>10} {'median sup':>11} "
        f"{'max sup':>8} {'mean Sh':>8} {'median Sh':>10} "
        f"{'median CAGR':>13} {'mean MDD':>10} {'pos/N':>8}"
    )
    for r in results:
        possible = N_STATES ** r.order
        print(
            f"{r.order:>5d} {r.distinct_ktuples:>9d} {possible:>10d} "
            f"{r.median_support:>11.1f} {r.max_support:>8d} "
            f"{r.mean_sharpe:>8.2f} {r.median_sharpe:>10.2f} "
            f"{r.median_cagr * 100:>12.1f}% "
            f"{r.mean_max_dd * 100:>9.1f}% "
            f"{r.positive_windows:>3d}/{r.n_windows}"
        )

    print()
    print("Most-frequent k-tuple per order (from a representative 1000-bar training window):")
    print("-" * 100)
    for r in results:
        print(
            f"  order={r.order}: history='{r.most_frequent_history}' "
            f"observed {r.most_frequent_support} times "
            f"({r.most_frequent_support / max(1, TRAIN_WINDOW - r.order) * 100:.1f}% of possible positions)"
        )

    print()
    print("=" * 140)
    print(f"Verdict — HOMC memory depth on {SYMBOL}")
    print("=" * 140)
    print(f"Buy & hold baseline: median Sharpe {bh_median_sharpe:.2f}, median CAGR {bh_median_cagr * 100:+.1f}%")
    print()
    best = max(results, key=lambda r: r.median_sharpe)
    print(f"Best HOMC: order={best.order}, median Sharpe {best.median_sharpe:.2f}, median CAGR {best.median_cagr * 100:+.1f}%")
    if best.median_sharpe > bh_median_sharpe:
        print(f"  → HOMC@order={best.order} BEATS buy & hold on S&P by "
              f"{best.median_sharpe - bh_median_sharpe:+.2f} Sharpe")
    else:
        print(f"  → No HOMC order beats buy & hold on S&P. Best gap: "
              f"{best.median_sharpe - bh_median_sharpe:+.2f} Sharpe")

    # Sparsity analysis: at which order does coverage drop below 10%?
    coverage_threshold_order = None
    for r in results:
        possible = N_STATES ** r.order
        coverage = r.distinct_ktuples / possible
        if coverage < 0.10:
            coverage_threshold_order = r.order
            break
    if coverage_threshold_order is not None:
        print(
            f"\nSparsity wall: at order={coverage_threshold_order}, "
            f"fewer than 10% of possible k-tuples are observed in training. "
            f"Higher orders are structurally undertrained."
        )


if __name__ == "__main__":
    main()
