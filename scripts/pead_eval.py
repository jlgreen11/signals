"""Post-Earnings Announcement Drift (PEAD) evaluation.

Fetches earnings data for 20 major US stocks, runs the PEAD strategy
across a grid of (surprise_threshold, hold_days) parameters, and
compares performance against SP500 B&H and equal-weight 20-stock B&H.

The PEAD anomaly is fundamentally different from the project's existing
price/vol pattern models: it uses FUNDAMENTAL information (earnings
surprises) that the market systematically underreacts to. After an
earnings surprise, stocks drift in the direction of the surprise for
60+ trading days. This is the second-strongest anomaly in the academic
literature.

Output:
  scripts/data/pead_eval.parquet  — raw per-config results
  scripts/PEAD_RESULTS.md        — comparison table
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.earnings import fetch_earnings_yfinance
from signals.data.storage import DataStore
from signals.model.pead import PEADStrategy, summarize_trades

warnings.filterwarnings("ignore", category=FutureWarning)

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "XOM", "LLY",
    "AVGO", "COST", "NFLX", "AMD", "ADBE",
]

START = "2021-04-01"
END = "2026-04-01"

PPY = 252.0  # equity calendar annualization
RF = historical_usd_rate("2018-2024")
COST_BPS = 5.0  # 5 bps entry + 5 bps exit

# Parameter grid
SURPRISE_THRESHOLDS = [3.0, 5.0, 10.0]
HOLD_DAYS_GRID = [30, 60, 90]


def _load_prices(store: DataStore) -> dict[str, pd.DataFrame]:
    """Load price data for all tickers from DataStore."""
    prices_dict: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(START, tz="UTC")
    end_ts = pd.Timestamp(END, tz="UTC")

    for ticker in TICKERS:
        df = store.load(ticker, "1d")
        if df.empty:
            print(f"  [WARN] no price data for {ticker} — skipping")
            continue
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if len(df) < 100:
            print(f"  [WARN] {ticker}: only {len(df)} bars in window — skipping")
            continue
        prices_dict[ticker] = df
        print(f"  {ticker}: {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")

    return prices_dict


def _compute_bh_equity(
    prices_dict: dict[str, pd.DataFrame],
    initial_cash: float = 10_000.0,
) -> tuple[pd.Series, pd.Series]:
    """Compute buy-and-hold equity curves.

    Returns (sp500_bh, equal_weight_bh).
    - sp500_bh: uses ^GSPC if available in DataStore, otherwise approximates
      with equal-weight of all tickers.
    - equal_weight_bh: equal allocation across all available tickers from day 1.
    """
    store = DataStore(SETTINGS.data.dir)
    start_ts = pd.Timestamp(START, tz="UTC")
    end_ts = pd.Timestamp(END, tz="UTC")

    # Try SP500 index
    sp = store.load("^GSPC", "1d")
    if sp.empty:
        sp = store.load("GSPC", "1d")

    sp500_eq: pd.Series | None = None
    if not sp.empty:
        sp = sp.loc[(sp.index >= start_ts) & (sp.index <= end_ts)]
        if len(sp) > 50:
            sp500_eq = (sp["close"] / sp["close"].iloc[0]) * initial_cash
            sp500_eq.name = "sp500_bh"

    # Equal-weight B&H across available tickers
    ew_returns: list[pd.Series] = []
    for _ticker, prices in prices_dict.items():
        r = prices["close"].pct_change().fillna(0.0)
        ew_returns.append(r)

    if ew_returns:
        # Align all return series to a common index
        combined = pd.concat(ew_returns, axis=1).fillna(0.0)
        ew_daily_ret = combined.mean(axis=1)
        ew_equity = (1 + ew_daily_ret).cumprod() * initial_cash
        ew_equity.name = "ew_bh"
    else:
        ew_equity = pd.Series(dtype=float, name="ew_bh")

    if sp500_eq is None:
        # Use equal-weight as SP500 proxy
        sp500_eq = ew_equity.copy()
        sp500_eq.name = "sp500_bh_proxy"

    return sp500_eq, ew_equity


def _metrics_row(name: str, equity: pd.Series) -> dict:
    """Compute metrics for an equity curve and return a summary dict."""
    if equity.empty or len(equity) < 2:
        return {
            "config": name,
            "sharpe": 0.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "calmar": 0.0,
            "final_equity": 0.0,
            "n_trades": 0,
        }
    m = compute_metrics(equity, [], risk_free_rate=RF, periods_per_year=PPY)
    return {
        "config": name,
        "sharpe": m.sharpe,
        "cagr": m.cagr,
        "max_dd": m.max_drawdown,
        "calmar": m.calmar,
        "final_equity": m.final_equity,
        "n_trades": m.n_trades,
    }


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("PEAD EVALUATION — Post-Earnings Announcement Drift")
    print("=" * 80)
    print(f"Universe: {len(TICKERS)} tickers")
    print(f"Window: {START} → {END}")
    print(f"Risk-free rate: {RF:.4f}")
    print(f"Cost: {COST_BPS} bps entry + {COST_BPS} bps exit")
    print()

    # 1. Load price data
    print("[1/4] Loading price data...")
    store = DataStore(SETTINGS.data.dir)
    prices_dict = _load_prices(store)
    if not prices_dict:
        print("ERROR: no price data available. Run `signals fetch` first.")
        sys.exit(1)
    available_tickers = list(prices_dict.keys())
    print(f"  → {len(available_tickers)} tickers with price data")
    print()

    # 2. Fetch earnings data
    print("[2/4] Fetching earnings data...")
    earnings_df = fetch_earnings_yfinance(
        available_tickers, start=START, end=END
    )
    if earnings_df.empty:
        print("ERROR: no earnings data fetched. API may be rate-limited.")
        print("       Try again later or check yfinance version.")
        sys.exit(1)

    # Report on data quality
    tickers_with_data = earnings_df["ticker"].nunique()
    events_total = len(earnings_df)
    events_with_surprise = earnings_df["surprise_pct"].notna().sum()
    print(f"  → {tickers_with_data} tickers with earnings data")
    print(f"  → {events_total} total earnings events")
    print(f"  → {events_with_surprise} events with surprise data")

    # Filter to only tickers that have both earnings and price data
    valid_tickers = set(available_tickers) & set(earnings_df["ticker"].unique())
    earnings_df = earnings_df[earnings_df["ticker"].isin(valid_tickers)]
    prices_dict = {k: v for k, v in prices_dict.items() if k in valid_tickers}
    print(f"  → {len(valid_tickers)} tickers with BOTH earnings + price data")
    print()

    # 3. Compute benchmarks
    print("[3/4] Computing benchmarks...")
    sp500_eq, ew_eq = _compute_bh_equity(prices_dict)
    print()

    # 4. Run PEAD strategy across parameter grid
    print("[4/4] Running PEAD strategy grid...")
    results: list[dict] = []
    trade_stats: list[dict] = []

    # Benchmarks
    if not sp500_eq.empty:
        results.append(_metrics_row("SP500 B&H", sp500_eq))
    if not ew_eq.empty:
        results.append(_metrics_row(f"EW-{len(valid_tickers)} B&H", ew_eq))

    for threshold in SURPRISE_THRESHOLDS:
        for hold in HOLD_DAYS_GRID:
            config_name = f"PEAD_t{threshold:.0f}_h{hold}"
            print(f"  Running {config_name}...")

            strategy = PEADStrategy(
                surprise_threshold_pct=threshold,
                hold_days=hold,
                max_positions=5,
                cost_bps=COST_BPS,
            )

            # Generate trades for summary
            trades_df = strategy.generate_trades(earnings_df, prices_dict)
            stats = summarize_trades(trades_df)

            # Run backtest
            equity = strategy.backtest(
                earnings_df,
                prices_dict,
                start=START,
                end=END,
                initial_cash=10_000.0,
            )

            row = _metrics_row(config_name, equity)
            row["n_pead_trades"] = stats.n_trades
            row["win_rate"] = stats.win_rate
            row["avg_net_return"] = stats.avg_net_return
            row["median_net_return"] = stats.median_net_return
            results.append(row)

            trade_stats.append({
                "config": config_name,
                "threshold": threshold,
                "hold_days": hold,
                **{k: getattr(stats, k) for k in stats.__dataclass_fields__},
            })

    elapsed = time.time() - t0

    # Output
    results_df = pd.DataFrame(results)
    print()
    print("=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)
    print(
        f"  {'config':<25}{'Sharpe':>10}{'CAGR':>10}{'MaxDD':>10}"
        f"{'Calmar':>10}{'Final$':>12}{'#trades':>10}{'WinRate':>10}"
    )
    print("  " + "-" * 95)
    for _, r in results_df.iterrows():
        wr = r.get("win_rate", "")
        wr_str = f"{wr:.1%}" if isinstance(wr, float) else ""
        nt = r.get("n_pead_trades", r.get("n_trades", ""))
        print(
            f"  {r['config']:<25}{r['sharpe']:>+10.3f}{r['cagr']:>+9.1%}"
            f"{r['max_dd']:>+10.1%}{r['calmar']:>+10.2f}"
            f"{r['final_equity']:>12,.0f}{str(nt):>10}{wr_str:>10}"
        )

    # Trade-level stats
    if trade_stats:
        print()
        print("=" * 100)
        print("TRADE-LEVEL STATS")
        print("=" * 100)
        tdf = pd.DataFrame(trade_stats)
        print(
            f"  {'config':<25}{'#trades':>10}{'winners':>10}{'losers':>10}"
            f"{'win%':>10}{'avg_net':>12}{'med_net':>12}"
        )
        print("  " + "-" * 85)
        for _, r in tdf.iterrows():
            print(
                f"  {r['config']:<25}{r['n_trades']:>10}"
                f"{r['n_winners']:>10}{r['n_losers']:>10}"
                f"{r['win_rate']:>9.1%}{r['avg_net_return']:>+11.2%}"
                f"{r['median_net_return']:>+11.2%}"
            )

    print(f"\n[elapsed] {elapsed:.1f}s")

    # Save results
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "pead_eval.parquet"
    results_df.to_parquet(out_parquet)
    print(f"[wrote] {out_parquet}")

    # Markdown report
    out_md = Path(__file__).parent / "PEAD_RESULTS.md"
    md_lines = [
        "# PEAD Evaluation Results",
        "",
        "**Post-Earnings Announcement Drift** — long-only strategy on 20 "
        "major US equities.",
        "",
        f"- **Window**: {START} to {END}",
        f"- **Tickers with data**: {len(valid_tickers)}/{len(TICKERS)}",
        f"- **Earnings events**: {events_with_surprise} with surprise data",
        f"- **Risk-free rate**: {RF:.4f} (historical USD 3M T-bill, 2018-2024)",
        f"- **Cost**: {COST_BPS} bps entry + {COST_BPS} bps exit",
        "- **Max positions**: 5 concurrent",
        "",
        "## Performance Comparison",
        "",
        "| Config | Sharpe | CAGR | Max DD | Calmar | Final$ | #Trades | Win% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in results_df.iterrows():
        wr = r.get("win_rate", "")
        wr_str = f"{wr:.1%}" if isinstance(wr, float) else "n/a"
        nt = r.get("n_pead_trades", r.get("n_trades", "n/a"))
        md_lines.append(
            f"| `{r['config']}` | {r['sharpe']:+.3f} | "
            f"{r['cagr']:+.1%} | {r['max_dd']:+.1%} | "
            f"{r['calmar']:+.2f} | {r['final_equity']:,.0f} | "
            f"{nt} | {wr_str} |"
        )

    if trade_stats:
        md_lines += [
            "",
            "## Trade-Level Statistics",
            "",
            "| Config | Trades | Winners | Losers | Win% | Avg Net | Med Net |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in pd.DataFrame(trade_stats).iterrows():
            md_lines.append(
                f"| `{r['config']}` | {r['n_trades']} | "
                f"{r['n_winners']} | {r['n_losers']} | "
                f"{r['win_rate']:.1%} | {r['avg_net_return']:+.2%} | "
                f"{r['median_net_return']:+.2%} |"
            )

    md_lines += [
        "",
        "## Interpretation",
        "",
        "PEAD exploits the market's systematic underreaction to earnings "
        "surprises. Unlike price/vol pattern models, this uses FUNDAMENTAL "
        "information. The academic literature shows the drift persists for "
        "60+ trading days after the announcement.",
        "",
        "Key questions:",
        "- Does PEAD beat equal-weight B&H on risk-adjusted basis (Sharpe)?",
        "- Does lower surprise threshold (more trades) help or hurt?",
        "- Is 60-day hold optimal, or does 30/90 work better?",
        "",
    ]
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
