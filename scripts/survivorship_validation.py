"""Comprehensive survivorship bias validation suite.

Four independent tests to determine whether the full-universe Sharpe
improvement (vs S&P-only) is driven by survivorship bias.

  TEST 1: Time-decay test — if bias inflates results, the effect should
          be STRONGER in earlier years (more dead stocks).
  TEST 2: Monte Carlo delisting simulation — randomly kill stocks and
          apply -50% terminal returns.
  TEST 3: Former-S&P-only universe — restrict to tickers that appeared
          in ANY historical S&P 500 snapshot.
  TEST 4: Sector concentration analysis — check if the strategy loads
          up on survivorship-heavy sectors.

Usage:
    python -u scripts/survivorship_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.backtest.bias_free import (
    BacktestResult,
    BiasFreData,
    clear_cache,
    default_acceleration_score,
    load_bias_free_data,
    run_bias_free_backtest,
    _get_constituents,
    _get_dead_tickers,
)
from signals.backtest.metrics import compute_metrics


P = lambda *a, **kw: print(*a, **kw, flush=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _slice_data(data: BiasFreData, start: str, end: str) -> BiasFreData:
    """Slice a loaded BiasFreData to a date range without reloading parquets."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    mask = [(start_ts <= d <= end_ts) for d in data.trading_dates]
    row_indices = [i for i, m in enumerate(mask) if m]

    if not row_indices:
        raise ValueError(f"No trading dates in range {start} to {end}")

    new_dates = [data.trading_dates[i] for i in row_indices]
    new_mat = data.close_mat[row_indices, :]

    return BiasFreData(
        close_mat=new_mat,
        tickers=data.tickers,
        ticker_to_idx=data.ticker_to_idx,
        trading_dates=new_dates,
        constituent_map=data.constituent_map,
        constituent_dates=data.constituent_dates,
        sectors=data.sectors,
    )


def _all_historical_sp500_tickers(data: BiasFreData) -> set[str]:
    """Return every ticker that appeared in ANY historical constituent snapshot."""
    tickers: set[str] = set()
    for ticker_list in data.constituent_map.values():
        tickers.update(ticker_list)
    return tickers


def _print_header(title: str) -> None:
    width = 74
    P()
    P("=" * width)
    P(f"  {title}")
    P("=" * width)


def _print_result_row(label: str, r: BacktestResult, extra: str = "") -> None:
    P(
        f"  {label:<42s} Sharpe={r.sharpe:+.3f}  CAGR={r.cagr:+.1%}  "
        f"MDD={r.max_drawdown:+.1%}  Trades={r.n_trades}"
        + (f"  {extra}" if extra else "")
    )


# ── TEST 1: Time-decay test ────────────────────────────────────────────────

def test_time_decay(data: BiasFreData) -> dict:
    """Compare full-universe vs S&P-only Sharpe across 5 non-overlapping periods."""
    _print_header("TEST 1: TIME-DECAY TEST")
    P("  If survivorship bias drives the improvement, the full-universe")
    P("  advantage should be LARGER in earlier periods (more dead stocks).")
    P()

    periods = [
        ("2000-2005", "2000-01-01", "2005-12-31"),
        ("2005-2010", "2005-01-01", "2010-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2015-2020", "2015-01-01", "2020-12-31"),
        ("2020-2026", "2020-01-01", "2026-04-13"),
    ]

    results = []
    for label, start, end in periods:
        P(f"  Running {label}...", end=" ")
        try:
            pdata = _slice_data(data, start, end)
        except Exception as e:
            P(f"SKIP ({e})")
            continue

        r_sp = run_bias_free_backtest(pdata, use_full_universe=False)
        r_fu = run_bias_free_backtest(pdata, use_full_universe=True)

        delta = r_fu.sharpe - r_sp.sharpe
        results.append({
            "period": label,
            "sp_sharpe": r_sp.sharpe,
            "fu_sharpe": r_fu.sharpe,
            "delta": delta,
            "sp_cagr": r_sp.cagr,
            "fu_cagr": r_fu.cagr,
        })
        P(
            f"S&P={r_sp.sharpe:+.3f}  Full={r_fu.sharpe:+.3f}  "
            f"Delta={delta:+.3f}"
        )

    # Summary
    if results:
        deltas = [r["delta"] for r in results]
        P()
        P(f"  Delta range: {min(deltas):+.3f} to {max(deltas):+.3f}")
        P(f"  Delta mean:  {np.mean(deltas):+.3f}  std: {np.std(deltas):.3f}")

        if len(deltas) >= 3:
            corr = np.corrcoef(range(len(deltas)), deltas)[0, 1]
            P(f"  Delta-vs-time correlation: {corr:+.3f}")
            if corr < -0.5:
                P("  ** WARNING: Delta decreases over time -> survivorship bias signal")
            elif abs(corr) < 0.5:
                P("  PASS: No systematic time trend in delta -> bias not driving improvement")
            else:
                P("  NOTE: Delta increases over time -> opposite of survivorship bias")

    return {"test": "time_decay", "results": results}


# ── TEST 2: Monte Carlo delisting simulation ───────────────────────────────

def test_monte_carlo_delisting(
    data: BiasFreData,
    n_seeds: int = 10,
    kill_rate: float = 0.03,
) -> dict:
    """Randomly kill stocks during backtest to simulate missing delistings."""
    _print_header("TEST 2: MONTE CARLO DELISTING SIMULATION")
    P(f"  Kill rate: {kill_rate:.0%} of stocks per year")
    P(f"  Terminal return: -50% (academic delisting average)")
    P(f"  Seeds: {n_seeds}")
    P()

    base_mat = data.close_mat.copy()
    n_dates, n_tickers = base_mat.shape
    dates = data.trading_dates

    # Baselines
    P("  Running baselines...")
    r_baseline = run_bias_free_backtest(data, use_full_universe=True)
    P(f"  Baseline (unmodified):  Sharpe={r_baseline.sharpe:+.3f}  CAGR={r_baseline.cagr:+.1%}")

    r_sp = run_bias_free_backtest(data, use_full_universe=False)
    P(f"  S&P-only baseline:     Sharpe={r_sp.sharpe:+.3f}  CAGR={r_sp.cagr:+.1%}")
    P()

    # Group rows by year
    year_ranges: dict[int, tuple[int, int]] = {}
    for i, dt in enumerate(dates):
        y = dt.year
        if y not in year_ranges:
            year_ranges[y] = (i, i)
        else:
            year_ranges[y] = (year_ranges[y][0], i)

    sim_sharpes = []
    sim_cagrs = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 42)
        sim_mat = base_mat.copy()

        n_killed_total = 0
        for year, (row_start, row_end) in sorted(year_ranges.items()):
            # Find columns with valid data in this year
            year_slice = sim_mat[row_start:row_end + 1, :]
            valid_cols = np.where(~np.all(np.isnan(year_slice), axis=0))[0]

            n_kill = max(1, int(len(valid_cols) * kill_rate))
            kill_cols = rng.choice(valid_cols, size=min(n_kill, len(valid_cols)), replace=False)

            for c in kill_cols:
                valid_rows = np.where(~np.isnan(sim_mat[row_start:row_end + 1, c]))[0]
                if len(valid_rows) == 0:
                    continue
                valid_rows += row_start

                delist_row = int(rng.choice(valid_rows))

                # Apply -50% terminal return at delist date
                last_price = sim_mat[delist_row, c]
                if not np.isnan(last_price) and last_price > 0:
                    sim_mat[delist_row, c] = last_price * 0.50

                # Set everything after delist to NaN
                sim_mat[delist_row + 1:, c] = np.nan
                n_killed_total += 1

        sim_data = BiasFreData(
            close_mat=sim_mat,
            tickers=data.tickers,
            ticker_to_idx=data.ticker_to_idx,
            trading_dates=data.trading_dates,
            constituent_map=data.constituent_map,
            constituent_dates=data.constituent_dates,
            sectors=data.sectors,
        )

        r = run_bias_free_backtest(sim_data, use_full_universe=True)
        sim_sharpes.append(r.sharpe)
        sim_cagrs.append(r.cagr)
        P(
            f"  Seed {seed:2d}: Sharpe={r.sharpe:+.3f}  CAGR={r.cagr:+.1%}  "
            f"(killed {n_killed_total} stock-years)"
        )

    mean_sharpe = np.mean(sim_sharpes)
    std_sharpe = np.std(sim_sharpes)
    mean_cagr = np.mean(sim_cagrs)
    P()
    P(f"  Simulated Sharpe: {mean_sharpe:+.3f} +/- {std_sharpe:.3f}")
    P(f"  Simulated CAGR:   {mean_cagr:+.1%}")
    P(f"  Baseline Sharpe:  {r_baseline.sharpe:+.3f}")
    P(f"  S&P-only Sharpe:  {r_sp.sharpe:+.3f}")
    haircut = r_baseline.sharpe - mean_sharpe
    P(f"  Haircut from simulated delistings: {haircut:+.3f}")
    if mean_sharpe > r_sp.sharpe:
        P("  PASS: Even with simulated delistings, full-universe beats S&P-only")
    else:
        P("  ** WARNING: Simulated delistings wipe out full-universe advantage")

    return {
        "test": "monte_carlo_delisting",
        "baseline_sharpe": r_baseline.sharpe,
        "sp_sharpe": r_sp.sharpe,
        "sim_mean_sharpe": mean_sharpe,
        "sim_std_sharpe": std_sharpe,
        "sim_sharpes": sim_sharpes,
    }


# ── TEST 3: Former-S&P-only universe ───────────────────────────────────────

def test_former_sp_only(data: BiasFreData) -> dict:
    """Restrict universe to tickers that appeared in ANY historical S&P snapshot."""
    _print_header("TEST 3: FORMER-S&P-ONLY UNIVERSE")
    P("  Only allow tickers that appear in ANY historical S&P 500 snapshot.")
    P("  This excludes never-in-S&P stocks (highest survivorship bias risk).")
    P()

    ever_sp500 = _all_historical_sp500_tickers(data)
    all_tickers = set(data.tickers)
    never_sp500 = all_tickers - ever_sp500
    in_sp500 = all_tickers & ever_sp500

    P(f"  Total tickers in data:     {len(all_tickers)}")
    P(f"  Ever in S&P 500:           {len(in_sp500)}")
    P(f"  Never in S&P 500:          {len(never_sp500)}")
    P()

    # A: S&P point-in-time (baseline)
    P("  Running A: S&P point-in-time...")
    r_sp = run_bias_free_backtest(data, use_full_universe=False)
    _print_result_row("A: S&P point-in-time", r_sp)

    # B: Full universe (all tickers)
    P("  Running B: Full universe...")
    r_full = run_bias_free_backtest(data, use_full_universe=True)
    _print_result_row("B: Full universe", r_full)

    # C: Former-S&P only — NaN out non-former-S&P tickers
    P("  Running C: Former-S&P only...")
    former_sp_cols = set()
    for t in in_sp500:
        if t in data.ticker_to_idx:
            former_sp_cols.add(data.ticker_to_idx[t])

    mat_former = data.close_mat.copy()
    for col_idx in range(len(data.tickers)):
        if col_idx not in former_sp_cols:
            mat_former[:, col_idx] = np.nan

    data_former = BiasFreData(
        close_mat=mat_former,
        tickers=data.tickers,
        ticker_to_idx=data.ticker_to_idx,
        trading_dates=data.trading_dates,
        constituent_map=data.constituent_map,
        constituent_dates=data.constituent_dates,
        sectors=data.sectors,
    )
    r_former = run_bias_free_backtest(data_former, use_full_universe=True)
    _print_result_row("C: Former-S&P only (full-universe mode)", r_former)

    # D: Never-S&P only (for contrast)
    P("  Running D: Never-S&P only...")
    mat_never = data.close_mat.copy()
    for col_idx in range(len(data.tickers)):
        t = data.tickers[col_idx]
        if t in ever_sp500:
            mat_never[:, col_idx] = np.nan

    data_never = BiasFreData(
        close_mat=mat_never,
        tickers=data.tickers,
        ticker_to_idx=data.ticker_to_idx,
        trading_dates=data.trading_dates,
        constituent_map=data.constituent_map,
        constituent_dates=data.constituent_dates,
        sectors=data.sectors,
    )
    r_never = run_bias_free_backtest(data_never, use_full_universe=True)
    _print_result_row("D: Never-S&P only (full-universe mode)", r_never)

    P()
    delta_full = r_full.sharpe - r_sp.sharpe
    delta_former = r_former.sharpe - r_sp.sharpe
    if delta_full != 0:
        pct_from_former = delta_former / delta_full * 100
    else:
        pct_from_former = 0

    P(f"  Full-universe improvement over S&P:   {delta_full:+.3f}")
    P(f"  Former-S&P improvement over S&P:      {delta_former:+.3f}")
    P(f"  % of improvement from former-S&P:     {pct_from_former:.0f}%")
    P(f"  % from never-S&P:                     {100 - pct_from_former:.0f}%")

    if pct_from_former > 70:
        P("  PASS: Most improvement comes from ex-S&P stocks (low bias risk)")
    elif pct_from_former > 40:
        P("  MIXED: Improvement splits between former and never-S&P")
    else:
        P("  ** WARNING: Most improvement comes from never-S&P stocks (high bias risk)")

    return {
        "test": "former_sp_only",
        "sp_sharpe": r_sp.sharpe,
        "full_sharpe": r_full.sharpe,
        "former_sp_sharpe": r_former.sharpe,
        "never_sp_sharpe": r_never.sharpe,
        "pct_from_former": pct_from_former,
    }


# ── TEST 4: Sector concentration analysis ──────────────────────────────────

def test_sector_concentration(data: BiasFreData) -> dict:
    """Analyze sector distribution of stocks selected by the full-universe strategy."""
    _print_header("TEST 4: SECTOR CONCENTRATION ANALYSIS")
    P("  Tracking which sectors the strategy selects over time.")
    P()

    mat = data.close_mat
    n_dates = len(data.trading_dates)

    short, long_ = 63, 252
    hold_days = 105
    n_long = 15
    max_per_sector = 2
    min_short_return = 0.10
    max_long_return = 1.50
    rebalance_freq = 21

    dead_tickers = _get_dead_tickers(data)
    alive_cols = [
        data.ticker_to_idx[t]
        for t in data.tickers
        if t not in dead_tickers
    ]

    def score_fn(cm, r, c, s=short, lg=long_):
        return default_acceleration_score(cm, r, c, s, lg, min_short_return, max_long_return)

    holdings: dict[int, dict] = {}
    cash = 100_000.0
    cost_rate = 10.0 * 1e-4
    bars_since_rebal = rebalance_freq

    rebalance_picks: list[dict] = []

    for row in range(n_dates):
        # Fixed-hold exits
        for col in list(holdings):
            if (row - holdings[col]["entry_row"]) >= hold_days:
                p = mat[row, col]
                if not np.isnan(p):
                    cash += holdings[col]["sh"] * p * (1 - cost_rate)
                del holdings[col]

        # Deploy idle cash
        if holdings and cash > 100:
            per = cash / len(holdings)
            for col in holdings:
                p = mat[row, col]
                if not np.isnan(p) and p > 0:
                    holdings[col]["sh"] += per / p
                    cash -= per

        # Rebalance
        bars_since_rebal += 1
        if bars_since_rebal >= rebalance_freq and row >= long_:
            eligible_cols = [c for c in alive_cols if not np.isnan(mat[row, c])]

            candidates = []
            for col in eligible_cols:
                if col in holdings:
                    continue
                score = score_fn(mat, row, col, short, long_)
                if score is None:
                    continue
                ticker = data.tickers[col]
                sector = data.sectors.get(ticker, "Unknown")
                candidates.append((col, score, sector))

            candidates.sort(key=lambda x: x[1], reverse=True)

            n_slots = n_long - len(holdings)
            if n_slots > 0 and candidates:
                sector_count: dict[str, int] = {}
                for h in holdings.values():
                    sector_count[h["sec"]] = sector_count.get(h["sec"], 0) + 1

                selected = []
                for col, _score, sector in candidates:
                    if len(selected) >= n_slots:
                        break
                    if sector_count.get(sector, 0) >= max_per_sector:
                        continue
                    selected.append((col, sector))
                    sector_count[sector] = sector_count.get(sector, 0) + 1

                if selected:
                    date = data.trading_dates[row]
                    for col, sector in selected:
                        rebalance_picks.append({
                            "date": date,
                            "year": date.year,
                            "ticker": data.tickers[col],
                            "sector": sector,
                        })

                equity = cash + sum(
                    h["sh"] * mat[row, c] for c, h in holdings.items()
                    if not np.isnan(mat[row, c])
                )
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
                                "ep": p, "sh": per_pos / p,
                                "entry_row": row, "sec": sector,
                            }
                            cash -= cost
                    if holdings and cash > 100:
                        per = cash / len(holdings)
                        for col in holdings:
                            p = mat[row, col]
                            if not np.isnan(p) and p > 0:
                                holdings[col]["sh"] += per / p
                                cash -= per

            bars_since_rebal = 0

    picks_df = pd.DataFrame(rebalance_picks)
    if picks_df.empty:
        P("  No picks recorded!")
        return {"test": "sector_concentration", "results": {}}

    P(f"  Total stock selections: {len(picks_df)}")
    P(f"  Unique tickers selected: {picks_df['ticker'].nunique()}")
    P(f"  Date range: {picks_df['date'].min().date()} to {picks_df['date'].max().date()}")
    P()

    # Overall sector distribution
    sector_counts = picks_df["sector"].value_counts()
    total = len(picks_df)
    P("  OVERALL SECTOR DISTRIBUTION:")
    P(f"  {'Sector':<35s} {'Picks':>6s} {'%':>6s}")
    P(f"  {'-'*50}")
    for sector, count in sector_counts.items():
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        P(f"  {sector:<35s} {count:>6d} {pct:>5.1f}% {bar}")

    # Herfindahl index
    shares = (sector_counts / total).values
    hhi = float(np.sum(shares ** 2))
    n_sectors = len(sector_counts)
    hhi_uniform = 1.0 / n_sectors if n_sectors > 0 else 1.0
    P(f"\n  Herfindahl-Hirschman Index: {hhi:.3f}")
    P(f"  (Uniform across {n_sectors} sectors would be {hhi_uniform:.3f})")
    if hhi > 0.25:
        P("  ** WARNING: High sector concentration — possible survivorship bias")
    elif hhi > 0.15:
        P("  NOTE: Moderate sector concentration")
    else:
        P("  PASS: Sector distribution is well-diversified")

    # Sector distribution by era
    eras = [
        ("2000-2005", 2000, 2005),
        ("2005-2010", 2005, 2010),
        ("2010-2015", 2010, 2015),
        ("2015-2020", 2015, 2020),
        ("2020-2026", 2020, 2026),
    ]
    P("\n  SECTOR DISTRIBUTION BY ERA (top 3 sectors per era):")
    P(f"  {'Era':<12s} {'#1':<22s} {'#2':<22s} {'#3':<22s}")
    P(f"  {'-'*78}")
    for era_label, y_start, y_end in eras:
        era_df = picks_df[(picks_df["year"] >= y_start) & (picks_df["year"] < y_end)]
        if era_df.empty:
            P(f"  {era_label:<12s} (no data)")
            continue
        era_sectors = era_df["sector"].value_counts()
        era_total = len(era_df)
        top3 = []
        for s, c in era_sectors.head(3).items():
            top3.append(f"{s[:16]} ({c/era_total:.0%})")
        while len(top3) < 3:
            top3.append("-")
        P(f"  {era_label:<12s} {top3[0]:<22s} {top3[1]:<22s} {top3[2]:<22s}")

    max_sector_pct = sector_counts.iloc[0] / total * 100
    max_sector_name = sector_counts.index[0]
    P(f"\n  Largest sector: {max_sector_name} ({max_sector_pct:.1f}%)")
    if max_sector_pct > 40:
        P("  ** WARNING: Single sector >40% — high concentration risk")
    elif max_sector_pct > 25:
        P("  NOTE: Largest sector >25% but sector cap (max_per_sector=2) limits it")
    else:
        P("  PASS: No single sector dominates")

    return {
        "test": "sector_concentration",
        "hhi": hhi,
        "n_sectors": n_sectors,
        "top_sector": max_sector_name,
        "top_sector_pct": max_sector_pct,
        "sector_counts": sector_counts.to_dict(),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    P()
    P("*" * 74)
    P("*  COMPREHENSIVE SURVIVORSHIP BIAS VALIDATION SUITE                     *")
    P("*  Tests whether full-universe Sharpe improvement is from bias           *")
    P("*" * 74)

    # Load data ONCE — all tests will slice or modify copies
    P("\nLoading data (one-time)...")
    clear_cache()
    data = load_bias_free_data()
    P(f"  Loaded: {len(data.trading_dates)} dates, {len(data.tickers)} tickers")
    P(f"  Range: {data.trading_dates[0].date()} to {data.trading_dates[-1].date()}")

    all_results = {}

    # TEST 1: Time-decay
    all_results["time_decay"] = test_time_decay(data)

    # TEST 2: Monte Carlo delisting
    all_results["monte_carlo"] = test_monte_carlo_delisting(data, n_seeds=10)

    # TEST 3: Former-S&P-only
    all_results["former_sp"] = test_former_sp_only(data)

    # TEST 4: Sector concentration
    all_results["sector"] = test_sector_concentration(data)

    # ── FINAL VERDICT ───────────────────────────────────────────────────
    _print_header("FINAL VERDICT")

    concerns = 0
    passes = 0

    t1 = all_results.get("time_decay", {}).get("results", [])
    if t1:
        deltas = [r["delta"] for r in t1]
        corr = np.corrcoef(range(len(deltas)), deltas)[0, 1] if len(deltas) >= 3 else 0
        if corr < -0.5:
            P("  TEST 1 (Time-decay):     ** CONCERN ** — delta decreases over time")
            concerns += 1
        else:
            P(f"  TEST 1 (Time-decay):     PASS (time-correlation={corr:+.2f})")
            passes += 1

    t2 = all_results.get("monte_carlo", {})
    if "sim_mean_sharpe" in t2:
        if t2["sim_mean_sharpe"] > t2["sp_sharpe"]:
            P(
                f"  TEST 2 (Delisting sim):  PASS (simulated={t2['sim_mean_sharpe']:+.3f} "
                f"> S&P-only={t2['sp_sharpe']:+.3f})"
            )
            passes += 1
        else:
            P(
                f"  TEST 2 (Delisting sim):  ** CONCERN ** (simulated={t2['sim_mean_sharpe']:+.3f} "
                f"<= S&P-only={t2['sp_sharpe']:+.3f})"
            )
            concerns += 1

    t3 = all_results.get("former_sp", {})
    if "pct_from_former" in t3:
        pct = t3["pct_from_former"]
        if pct > 70:
            P(f"  TEST 3 (Former-S&P):     PASS ({pct:.0f}% of improvement from ex-S&P stocks)")
            passes += 1
        elif pct > 40:
            P(f"  TEST 3 (Former-S&P):     MIXED ({pct:.0f}% from former-S&P)")
            passes += 0.5
            concerns += 0.5
        else:
            P(f"  TEST 3 (Former-S&P):     ** CONCERN ** (only {pct:.0f}% from former-S&P)")
            concerns += 1

    t4 = all_results.get("sector", {})
    if "hhi" in t4:
        hhi = t4["hhi"]
        if hhi > 0.25:
            P(f"  TEST 4 (Sector conc.):   ** CONCERN ** (HHI={hhi:.3f})")
            concerns += 1
        else:
            P(f"  TEST 4 (Sector conc.):   PASS (HHI={hhi:.3f})")
            passes += 1

    P()
    total = passes + concerns
    if total > 0:
        if concerns == 0:
            P(f"  OVERALL: {int(passes)}/{int(total)} tests PASS — "
              "NO EVIDENCE of survivorship bias driving the improvement")
        elif concerns <= 1:
            P(f"  OVERALL: {concerns:.0f}/{int(total)} concerns — "
              "MINOR survivorship bias signal, results likely robust")
        else:
            P(f"  OVERALL: {concerns:.0f}/{int(total)} concerns — "
              "SIGNIFICANT survivorship bias signal, interpret with caution")
    P()


if __name__ == "__main__":
    main()
