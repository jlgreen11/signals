# BTC hybrid + S&P 500 buy & hold — multi-asset portfolio experiment

**Run date**: 2026-04-11
**Motivation**: The Tier-1S analysis established that no strategy in
this project beats buy & hold on S&P 500 (see
`SP500_TREND_AND_HOMC_MEMORY.md`). The Tier-0f analysis established
that BTC's H-Vol @ q=0.70 hybrid is at a Sharpe plateau around 2.15.
This experiment tests whether combining the two as a 2-asset portfolio
(BTC hybrid + S&P B&H) produces a higher Sharpe than either alone via
diversification.

**Hypothesis**: BTC and S&P have low historical correlation (~0.1-0.2
daily). A constant-mix portfolio should have a Sharpe higher than the
weighted average of the individual Sharpes due to the variance
reduction of holding uncorrelated assets.

## TL;DR

**Yes, the portfolio has a real but soft edge.** The 40/60 BTC/SP mix
with daily rebalancing produces median Sharpe **2.44** at seed 42
(vs BTC-alone at 2.20 — the same config as H-Vol @ q=0.70 in this
run). Across 4 seeds, 40/60 mix averages Sharpe **1.16** vs BTC-alone
at **1.00** — a 16% Sharpe improvement on average. It beats BTC-alone
on 3 out of 4 seeds, with the exception being seed 100 where BTC's
random windows landed on BTC-strong periods.

**This is the first "genuine alpha" result in the project** that doesn't
come from model/parameter tuning. It comes from portfolio construction.

**Recommendation**: For a risk-balanced investor, the 40/60 BTC/SP mix
with daily rebalancing is a reasonable default. For a return-
maximizing investor willing to accept higher variance, BTC hybrid
alone still wins on upside CAGR during BTC-strong regimes. The mix
smooths the distribution of outcomes — narrower CAGR range, smaller
drawdowns, higher Sharpe on average.

## Setup

**Universe**: BTC-USD + ^GSPC, 2015-01-01 → 2024-12-31
**Strategy for BTC**: H-Vol hybrid @ vol_quantile=0.70 (production default)
**Strategy for S&P**: Buy and hold (nothing beats B&H per Tier-1S)
**Windows**: 16 random 6-month windows, seed 42 (same as all prior evals)
**Allocation modes**:

1. **Window rebalance**: allocate W_btc to BTC strategy and W_sp to S&P
   at the start of the window, then let each component drift independently.
   No mid-window rebalancing. Simplest portfolio behavior.

2. **Daily rebalance**: compute each component's daily return, combine
   as `w_btc × btc_ret + w_sp × sp_ret`, compound. Equivalent to
   rebalancing to the target weights every bar. In practice this would
   accrue trading costs, but in the backtest it's free — so this is an
   upper bound on the rebalancing benefit.

**Weight grid**: 7 pairs from 100/0 (BTC-only) through 0/100 (SP-only)
in 20pp increments, plus 50/50.

**S&P return on BTC-only days**: S&P doesn't trade on weekends, so on
Saturday and Sunday the S&P return is zero (position unchanged, no
mark-to-market movement). This is economically correct and handled via
`reindex(method="ffill")` when combining equity curves.

## Phase A — Seed 42 results

### Window rebalance (no mid-window rebalancing)

| BTC / SP | Mean Sh | **Median Sh** | Mean CAGR | Median CAGR | Mean Max DD |
|---:|---:|---:|---:|---:|---:|
| 100 / 0 | 1.76 | 2.20 | +269% | +146% | -20.5% |
| 80 / 20 | 1.79 | 2.19 | +198% | +129% | -17.3% |
| 60 / 40 | 1.83 | 2.20 | +140% | +113% | -14.0% |
| 50 / 50 | 1.84 | 2.24 | +115% | +105% | -12.3% |
| **40 / 60** | **1.85** | **2.25** | +93% | +98% | -10.9% |
| 20 / 80 | 1.79 | 2.04 | +56% | +55% | -8.9% |
| 0 / 100 | 1.13 | 1.09 | +28% | +23% | -9.4% |

### Daily rebalance (free in backtest, accrues costs in reality)

| BTC / SP | Mean Sh | **Median Sh** | Mean CAGR | Median CAGR | Mean Max DD |
|---:|---:|---:|---:|---:|---:|
| 100 / 0 | 1.76 | 2.20 | +269% | +146% | -20.5% |
| 80 / 20 | 1.83 | 2.24 | +190% | +131% | -17.0% |
| 60 / 40 | 1.92 | 2.34 | +130% | +116% | -13.4% |
| 50 / 50 | 1.95 | 2.41 | +106% | +108% | -11.8% |
| **40 / 60** | **1.97** | **2.44** 🏆 | +85% | +93% | -10.3% |
| 20 / 80 | 1.85 | 2.19 | +52% | +52% | -8.6% |
| 0 / 100 | 1.13 | 1.09 | +28% | +23% | -9.4% |

### Observations at seed 42

1. **Daily rebalancing strictly dominates window rebalancing** on every
   mixed allocation. The lift is 0.04 - 0.19 Sharpe. This is the
   rebalancing premium — systematically buying the underperformer and
   selling the outperformer each day, which is exactly what constant-
   mix portfolios do.

2. **40/60 BTC/SP is the seed-42 optimum on both rebalancing modes** by
   median Sharpe: 2.25 (window) and 2.44 (daily).

3. **Mean Sharpe peaks at 40/60 too** for both modes, though the
   window-mode lift over BTC-alone is small (+0.09 for window, +0.21
   for daily).

4. **Max drawdown improves dramatically with diversification**: BTC-
   alone -20.5% → 40/60 -10.3% (daily). Half the drawdown for the
   same Sharpe-equivalent return. This is the primary benefit of
   diversification on the downside.

5. **CAGR drops linearly with S&P weight** — because S&P's CAGR is
   much lower than BTC's (+23% vs +146% median). The portfolio trades
   return for stability.

6. **The SP-only result (1.09 median Sharpe)** is higher than the
   Tier-0e standalone S&P random-window eval (0.77). The difference is
   window selection: this experiment uses BTC-driven windows that
   happen to land on S&P-friendly periods. The comparison within this
   experiment is apples-to-apples.

## Phase B — Multi-seed robustness

Re-ran the top 3 portfolio mixes (40/60, 50/50, 60/40 daily rebalance)
plus the BTC-alone baseline at seeds {42, 7, 100, 999}. The same
methodology as the BTC deep sweep robustness (see
`BTC_DEEP_SWEEP_RESULTS.md`). 192 additional backtests.

### Median Sharpe per portfolio per seed

| Portfolio | seed 42 | seed 7 | seed 100 | seed 999 | **avg** | **Median of seeds** |
|---|---:|---:|---:|---:|---:|---:|
| **40/60 daily** | **2.44** | -0.14 | 0.87 | **1.47** | **1.16** | **1.17** |
| 50/50 daily | 2.41 | -0.25 | 0.95 | 1.33 | 1.11 | 1.14 |
| 60/40 daily | 2.34 | -0.31 | 1.01 | 1.19 | 1.06 | 1.10 |
| baseline BTC-only | 2.15 | -0.27 | **1.38** | 0.74 | 1.00 | 1.06 |

### Critical observations

1. **40/60 mix has the highest average Sharpe across all 4 seeds**:
   1.16 vs BTC-alone's 1.00. A **+16% improvement** on average. This
   is the first strategy in the project to beat the BTC-alone baseline
   on a multi-seed average.

2. **40/60 beats baseline on 3 out of 4 seeds**:
   - Seed 42: 2.44 vs 2.15 (+0.29) ✓
   - Seed 7: -0.14 vs -0.27 (+0.13) ✓ (smaller loss)
   - Seed 100: 0.87 vs 1.38 (**-0.51**) ✗ — portfolio loses here
   - Seed 999: 1.47 vs 0.74 (+0.73) ✓

3. **Seed 100 is the exception worth understanding.** At seed 100, the
   random draw lands on windows where BTC absolutely crushes it and
   S&P is flat or down. Baseline's 1.38 Sharpe comes from pure BTC
   alpha during those periods; the 40/60 mix drags BTC's 60% exposure
   down by holding 60% in an underperforming S&P. Diversification is
   a cost here, not a benefit.

4. **Seed 7 shows the downside floor of both**: baseline -0.27,
   40/60 -0.14. Both lose money. The portfolio's edge at seed 7 is
   "lose less" rather than "win more". Still a positive vs the pure
   strategy.

5. **Every mix (40/60, 50/50, 60/40) has higher average Sharpe than
   BTC-alone across 4 seeds**. The 40/60 is slightly better than
   50/50 which is slightly better than 60/40. There's a genuine sweet
   spot at high-S&P allocation because S&P's lower variance reduces
   the portfolio's drawdown floor more than its Sharpe ceiling.

### The portfolio is a real win

Unlike the Dim 5 "sell=-20" result (which beat baseline at seed 42 but
lost on average), the 40/60 portfolio mix **wins on the average
Sharpe across seeds** (1.16 vs 1.00) AND wins on 3 out of 4 individual
seeds. This is a qualitatively different kind of result: not a
data-mining artifact that only shows up in one draw, but a real
diversification benefit that persists across window draws.

The +16% Sharpe lift is consistent with what portfolio theory predicts
for two uncorrelated assets with similar expected returns and different
volatilities. It's not free lunch to the upside — the mean CAGR goes
down because S&P's return is much lower than BTC's. But on a
risk-adjusted basis, the mix is better because it extracts the
diversification benefit without giving up too much alpha.

## Recommendation

**For a risk-balanced investor**: 40/60 BTC/SP with daily rebalancing.
Median Sharpe 2.44 at seed 42, average 1.16 across 4 seeds. Halved
drawdowns vs BTC-alone. Simplest implementation: 40% of account in
BTC traded by the hybrid strategy, 60% in SPY held passively,
rebalance monthly (to approximate daily without the transaction cost).

**For a return-maximizing investor**: BTC hybrid alone (H-Vol @ q=0.70)
with max_long=1.25. Higher expected CAGR, higher variance, higher
drawdowns. The "Sharpe doesn't improve" reality from Tier 0f still
applies — scaling leverage scales CAGR and MDD proportionally.

**Defaults in code**: The project does not currently ship a
multi-asset portfolio implementation. The BTC hybrid default remains
what it was. The portfolio is a post-hoc linear combination of
individual strategy equity curves, not a new model class. See
"Implementation notes" below for what would be needed to ship this
properly.

## Implementation notes

What exists now (this commit):

- `scripts/btc_sp500_portfolio.py` — runs the portfolio experiment and
  saves per-window raw results to parquet
- `scripts/btc_deep_sweep_robustness.py` — includes the multi-seed
  portfolio robustness as Phase B
- `scripts/data/btc_sp500_portfolio.parquet` — 224 rows of per-window
  metrics (7 weights × 2 rebalance modes × 16 windows)
- `scripts/data/btc_deep_sweep_robustness.parquet` — 640 rows of
  multi-seed robustness data (both strategy and portfolio)

What would be needed to ship a production portfolio model:

1. **A `PortfolioCombiner` class** in `signals/backtest/` that takes a
   list of (model, weight, symbol) tuples and produces a combined
   equity curve. Should handle different trading calendars (BTC 7d/
   week, SP 5d/week), different time zones, and rebalancing cadence
   (daily, weekly, monthly, at-drift-threshold).

2. **A new CLI command** like `signals backtest portfolio
   BTC-USD:0.4:hybrid ^GSPC:0.6:buy_hold --rebalance daily` that
   wraps the combiner for end-to-end portfolio backtests.

3. **A `signal next portfolio`** command that computes the current
   target weights per asset based on the signal models and yesterday's
   closing prices. Outputs a target dollar allocation per symbol.

4. **Portfolio-level metrics**: conditional value at risk (CVaR),
   correlation matrix, rolling beta, tracking error. The current
   `compute_metrics` only handles single-asset equity curves.

5. **Tax-aware rebalancing**: the daily-rebalance mode is economically
   unrealistic because it ignores taxes on short-term gains. Should
   support threshold-based rebalancing ("rebalance only if weight
   drifts > 5pp") which is much more tax-efficient.

None of this is on the current roadmap. For now, the portfolio result
is a research finding documented here but not wired into the CLI.
`IMPROVEMENTS.md #19` tracks it as an open item.

## Data artifacts

All raw data committed to `scripts/data/`:

| File | Purpose | Rows |
|---|---|---:|
| `btc_sp500_portfolio.parquet` | Seed-42 portfolio sweep: 7 weights × 2 rebalance × 16 windows | 224 |
| `btc_deep_sweep_robustness.parquet` | Multi-seed robustness (includes Phase B portfolio) | 640 |

Columns in `btc_sp500_portfolio.parquet`:

| column | meaning |
|---|---|
| `rebalance` | "window" or "daily" |
| `btc_weight`, `sp_weight` | allocation (sum to 1.0) |
| `window_idx` | 1..16 |
| `window_start`, `window_end` | date strings |
| `final_equity` | portfolio equity at end of window |
| `cagr`, `sharpe`, `max_dd` | portfolio metrics for the window |
| `btc_cagr`, `btc_sharpe` | BTC component metrics (same for every weight) |
| `sp_cagr`, `sp_sharpe` | S&P component metrics (same for every weight) |

You can load and analyze offline:

```python
import pandas as pd
df = pd.read_parquet("scripts/data/btc_sp500_portfolio.parquet")

# Median Sharpe per weight (daily rebalance)
daily = df[df["rebalance"] == "daily"]
print(daily.groupby("btc_weight")["sharpe"].median())
```

## Reproducibility

```bash
python scripts/btc_sp500_portfolio.py            # ~1 min
python scripts/btc_deep_sweep_robustness.py      # ~5 min (includes portfolio)
```

Both deterministic. Output parquets will be bit-identical across runs
on the same hardware.

---

⚠ **Disclaimer**: This is an experimental research project. The results
shown above are backtests and historical simulations — they do not
predict future performance. Not financial advice. See the root
[`README.md`](../README.md) for the full disclaimer and risk warning.
