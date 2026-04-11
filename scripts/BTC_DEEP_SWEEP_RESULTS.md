# BTC deep backtest sweep + robustness validation

**Run date**: 2026-04-11
**Test parameters (historical — see SKEPTIC_REVIEW.md)**:
- `hybrid_vol_quantile` baseline: **q=0.70** (the default at the time of this run)
- Window sampler: **buggy (overlapping)** — the 16 "random" windows at seed 42
  shared 75–95% of their bars. Numbers in this doc are NOT comparable to
  post-Round-2 results that use the non-overlap sampler.
- Multi-seed robustness: **yes**, 4 seeds {42, 7, 100, 999} applied in Phase 3.
- See `IMPROVEMENTS_PROGRESS.md` for the Round-2 corrections that supersede
  this doc's numerical claims.

**Scope**: BTC-USD only (S&P 500 was explored in Tier-1S; nothing worked).
**Goal**: map the full hyperparameter surface for the BTC backtest and find
out whether any parameter combination robustly improves on the H-Vol @
q=0.70 production default (median Sharpe 2.15 across 16 random 6-month
windows, seed 42).

## TL;DR

**No parameter tweak robustly improves the baseline.** The deep sweep
(1,616 backtests across 5 dimensions) surfaced several apparent winners
at seed 42 — most notably `sell_bps=-20` at median Sharpe 2.40. All of
them failed a 4-seed robustness check. On average across seeds, the
apparent winners are actually **worse** than the baseline (sell=-20
average Sharpe 0.94 vs baseline 1.00).

**The baseline is a plateau.** Production defaults (H-Vol @ q=0.70,
buy=25, sell=-35, max_long=1.0, retrain_freq=21, 5 states, order 5) are
at a genuine multi-seed local optimum. Further parameter-space exploration
is unlikely to produce a robust improvement without changing the model
class.

**The portfolio experiment (BTC hybrid + S&P B&H) has a soft edge.** The
40/60 BTC/SP mix with daily rebalancing scored median Sharpe 2.44 at
seed 42, beating BTC-alone (2.15). Multi-seed robustness kept the edge
on 3 out of 4 seeds. Average Sharpe across seeds: 40/60 mix = 1.16 vs
BTC-alone = 1.00 (+16%). A real diversification benefit, but not
uniform — at seed 100 specifically, BTC alone was much better. See
`scripts/BTC_SP500_PORTFOLIO_RESULTS.md` for the full portfolio
analysis.

## Methodology

Random-window evaluation across 16 non-overlapping 6-month windows
sampled from BTC 2015-01-01 → 2024-12-31, seed 42 (matching the existing
eval suite). For each configuration, per-window metrics are captured and
median/mean/positive-count are reported. Parquet-format raw data is
saved to `scripts/data/btc_deep_sweep_results.parquet` (long format,
one row per config×window) and `scripts/data/btc_deep_sweep_summary.parquet`
(one row per config with aggregates).

All five experiment dimensions below share the same 16 random windows
so results are directly comparable.

## Experiment dimensions

### Dim 1 — HOMC order × n_states

Fit HOMC at orders {1..9} × n_states {3, 5, 7} = 27 configs. Asks:
"where is the best HOMC memory-depth / state-count combination on BTC?"

**Result**: order=5, n_states=5 is the clear winner (median Sharpe 1.83,
matching the Tier 0b/0c result). Runner-up is order=8 / n_states=3 at
1.42 — notable because it suggests smaller state spaces can support
higher memory orders. Order 1 / 9 / 7-9 with n=7 produce zero trades
(sparsity wall or marginal-fallback collapse).

**Top 3:**

| Rank | Order | n_states | Median Sh | Mean Sh | Median CAGR | Mean MDD | Pos |
|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 5 | 5 | **1.83** | 1.39 | +118% | -22.7% | 10/16 |
| 2 | 8 | 3 | 1.42 | 0.74 | +103% | -26.6% | 10/16 |
| 3 | 3 | 3 | 1.39 | 0.89 | +76% | -25.0% | 9/16 |

**Interpretation**: BTC's Markov memory peaks at order=5 with 5 states.
Further memory depth doesn't help — same sparsity wall we saw on S&P,
just shifted one order higher because BTC's training window is 1000 bars.

### Dim 2 — Composite grid × train_window × laplace_alpha

Composite state grid {2×2, 3×3, 4×4, 5×5} × train_window {252, 504, 1000}
× laplace_alpha {0.01, 1.0} = 24 configs. Asks: "is the production 3×3
/ 252 / 0.01 composite actually optimal, or is there a better composite
config we've missed?"

**Result**: composite 5×5 with train_window=1000 and alpha=0.01 scored
median Sharpe **1.59**, above the production composite 3×3/252/0.01 at
**1.44**. That's +0.15 Sharpe, which looked like a real improvement.

**Top 3:**

| Rank | Grid | Train | Alpha | Median Sh | Mean Sh | Median CAGR |
|:---:|---|---:|---:|---:|---:|---:|
| 1 | 5×5 | 1000 | 0.01 | **1.59** | 1.32 | +143% |
| 2 | 3×3 | 252 | 0.01 | 1.44 | 1.10 | +44% |
| 3 | 4×4 | 1000 | 1.0 | 1.38 | 0.76 | +82% |

**But**: robustness-tested at seeds 7, 100, 999 and the lift disappears.
Composite 5×5/1000 averaged 0.77 Sharpe across 4 seeds vs baseline's
1.00. Not a real improvement.

### Dim 3 — Hybrid vol_quantile × max_long

H-Vol routing with vol_quantile {0.50, 0.60, 0.70, 0.75, 0.80} × max_long
{1.0, 1.25, 1.5, 2.0} = 20 configs. Asks: "does the Tier-0e vol quantile
and the Tier-0f leverage plateau still hold?"

**Result**: q=0.70 remains clearly the best quantile. At q=0.70,
max_long is flat across {1.0, 1.25, 1.5, 2.0} on Sharpe but scales CAGR
proportionally — exactly reproducing the Tier-0f leverage plateau
finding.

**Top 4 at q=0.70:**

| max_long | Median Sh | Mean Sh | Median CAGR | Mean MDD |
|---:|---:|---:|---:|---:|
| 1.0 | **2.15** | 1.59 | +156% | -21.3% |
| 1.25 | **2.15** | 1.59 | +216% | -26.1% |
| 1.5 | 2.07 | 1.59 | +288% | -30.5% |
| 2.0 | 2.09 | 1.61 | +384% | -35.8% |

Other quantiles: q=0.60 scored 1.85, q=0.75 scored 1.92 (old default),
q=0.80 scored 1.87, q=0.50 scored 1.46. **q=0.70 wins with a ~0.20
Sharpe margin**, consistent with Tier 0e.

### Dim 4 — Hybrid retrain frequency

H-Vol @ q=0.70 with retrain_freq {7, 14, 21, 42, 63} = 5 configs. Asks:
"does the production 21-bar retrain frequency matter?"

**Result**: retrain_freq=14 scores 2.16 (vs 21 at 2.15), retrain_freq=42
scores 2.13 — all within noise. retrain_freq=7 (weekly) scores 1.92 and
retrain_freq=63 (quarterly) scores 1.82, both meaningfully worse.

**Interpretation**: the production 21-bar retrain is at the sweet spot.
Weekly retraining is too frequent (over-fitting to noise), quarterly is
too slow (missing regime shifts). Monthly (21-42 bars) is flat.

### Dim 5 — Hybrid buy/sell threshold grid

H-Vol @ q=0.70 with buy_bps {10, 15, 20, 25, 30} × sell_bps {-10, -15,
-20, -25, -30} = 25 configs. Asks: "are the production thresholds
(25/-35) optimal?"

**Result — apparent winner**: buy=25, sell=-20 AND buy=25, sell=-25 both
scored median Sharpe **2.40**. At seed 42 this looked like the BIGGEST
parameter improvement of the whole sweep (+0.25 over the production
baseline at 2.15).

**Top 5:**

| Rank | Buy | Sell | Median Sh | Mean Sh | Median CAGR | Mean MDD |
|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | -20 | **2.40** | 1.62 | +177% | -21.5% |
| 2 | 25 | -25 | **2.40** | 1.58 | +170% | -21.8% |
| 3 | 20 | -20 | 2.27 | 1.51 | +171% | -21.8% |
| 4 | 20 | -25 | 2.27 | 1.48 | +171% | -22.0% |
| 5 | 15 | -20 | 2.24 | 1.41 | +171% | -22.9% |

Note the pattern: **sell_bps between -20 and -25 produces the best
single-seed results** regardless of buy_bps. The production default
sell=-35 looked suboptimal at seed 42.

**But robustness testing killed this result** (see below).

## Phase 3: Multi-seed robustness

Took the top winners from the sweep and re-ran at seeds {7, 100, 999}
to check whether the edges were real or data-mining artifacts. Configs
tested:

- **baseline**: H-Vol @ q=0.70 production default (sell=-35, max_long=1.0)
- **new-A**: sell_bps=-20 (Dim 5 winner)
- **new-B**: sell_bps=-25 (Dim 5 co-winner)
- **new-C**: max_long=1.25 (Dim 3 — same Sharpe, +60pp CAGR at seed 42)
- **new-D**: composite 5×5 / 1000 / 0.01 (Dim 2 winner)
- **new-E**: retrain_freq=14 (Dim 4 marginal)
- **new-F**: combined sell=-20 + max_long=1.25

### Results

| Config | seed 42 | seed 7 | seed 100 | seed 999 | **avg** |
|---|---:|---:|---:|---:|---:|
| **baseline** | 2.15 | -0.27 | **1.38** | 0.74 | **1.00** |
| new-A (sell=-20) | **2.40** | **-0.45** | 1.08 | 0.74 | 0.94 |
| new-B (sell=-25) | **2.40** | -0.27 | 1.15 | 0.74 | 1.00 |
| new-C (max_long=1.25) | 2.15 | -0.28 | 1.38 | 0.76 | 1.00 |
| new-D (5×5 composite) | 1.59 | -0.02 | 1.26 | 0.23 | 0.77 |
| new-E (retrain=14) | 2.16 | -0.14 | 1.06 | 0.84 | 0.98 |
| new-F (combined) | **2.41** | **-0.45** | 1.08 | 0.76 | 0.95 |

### Critical findings

1. **Zero configs beat baseline at all 4 seeds.** The strict test says
   every apparent winner is a single-seed fluke.

2. **sell_bps=-20 is actually WORSE on average.** Average Sharpe 0.94 vs
   baseline 1.00. The +0.25 Sharpe lift at seed 42 came with a -0.18
   loss at seed 7 (and smaller losses at 100 and 999). Over 4 independent
   draws, the change is net NEGATIVE. This is a classic data-mining
   artifact: the "improvement" is really the model fitting noise in one
   specific window draw.

3. **sell_bps=-25 is net-neutral.** Average 1.00, same as baseline. It
   wins big at seed 42 (+0.25) but loses at seed 100 (-0.23). No edge.

4. **max_long=1.25 is a genuine free lunch on CAGR** (same Sharpe across
   seeds, 60-100pp higher CAGR at each). This isn't a "win" in the
   statistical sense but it's a legitimate pick-your-point-on-the-curve
   option for users willing to take the drawdown exposure. Same finding
   as Tier 0f.

5. **Composite 5×5 is net-worse.** Average 0.77 vs baseline 1.00. Bigger
   state space fits noise, degrades on unseen data.

6. **retrain_freq=14 is net-neutral.** Slight wins at seeds 7 and 999
   (+0.13 and +0.10), slight losses at 100 (-0.32). Average 0.98 ≈ 1.00.
   Not meaningfully different from the production 21.

### Key observation: negative seeds are REAL

Look at seed 7: **every config** has negative median Sharpe. Baseline
-0.27, new-A -0.45, new-D -0.02 (least bad). This isn't a bug — seed 7's
16 random windows happen to hit periods where the BTC model struggles
(likely concentrated in 2018 bear / 2022 crypto winter eras). The
baseline's "plateau" isn't guaranteed profitability; it's the least-bad
configuration across diverse market regimes.

A real strategy improvement would show positive Sharpe at ALL four
seeds. None do. The whole BTC model class has a hard floor at certain
market regimes that parameter tuning can't cross.

## Dimension-by-dimension verdicts

| Dim | Apparent seed-42 winner | Baseline (seed 42) | Robust? | Final verdict |
|---|---|---|:---:|---|
| 1 — HOMC memory | order=5/n=5 @ 1.83 | (hybrid uses this internally) | ✅ | Confirms hybrid's internal HOMC config |
| 2 — Composite grid | 5×5/1000/0.01 @ 1.59 | 3×3/252/0.01 @ 1.44 | ❌ | Composite 5×5 over-fits; keep 3×3 |
| 3 — Vol quantile + leverage | q=0.70 × any max_long | same | ✅ | q=0.70 confirmed; max_long is free CAGR dial |
| 4 — Retrain frequency | 14 @ 2.16 | 21 @ 2.15 | ≈ | Net-neutral; keep 21 |
| 5 — Buy/sell thresholds | 25/-20 or 25/-25 @ 2.40 | 25/-35 @ 2.15 | ❌ | Single-seed artifact; keep -35 |

**Dim 1 is a legitimate confirmation** — the HOMC order=5/n=5 config is
what the hybrid already uses internally, so this dimension's result is a
consistency check that passed. Dim 3's quantile result is another
confirmation. The other three dimensions either produced non-robust
"winners" or net-neutral tweaks.

## Implications

1. **The production H-Vol @ q=0.70 defaults are at a plateau.** Not
   because they're uniquely good but because parameter tweaks don't
   robustly beat them across diverse random draws. The Sharpe ~1.0
   average across 4 seeds is the practical ceiling of this model class
   on BTC.

2. **Future improvements have to come from elsewhere.** Parameter tuning
   within the current model classes is exhausted. Candidates (none
   implemented):
   - **Different features**: macro (VIX, credit spreads, yield curve),
     on-chain (BTC dominance, exchange flows), cross-asset (DXY, gold).
     The current composite state is return × vol only.
   - **Different model classes**: gradient boosting, recurrent nets.
     These might find structure the Markov backbone can't.
   - **Non-stationary parameters**: time-varying quantiles based on
     realized volatility regime. The fixed q=0.70 might be optimal on
     average but a dynamic quantile could do better per-regime.
   - **Portfolio construction** (already explored — soft win, see
     BTC_SP500_PORTFOLIO_RESULTS.md)

3. **The multi-seed robustness methodology is now battle-tested.** This
   is the second experiment in the project (after Tier-1S) where
   single-seed results looked promising but failed multi-seed validation.
   The general rule holds: **any apparent winner from a sweep MUST be
   re-run at 3+ alternative seeds before promotion**. Takes ~2-5 minutes
   per config, prevents weeks of follow-up work chasing ghosts.

4. **`new-A` (sell=-20) is a particularly important cautionary tale.**
   At seed 42 it looked like a +12% Sharpe improvement — big enough that
   a naive analysis would promote it. On average across seeds it's
   -6% WORSE. Without the robustness check, this would have been a
   tangible regression in production. DSR (deflated Sharpe) corrects
   for multi-trial bias within one draw but doesn't catch this class of
   error because the draw itself is the source of the bias.

## Data artifacts

Raw data saved to `scripts/data/`:

| File | Description | Rows |
|---|---|---|
| `btc_deep_sweep_results.parquet` | Long format: per-config × per-window | 1,616 |
| `btc_deep_sweep_summary.parquet` | Aggregate: per-config median/mean/etc | 101 |
| `btc_deep_sweep_robustness.parquet` | Multi-seed test: per-config × per-seed × per-window | 640 |
| `btc_sp500_portfolio.parquet` | Portfolio experiment: per-weight × per-rebalance × per-window | 224 |

All parquets are gitignored by default via `*.parquet`, but the
`!scripts/data/*.parquet` exception keeps them tracked so a fresh clone
can inspect the raw numbers without re-running the sweeps.

## Reproducibility

```bash
# Main deep sweep (1,616 backtests, ~3 minutes)
python scripts/btc_deep_sweep.py

# Multi-seed robustness (640 backtests, ~3 minutes)
python scripts/btc_deep_sweep_robustness.py

# Portfolio experiment (224 portfolios built from 16 windows, ~1 minute)
python scripts/btc_sp500_portfolio.py
```

Deterministic across runs (seed 42 for the main sweep; seeds {42, 7,
100, 999} for robustness).

---

⚠ **Disclaimer**: This is an experimental research project. The results
shown above are backtests and historical simulations — they do not
predict future performance. Not financial advice. See the root
[`README.md`](../README.md) for the full disclaimer and risk warning.
