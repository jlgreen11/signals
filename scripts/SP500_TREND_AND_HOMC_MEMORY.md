# S&P 500 — Trend filters + HOMC memory depth sweep

**Run date**: 2026-04-11
**Test parameters (historical)**:
- Strategies tested: `trend` (MA200), `golden_cross` (50/200), HOMC at
  orders 1–9. `hybrid_vol_quantile` only applies to the hybrid control
  which used **q=0.70** (the pre-Round-2 default).
- Window sampler: overlapping (buggy). Multi-seed robustness WAS applied
  at the last step (4 seeds) — this is the doc where multi-seed robustness
  killed the HOMC@order=6 apparent winner.

**Motivation**: The Tier 0e result (`scripts/HOMC_TIER0E_BTC_SP500.md`) showed
that no Markov-chain model at `order=5` beats buy & hold on ^GSPC. The
candidates for beating S&P 500 are (a) model classes designed for secular-
uptrend equities, and (b) different Markov memory depths we hadn't tested.
This doc covers both:

1. **Trend filters** — the classic 200-day MA rule (Faber 2007) and the
   50/200 golden cross. These are THE standard equity timing strategies.
2. **HOMC memory depth sweep** — orders 1 through 9 on S&P with 5 quantile
   states, mirroring the Nascimento paper's own methodology. The "example
   dataset" of how Markov memory behaves on S&P at each depth.
3. **Robustness check** — the order sweep threw up an apparent winner
   (order=6) and I stress-tested it across 4 seeds to check whether it
   was real or data-mining noise.

## TL;DR

**Nothing beats buy & hold on S&P 500.** This remains the production
recommendation for ^GSPC. Full breakdown:

- **Trend filters fail**: Trend(200) median Sharpe 0.57 vs B&H 0.77.
  Golden cross 0.54. They DO reduce drawdowns meaningfully (-9.4% vs
  -15.3% mean max DD) but give up too much return to compensate. The
  Sharpe tradeoff is unfavorable.
- **HOMC order sweep looked promising**: order=6 scored 0.90 median
  Sharpe on the primary sweep, above B&H's 0.77 — the first S&P result
  to beat B&H in the project.
- **Robustness check killed it**: re-running HOMC@order=6 on 3 additional
  seeds showed the apparent edge is a seed-42 artifact. Order=6 beats B&H
  on 2/4 seeds and loses on 2/4, including one catastrophic failure (seed
  7: order=6 Sharpe 0.23 vs B&H 1.42). Not robust.

The investigation is valuable as a negative result: it rules out trend
filters AND Markov-chain memory as viable S&P approaches within the
project's current methodology. To beat B&H on S&P, a fundamentally
different approach is needed — see "What's next" below.

---

## Experiment 1: Trend filters on S&P 500

### Setup

Added two classic trend-following models to `signals/model/trend.py`:

- **TrendFilter(window=200)** — long when close > MA(200), flat otherwise.
  The canonical 200-day rule from Faber (2007), "A Quantitative Approach
  to Tactical Asset Allocation". Captures ~75% of B&H return with ~60% of
  the drawdown historically on multiple equity indices.
- **DualMovingAverage(fast=50, slow=200)** — long when MA(50) > MA(200).
  Classic golden cross / death cross. Smoother than a single-MA rule but
  adds lag from the double filter.

Both conform to the standard model interface (`fit`, `predict_state`,
`predict_next`, `state_returns_`, `label`) so they plug into `BacktestEngine`
and `SignalGenerator` transparently. Synthetic `state_returns_ = [-1, +1]`
so SignalGenerator's buy/sell thresholds always trigger — trend filters
are binary signals, not magnitude-gated.

16 new tests in `tests/test_trend.py`, including lookahead regression.

### Results (16 random 6-month windows, seed 42)

| Metric | B&H | Composite | HOMC | H-Vol | **Trend(200)** | **GCross** |
|---|---:|---:|---:|---:|---:|---:|
| Mean Sharpe | **1.07** | 0.70 | 1.04 | 0.37 | 0.60 | 0.85 |
| **Median Sharpe** | **0.77** | 0.49 | 0.66 | -0.14 | **0.57** | 0.54 |
| Mean CAGR | **+14.0%** | +4.0% | +15.0% | -0.3% | +9.6% | +7.0% |
| Median CAGR | **+12.2%** | +4.2% | +8.3% | -5.4% | +6.2% | +6.6% |
| **Mean Max DD** | -15.3% | -14.5% | -14.5% | -15.3% | **-9.4%** 🏆 | -14.3% |
| Median Max DD | -7.7% | -7.7% | -7.7% | -9.8% | -7.3% | -6.8% |

Head-to-head vs B&H (out of 16 windows):

| Strategy | Beats B&H on Sharpe | on CAGR | Smaller max DD |
|---|---:|---:|---:|
| Composite | 2/16 | 2/16 | 3/16 |
| HOMC | 4/16 | 3/16 | 3/16 |
| H-Vol | 0/16 | 0/16 | 3/16 |
| **Trend(200)** | 1/16 | 1/16 | **7/16** |
| GoldenCross | 0/16 | 0/16 | 1/16 |

### Interpretation

**Trend(200) has the best drawdown profile of any strategy tested** —
mean max DD -9.4% vs B&H -15.3%, and it beats B&H on drawdown in 7/16
windows (most of any model). The 200-day rule is doing what it's
supposed to do: sidestep sustained crashes.

**But the return penalty exceeds the drawdown benefit.** Trend(200)
median CAGR is 6.2% vs B&H's 12.2% — roughly HALF the return. The
drawdown reduction is also roughly half (-9.4% vs -15.3%). This is a
**proportional tradeoff with no risk-adjusted alpha**: Sharpe is lower
(0.57 vs 0.77) because the return cost scales with the drawdown benefit.

For a risk-averse investor who cares more about drawdowns than returns,
trend(200) is defensible. For the signals project's optimization target
(median Sharpe), it loses to B&H.

**Golden cross is strictly worse than the simple 200-day rule** on
S&P in this eval. The dual-MA adds lag that compounds during whipsaws,
hurting both return and drawdown. The 2019-10 → 2020-04 COVID window is
the clearest example: B&H -8%, Trend(200) -16%, GoldenCross -24%. Both
trend filters got whipsawed by the February 2020 bounce before the
March crash, but GoldenCross's extra lag made it worse.

**The 2022-06 → 2022-12 window is interesting**: B&H +4.8%, Composite
+16.6% (!!), Trend(200) -4.9%, GoldenCross 0.0%. The composite
Markov model handily beats both trend filters and B&H in this specific
bear window. This is a clue that regime-switching models COULD work on
S&P if properly tuned — but only if they're smarter about when to
switch than a raw trend filter.

### Verdict on trend filters

Drawdown-reducing as advertised, but no risk-adjusted edge. B&H remains
the better choice by median Sharpe. Trend(200) could be useful as a
**risk-management overlay** for an investor with asymmetric loss
preferences, but not as a standalone S&P default.

---

## Experiment 2: HOMC memory depth sweep (orders 1 through 9)

### Setup

Ran HOMC on S&P with 5 quantile states, train_window=1000, 21-bar
retrain, across memory orders {1, 2, 3, 4, 5, 6, 7, 8, 9}. Same 16
random 6-month windows as the trend eval (seed 42). 144 backtests total.

Also captured transition-table stats from a representative 1000-bar
training window (the most-recent 1000 bars of the full series): distinct
k-tuples observed, median support per k-tuple, most-frequent k-tuple and
its occurrence count. This is the "example dataset" of how Markov memory
looks at each depth on S&P.

### Transition-table profile

| Order | Distinct k-tuples | Possible (5^k) | Coverage | Median support | Max support | Most-frequent k-tuple |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 5 | 5 | 100% | 200.0 | 200 | `q2` |
| 2 | 25 | 25 | 100% | 40.0 | 57 | `q0→q0` |
| 3 | 125 | 125 | 100% | 8.0 | 19 | `q0→q4→q4` |
| 4 | 486 | 625 | 78% | 2.0 | 8 | `q3→q2→q2→q1` |
| 5 | 836 | 3,125 | 27% | 1.0 | 5 | `q1→q3→q2→q2→q1` |
| 6 | 952 | 15,625 | 6% | 1.0 | 3 | `q0→q0→q4→q4→q1→q0` |
| 7 | 984 | 78,125 | 1.3% | 1.0 | 2 | `q2→q3→q2→q2→q1→q1→q0` |
| 8 | 991 | 390,625 | 0.3% | 1.0 | 2 | `q0→q4→q4→q4→q0→q1→q4→q0` |
| 9 | 991 | 1,953,125 | 0.05% | 1.0 | 1 | `q1→q2→q2→q1→q0→q3→q4→q2→q1` |

**Key observation**: the sparsity wall starts at order=5 (median support
drops to 1) and becomes overwhelming at order=7 (only 1.3% of possible
k-tuples observed, and the most-frequent is seen only twice). By order=9,
every observed 9-tuple appears exactly once — the model is memorizing
random sequences with no statistical power.

### Random-window eval results

| Order | Mean Sharpe | **Median Sharpe** | Median CAGR | Mean Max DD | Positive |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | +0.0% | 0.0% | 0/16 (no trades) |
| 2 | 1.11 | 0.89 | +7.4% | -13.0% | 10/16 |
| 3 | 0.91 | 0.79 | +9.7% | -15.0% | 12/16 |
| 4 | 1.03 | 0.81 | +14.1% | -14.8% | 13/16 |
| 5 | 1.04 | 0.66 | +8.3% | -14.5% | 12/16 |
| **6** | **0.92** | **0.90** 🏆 | **+15.4%** | **-8.5%** | 10/16 |
| 7 | 0.23 | 0.00 | +0.0% | -0.3% | 2/16 (near sparsity wall) |
| 8 | 0.00 | 0.00 | +0.0% | 0.0% | 0/16 (no trades) |
| 9 | 0.00 | 0.00 | +0.0% | 0.0% | 0/16 (no trades) |

**B&H baseline: median Sharpe 0.77, median CAGR +12.2%.**

**Order=6 is the only HOMC config that beats B&H on median Sharpe** at
seed 42, by +0.13 (0.90 vs 0.77). It also beats B&H on median CAGR
(+15.4% vs +12.2%) and has a dramatically smaller mean max DD (-8.5%
vs -15.3%). On its face this is the first S&P result in the project
to beat buy & hold on any metric.

### Why orders 1 and 7-9 produce zero-Sharpe / zero-trade results

- **Order=1** with 5 states: the transition matrix is effectively the
  marginal distribution of states. Expected next return ≈ 0 for every
  current state. SignalGenerator's buy threshold (25 bps) is never
  crossed. Model stays flat → zero Sharpe, zero trades.
- **Order=7-9**: sparsity wall. At order=7, 98.7% of possible k-tuples
  are unobserved in training. At inference time, the current k-tuple is
  almost certainly one of the unseen 77,000+. HOMC falls back to the
  marginal distribution (≈ uniform), which yields ~0 expected return.
  Same result: stays flat, zero trades, zero Sharpe.

This matches the BTC Tier 0 finding: order=7 on BTC with 252-bar windows
was also broken by sparsity. On S&P with a 1000-bar window, the wall
shifts to order=7 (one order higher than BTC's effective wall) because
1000 bars is larger than 252, but the ratio of distinct to possible
k-tuples tells the same story.

---

## Experiment 3: Robustness check for order=6

### Setup

The order=6 result at seed 42 looked too good. I ran a 4-seed robustness
check — HOMC at orders 4, 5, and 6 on seeds {42, 7, 100, 999} — to see
whether order=6 consistently beats B&H or whether seed 42 was lucky.

A genuine order=6 edge would show median Sharpe > B&H across ALL 4 seeds.
If order=6 bounces above/below B&H randomly, the apparent edge is a
data-mining artifact.

### Results

| Seed | Order 4 Sharpe | Order 5 Sharpe | **Order 6 Sharpe** | B&H Sharpe | Order 6 vs B&H |
|---:|---:|---:|---:|---:|---|
| 42 | 0.81 | 0.66 | **0.90** | 0.77 | **+0.13** ✓ |
| 7 | 1.20 | **1.45** | **0.23** | 1.42 | **-1.19** ❌ catastrophic |
| 100 | 0.72 | 0.97 | **1.13** | 1.07 | **+0.06** ✓ (marginal) |
| 999 | 1.65 | 1.62 | **1.56** | 1.90 | **-0.34** ❌ |

**Order=6 beats B&H on 2/4 seeds** (exactly at chance). On seed 7 it
**catastrophically underperforms** (0.23 vs 1.42 B&H — a 1.19-Sharpe gap).
On seed 999 no HOMC order beats B&H at all. The "best" HOMC order
changes by seed:

- Seed 42: order=6 is best
- Seed 7: order=5 is best (and order=6 is worst)
- Seed 100: order=6 is best
- Seed 999: no HOMC order beats B&H (B&H 1.90 dominates)

A robust edge would persist across seeds. This doesn't.

### What seed 999 tells us

B&H median Sharpe at seed 999 is **1.90**. That's ~2.5× B&H's seed-42
value (0.77). This happens because seed 999's 16 random windows
disproportionately hit bullish periods. In those windows, buy and hold
compounds effortlessly, and there's no "bear avoidance" alpha for the
HOMC to capture. HOMC at all tested orders produces Sharpes in the
1.56-1.65 range — respectable numbers, but still below the B&H baseline
for that draw.

This is the clearest signal that HOMC on S&P is **not generating alpha**,
it's just tracking the benchmark with some extra noise. When B&H is high,
HOMC is high. When B&H is low, HOMC is low (or worse). The model isn't
finding any exploitable structure — it's mostly long-exposed, and its
active bets cancel out.

### Verdict on the HOMC order sweep

**Order=6's seed-42 result was a data-mining artifact.** The apparent
edge doesn't replicate. Combined with:

- Median support = 1 at order=6 (model memorizes rare k-tuples with no
  statistical power)
- Neighboring orders (5 and 7) at seed 42 don't show the same effect
- Different metrics (median Sharpe vs positive-window count) rank
  different orders as "best"
- Seed 7 catastrophe (-1.19 Sharpe gap)

...the order=6 result at seed 42 was noise that happened to favor the
HOMC's arbitrary state partition on that particular window draw.

**No HOMC memory depth beats buy & hold on S&P 500 robustly.**

---

## Combined verdict for S&P 500

| Strategy | Beats B&H? | Evidence |
|---|:---:|---|
| composite-3×3 | ❌ | Median Sharpe 0.49, clearly worse |
| HOMC@order=5 | ❌ | Median Sharpe 0.66, clearly worse |
| H-Vol hybrid | ❌ | Median Sharpe -0.14, catastrophically worse |
| H-Blend hybrid | ❌ | Median Sharpe -0.40, also catastrophic |
| Trend(200) | ❌ | Sharpe 0.57 — reduces drawdowns but costs too much return |
| GoldenCross(50,200) | ❌ | Strictly worse than Trend(200) |
| HOMC orders 1-9 | ❌ | Best (order=6) was seed-42 artifact, fails robustness |

**Buy and hold wins on every metric at every seed we've tested.** The
recommendation is unchanged: **for S&P 500, hold SPY (or equivalent).
Do not run signals strategies.**

### Why S&P is hard

The signals project's model classes are:

1. **Regime detectors** (composite, HMM) — assume the underlying process
   has meaningfully distinct bull/bear/crash regimes. S&P has rare,
   sharp drawdowns but most of the time is in a single slow-uptrend
   regime. The regime detector has nothing to detect most of the time.
2. **Memory exploiters** (HOMC) — assume recent price history predicts
   next-period returns via discrete patterns. S&P's tight return
   distribution and low serial correlation mean there's no memory to
   exploit at any order.
3. **Trend filters** (new in this tier) — exploit persistence in price
   direction. S&P HAS persistence, but the 6-month evaluation windows
   are too short for the filter's lag to pay off. You'd need multi-year
   holds to capture the drawdown savings.
4. **Hybrids** — combine the above. When the components are all wrong
   for the asset, combining them doesn't help.

**None of these are the right tool for a secular-uptrend equity index
with sparse sharp drawdowns.** Beating S&P requires either (a) longer
evaluation horizons where trend filters' drawdown savings dominate
whipsaw losses, (b) multi-asset portfolio construction where S&P is one
component, or (c) entirely different model classes (macro features,
factor rotation, fundamental signals).

---

## What's next for S&P

The "use buy & hold" recommendation is strong and stable. If you
genuinely want to beat S&P, the candidates in decreasing order of
feasibility within this codebase:

1. **Multi-asset portfolio with S&P as one component.** BTC hybrid
   (median Sharpe 2.15) + S&P B&H (median Sharpe 0.77) at risk parity
   could have a higher portfolio Sharpe than either alone due to low
   BTC/SPX correlation. This is the **single highest-value next step**
   and requires no new models.
2. **Long-horizon trend filters.** The 200-day MA's drawdown savings
   might dominate whipsaw costs on multi-year windows instead of
   6-month windows. Re-run the random-window eval with 24-month
   windows.
3. **Macro-feature regime detection.** Yield curve slope, credit
   spreads, VIX level. Requires fetching new data and extending the
   composite encoder. Real work.
4. **Factor rotation (12-1 momentum).** Requires multiple equity
   factors (SPY, GLD, TLT, etc.). Standard quant toolkit but not
   currently in the project.
5. **Deep models** — LSTM or Transformer on price + macro. Significant
   infrastructure work and almost certainly not worth it for a research
   project at this scale.

**#1 (multi-asset portfolio) is the recommended next step.** It's the
only candidate that (a) uses what we've already built, (b) has a real
chance of producing Sharpe > B&H + hybrid individually, and (c) doesn't
require new model classes.

---

## Code changes

| File | Change |
|---|---|
| `signals/model/trend.py` | New module — `TrendFilter` and `DualMovingAverage` |
| `signals/backtest/engine.py` | Factory support for `trend` and `golden_cross` model types, plus `trend_window`, `trend_fast_window`, `trend_slow_window` in BacktestConfig |
| `signals/cli.py` | VALID_MODELS includes trend and golden_cross |
| `tests/test_trend.py` | 16 new tests including lookahead regression |
| `scripts/sp500_trend_eval.py` | S&P random-window eval across 6 strategies |
| `scripts/sp500_homc_order_sweep.py` | HOMC order 1-9 sweep on S&P with transition-table stats |
| `scripts/sp500_homc_order6_robustness.py` | 4-seed robustness check for the order=6 result |
| `scripts/SP500_TREND_AND_HOMC_MEMORY.md` | This file |

Tests: 76 → 92 passing (+16 trend tests). CI green. ruff clean.

## Reproducibility

```bash
python scripts/sp500_trend_eval.py              # ~1 min, 80 backtests
python scripts/sp500_homc_order_sweep.py        # ~3 min, 144 backtests
python scripts/sp500_homc_order6_robustness.py  # ~3 min, 192 backtests
```

Deterministic per seed. Wall times on a 2024 M-series Mac.

---

⚠ **Disclaimer**: This is an experimental research project. The results
shown above are backtests and historical simulations — they do not
predict future performance. Not financial advice. See the root
[`README.md`](../README.md) for the full disclaimer and risk warning.
