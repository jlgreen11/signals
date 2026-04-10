# Signals — Roadmap of methodology improvements

**Status as of 2026-04-11**: 76 tests passing, CI green on Python 3.11/3.12,
walk-forward engine verified lookahead-clean. **Production scope: BTC-USD
and ^GSPC only** — ETH and SOL deprioritized per user direction. **Current
production defaults:**

- **BTC**: H-Vol hybrid at `hybrid_vol_quantile=0.70` (retuned from 0.75
  after vol quantile sweep). Median Sharpe **2.15** across 16 random 6-month
  windows — best in project history. H-Blend (continuous ramp) is an
  available alternative at median Sharpe 2.06.
- **S&P 500 (^GSPC)**: **Buy & hold.** No active strategy in the project
  beats B&H on S&P. HOMC is the closest at mean Sharpe 1.04 vs B&H's 1.07
  but still loses on the median (0.66 vs 0.77). The hybrids are actively
  negative on S&P (H-Vol median -0.03, H-Blend -0.40). See
  `scripts/HOMC_TIER0E_BTC_SP500.md` for the full comparison.

Full H-Vol comparison on BTC still pinned in `HOMC_TIER0C_HYBRID_RESULTS.md`;
the comprehensive BTC + S&P result is in `HOMC_TIER0E_BTC_SP500.md`; the
sizing and blend ramp sweeps are in `HOMC_TIER0F_SIZING_BLEND.md`.

## Quick note on leverage (BTC)

`max_long` defaults to 1.0 (no leverage) but is a clean risk/return dial.
Sharpe is flat across max_long ∈ [1.0, 2.0]; CAGR scales linearly with
leverage, MDD scales proportionally. Tier-0f sweep:

| `max_long` | Median Sharpe | Median CAGR | Mean Max DD |
|---:|---:|---:|---:|
| 1.00 (default) | 2.15 | +156% | -21% |
| 1.25 | 2.15 | +216% | -26% |
| **1.50** | **2.16** | **+288%** | -30% |
| 2.00 | 2.13 | +480% | -38% |

Pass `--max-long 1.5` if you want ~1.85× the default return at the same
Sharpe and a 9 pp larger drawdown. The default is left at 1.0 deliberately
— it's a risk-tolerance call, not a methodology one.

This document is the **forward-looking improvement plan**. It is divided into
three tiers by effort. Each item explains what to build, why it matters, and
how to validate it (so you can decide whether to keep the change).

The plan is grounded in two pieces of evidence:

1. **The 16-window random evaluation** (`scripts/RANDOM_WINDOW_EVAL.md`) showed
   that the strategy gets the *direction* right far more often than the
   *magnitude* — same direction as buy & hold in winning windows but only half
   the size, and 13/16 windows positive vs B&H 9/16. The structural cost is
   under-participation in strong bull rallies.

2. **The Nascimento et al. (2022) paper** (the source for HOMC) found
   empirically that BTC, ETH, and XRP have **7 steps of memory** and LTC has
   **9** when discretized at granularity 0.01 (1% return bins). The signals
   HOMC backend defaults to `order=3` and uses *quantile* rather than absolute
   bins, so it is structurally undertuned vs the paper's own findings.

The single highest-information experiment is the order=7 HOMC sweep
(see Tier 1 / item 0). Results are pinned at
`scripts/HOMC_ORDER7_RESULTS.md`. If that test confirms the paper's claim,
several Tier-1/2 items move ahead in priority.

---

## Tier 0 — Validation experiment (run first, before any code change)

### 0. HOMC at order=7 with holdout validation — **[x] DONE 2026-04-10**

**Result**: Negative. The paper's claim does not transfer. See
`scripts/HOMC_ORDER7_RESULTS.md` for the full analysis. Headline numbers:

- 25 sweep configs, all DSR = 0.00 (none survive multi-trial deflation)
- Best in-sample: 20.9% CAGR / 0.54 Sharpe / -55% MDD with 8 trades over
  ~5.5 years (B&H over the same window: 34.9% CAGR)
- Holdout collapses to 1.41% CAGR with 1 trade in 512 bars — the strategy
  effectively stops trading because order=7 in a 252-bar walk-forward
  window can't find repeating k-tuples and falls back to the marginal
  distribution

Implication: HOMC at high order is incompatible with adaptive walk-forward
on standard window sizes. Tier-1 #5/#6/#7 (asymmetric sizing, cap removal,
multi-asset eval) move ahead of #1/#2 (granularity encoder + sweep). Tier-3
#12 (sparse k-tuple representation) is deprioritized — it solves a
non-bottleneck.

---

### 0a. Larger train window + lower order — **[x] DONE 2026-04-10 — INCONCLUSIVE**

**Result**: Mixed — strict decision rule says demote, but a controlled
composite baseline run on the same data reveals genuine HOMC-specific
out-of-sample structure that the conventional rule would miss. See
`scripts/HOMC_ORDER5_W1000_RESULTS.md` for the full analysis. Headline:

| | HOMC@order=5 | Composite@3×3 |
|---|---:|---:|
| In-sample Sharpe (2015-2022) | 0.42 | **0.62** |
| **Holdout Sharpe** (2023-2024) | **1.76** | 0.63 |
| Holdout CAGR | **110.1%** | 11.8% |
| In→Out Sharpe ratio | **4.2×** | 1.0× |
| DSR (in-sample) | 0.00 | 0.00 |

The composite baseline is the critical control: if HOMC's holdout strength
were just an artifact of 2023-2024 being a friendly period, the composite
should also have shown a Sharpe lift. It didn't (0.63 → 0.63). The HOMC
found something the composite cannot find. But the in-sample 0.42 Sharpe
means a live operator running this from 2015 would have killed the model
years before the magic period.

**Verdict**: INCONCLUSIVE. Neither promote nor demote until the
confirmatory experiments below run.

**A note on DSR**: this is the first experiment in the project where the
deflated Sharpe was *misleading* rather than corrective. DSR = 0.00
correctly identified the in-sample as indistinguishable from noise, but
it cannot tell you when an in-sample-noisy model has out-of-sample
structure visible from a clean holdout against a controlled baseline.
The right rule is "DSR is the bar for the in-sample result; out-of-sample
validation against a control is the bar for the strategy."

### 0b. Confirmatory experiments — **[x] DONE 2026-04-10 — REGIME-BIASED**

All four experiments run. Full writeup at
`scripts/HOMC_TIER0B_COMPREHENSIVE.md`. Headline:

| # | Experiment | Result |
|---|---|---|
| 1 | 30% holdout on BTC | Sharpe **1.76 → 0.48** (73% collapse) — bull cherry-picked |
| 2 | Random-window eval (16 windows) | HOMC wins **11/16** on both CAGR and Sharpe; mean Sharpe 1.39 vs composite 1.10; median 1.83 vs 1.44 |
| 3 | ETH holdout | HOMC 0.18 vs composite 0.39 (HOMC loses) |
| 3 | SOL holdout | HOMC 1.45 vs composite 1.11; in-sample **DSR 0.70** (first non-zero in project) |
| 4 | Lower-order BTC | Monotonic: order 3 → 0.63, order 4 → 1.19, order 5 → 1.76 on bull holdout |

**Verdict**: HOMC is a bull-regime specialist, not a universal alpha.
Composite is the opposite (bear-resistant, bull-conservative). They are
complementary, not competitors. Both models fail on ETH specifically.

**What does not change**: composite-3×3 stays as the production default
because a single-model default must handle both regimes and composite's
bear defense is the critical safety feature.

**What does change** — priorities flip:

1. **Tier-3 #13 (Hybrid model) is promoted to top priority.** Strongest
   evidence now that a regime-router (HMM → "bull or bear?" → pick
   HOMC/composite accordingly) should beat any single model across all
   windows. This was "nice to have" before; it is now the single
   highest-value experiment.
2. **Tier-1 #1 (AbsoluteGranularityEncoder) is deprioritized.** The
   paper-style granularity sweep is less interesting now that the
   quantile encoder already produces a winning HOMC at the right
   operating point. Marginal improvement on an already-measurable edge.
3. **Tier-1 #5/#6 (aggressive sizing + cap removal) remain high
   priority.** They compound with both models and are the cheapest
   improvements on the list.
4. **Tier-1 #7 (multi-asset suite) becomes more valuable** — now we have
   a real reason to segment assets by regime profile. SOL clearly fits
   HOMC; ETH clearly fits composite; BTC is mixed.

**New experiment added to the list**:
- **0c**: 30% holdout on BTC at order=3 and order=4 (we have order=5 but
  not the lower orders under bear stress). If order=3 is more
  bear-robust, HOMC@order=3/window=1000 becomes the sturdier default for
  bull-biased single-model use, and we've found the real order-vs-
  robustness tradeoff surface.

### 0d. Tier-3 #13 Hybrid model — **[x] DONE 2026-04-10 — NEW DEFAULT**

Implemented `signals/model/hybrid.py` `HybridRegimeModel` — regime router
with two strategies (HMM-based and vol-based). Vol-based is the new default.

**Validation results** (full writeup: `scripts/HOMC_TIER0C_HYBRID_RESULTS.md`):

| Test | Composite | HOMC | H-HMM | **H-Vol** |
|---|---:|---:|---:|---:|
| Random 16-window (median Sharpe) | 1.44 | 1.83 | 0.37 | **1.92** |
| BTC 20% holdout | 0.63 | 1.76 | 1.18 | **2.21** |
| BTC 30% holdout (bear stress) | — | 0.48 | 0.40 | **0.99** |
| SOL 20% holdout | 1.11 | 1.45 | 1.43 | **2.11** |
| ETH 20% holdout | **0.39** | 0.18 | 0.26 | **-0.40** ❌ |

H-Vol wins 4 out of 5 tests by a meaningful margin. The ETH failure is
real and unresolved — for ETH specifically, composite remains the default.
H-Vol is promoted to default for BTC, SOL, and likely all other trending
crypto assets.

**Key insight**: HMM latent-state routing whipsaws on ambiguous regimes
(median Sharpe 0.37, worse than any single model). Vol-based routing is
stable because vol is a deterministic function of recent price action.
This is the first experiment where DSR was outperformed by a different
validation methodology.

**Implementation changes**:
- `BacktestConfig.hybrid_routing_strategy` default is now `"vol"`
- `BacktestConfig.hybrid_vol_quantile` default `0.75` — untuned
- `HybridRegimeModel.composite_train_window=252` by default — composite
  gets its tuned window regardless of the hybrid's wider training slice
- 12 new tests in `tests/test_hybrid.py` including lookahead regression
  on both routing strategies

**What this does NOT resolve**:
1. **ETH is still broken.** No model beats buy & hold reliably on ETH.
2. **Vol quantile threshold is unvalidated.** Default 0.75 is ad hoc.
3. **Mean CAGR is lower than HOMC** (peak-return tradeoff for consistency).

---

### 0c. (Never run — obsoleted by 0d) 30% holdout for lower-order HOMC

Was going to be a follow-up to the Tier-0b monotonic-order-vs-bull finding.
Superseded by the hybrid result: the hybrid now handles bear regimes
better than any lower-order HOMC could, so the order-vs-bear-robustness
frontier is less interesting as a standalone question. Left in the doc as
a historical pointer.



```bash
# Order 3
signals backtest sweep BTC-USD --model homc \
  --start 2015-01-01 --end 2024-12-31 \
  --states 5 --order 3 --train-window 1000 \
  --buy-grid "10,15,20,25,30" --sell-grid "-10,-15,-20,-25,-30" \
  --stop-grid "0" --no-short --rank-by sharpe --top 10 \
  --holdout-frac 0.3

# Order 4
signals backtest sweep BTC-USD --model homc \
  --start 2015-01-01 --end 2024-12-31 \
  --states 5 --order 4 --train-window 1000 \
  --buy-grid "10,15,20,25,30" --sell-grid "-10,-15,-20,-25,-30" \
  --stop-grid "0" --no-short --rank-by sharpe --top 10 \
  --holdout-frac 0.3
```

**Decision rule**:
- If order=3 holdout Sharpe ≥ 0.7 (vs order=5's 0.48) under 30% holdout,
  order=3 is the more robust operating point and should be the single-
  model HOMC default.
- If order=3 also collapses to ~0.5, the HOMC fragility is inherent to
  the model class and only the hybrid approach (Tier-3 #13) resolves it.

---

## Tier 1 — Quick wins (each ~half a day)

### 1. AbsoluteGranularityEncoder

**What**: A new state encoder that bins returns by *absolute width* (e.g.
0.005 = 50 bps, 0.01 = 1%, 0.02 = 2%) instead of by quantile. The existing
`QuantileStateEncoder` adapts to the empirical distribution and loses the
economic interpretation; the absolute encoder lets you say "this state means a
1% up move" and is what the Nascimento paper actually uses.

**Why it matters**: Quantile bins become unstable as the underlying
distribution shifts (a "deep bear" bin in low-vol 2019 is at a different
return level than a "deep bear" bin in high-vol 2022). Absolute bins are
stationary by construction. The paper's headline 7-step memory finding was
only observable at a 0.01 granularity — it likely vanishes under quantile
binning because the bins keep getting redrawn.

**How to validate**: Once Tier 0 (HOMC@order=7) completes, repeat the
experiment with the absolute encoder at granularity 0.01. If the absolute
encoder produces a higher DSR or holdout Sharpe than the quantile encoder at
the same order, it's a permanent improvement.

**Where**: `signals/model/states.py` — new class `AbsoluteGranularityEncoder`,
plus a `--granularity` flag on the HOMC and composite training paths.

### 2. Order-and-granularity sweep flags

**What**: Add `--order-grid "1,3,5,7,9"` and `--granularity-grid
"0.005,0.01,0.02"` to `backtest sweep`. The deflated-Sharpe column already in
place will tell you if any winning combination survives the multi-trial
deflation.

**Why it matters**: Sweep is currently fixed-order. Without an order grid you
can't tell if the paper's 7-step finding holds — you can only test it
manually one order at a time. With holdout already wired up, this becomes a
proper out-of-sample order-selection procedure.

**How to validate**: Run the grid on BTC over 2018-2022 with `--holdout-frac
0.2` reserving 2023-2024. The best in-sample (order, granularity) pair should
not catastrophically degrade on the holdout. If the holdout Sharpe is within
30% of in-sample, the configuration is real.

**Where**: `signals/cli.py` — extend `backtest_sweep`. Already has
multi-trial DSR and holdout split, so this is purely a Cartesian-product
extension.

### 3. AIC/BIC for HOMC order selection

**What**: Add `model.aic` and `model.bic` properties to `HigherOrderMarkovChain`
(and the composite/HMM for parity). With `k` states, order `m`, and `n`
training observations, the free parameter count is `k^m × (k − 1)`. AIC =
2p − 2·log L; BIC = log(n)·p − 2·log L. Then a CLI helper
`signals model select-order BTC-USD --max-order 10` fits orders 1..N and
prints both criteria, letting you pick objectively.

**Why it matters**: The Nascimento paper picks order by eyeballing test MAPE
across orders 1..10 — methodologically weak but pragmatic. AIC/BIC is the
principled way to do the same thing and lets you justify the chosen order
without an out-of-sample run for every change.

**How to validate**: AIC/BIC should agree within ±2 orders for a
well-specified model. If they disagree by more than 4, the model is
mis-specified or the transition matrix is too sparse — that's a useful
warning.

**Where**: `signals/model/homc.py`, `signals/model/composite.py`,
`signals/model/hmm.py`. The HMM already tracks log-likelihood; the others
need it added.

### 4. Activate `top_rules()` as a trading signal

**What**: A `RuleBasedSignalGenerator` that only emits BUY/SELL when the
current k-tuple matches one of the top-K most-frequent training rules with
strong directional consensus (e.g. P(next direction) ≥ 0.7). Falls back to
HOLD when no rule matches.

**Why it matters**: The Nascimento paper's actual proposal is **rule
extraction**, not full marginal-distribution trading. Your HOMC has the
`top_rules()` method already, but the strategy ignores it — it computes the
expected next return from the entire transition matrix and trades the
average. The rule-based variant is sparser, more conservative, and easier to
interpret. It's a natural complement to the composite default.

**How to validate**: Run the rule-based generator on the 16-window random
evaluation set. Expected outcome: lower trade count, higher win rate per
trade, possibly lower CAGR but higher Sharpe. If win rate goes up and Sharpe
holds, integrate as an optional `--rule-based` flag on `signal next` and
`backtest run`.

**Where**: New file `signals/model/rule_signals.py` next to the existing
`signals.py`. Implements the same `SignalGenerator` interface so the engine
can swap.

### 5. Asymmetric / aggressive position sizing — **[x] DONE 2026-04-11 — SHARPE PLATEAU**

Swept (target_scale_bps, max_long) over 16 combinations on BTC
(4 scales × 4 max_long values). Full writeup
`scripts/HOMC_TIER0F_SIZING_BLEND.md`.

**Two findings**:

1. **`target_scale_bps` is effectively inert** in this regime. HOMC's
   expected-return magnitudes are large enough that
   `|expected|/scale >> max_long` for every tested scale, so the
   magnitude saturates at the `max_long` clip and the scale denominator
   drops out of the result. Scales 5, 10, 15, 20 all produce identical
   median Sharpe at each `max_long` level.

2. **`max_long` is a clean risk/return dial** with basically flat
   Sharpe:

   | max_long | Median Sharpe | Median CAGR | Mean MDD |
   |---:|---:|---:|---:|
   | 1.00 (default) | 2.15 | +156% | -21% |
   | 1.25 | 2.15 | +216% | -26% |
   | 1.50 | 2.16 | +288% | -30% |
   | 2.00 | 2.13 | +480% | -38% |

**Prerequisite code fix shipped**: removed the hardcoded 1.0 cap in
`SignalGenerator.generate()`. Before the fix, `max_long > 1.0` had no
effect because magnitude was already capped at 1.0 before the max_long
clip. Now magnitude is `|expected|/scale` uncapped, with `max_long` as
the real leverage ceiling. Backward compatible (`max_long=1.0` gives
identical behavior).

**Verdict**: No default change. The baseline (`scale=20, max_long=1.0`)
is at the Sharpe plateau with the most conservative drawdown profile.
If you're willing to take -30% drawdowns for ~1.85× the return,
`max_long=1.5` is the clear pick. This is a user risk-tolerance
decision, not a methodology decision.

### 6. Confidence-weighted targets without the unit cap

**What**: The current target formula is `min(1, |E[r]| / scale) × confidence`
which caps every signal at 100% notional. Removing the cap (with
`max_long=1.5` or `2.0`) lets the strongest signals get more allocation.

**Why it matters**: Same root cause as #5 — the cap is the structural reason
why a 5x-conviction signal trades the same size as a 1x-conviction signal.
Removing it is the simplest possible sizing fix.

**How to validate**: Run the random-window eval. Specifically watch the
2019-01 cluster of windows where the strategy was directionally correct but
3× too small. If those windows close the gap on B&H, the cap removal worked.

**Where**: `signals/model/signals.py` `SignalGenerator.generate()` — one
line.

### 7. Multi-asset random-window benchmark

**What**: Extend `scripts/random_window_eval.py` to take a list of symbols
(BTC-USD, ETH-USD, SOL-USD, etc.) and produce a per-symbol Sharpe-capture
table. Mirrors the Nascimento paper's BTC/ETH/LTC/XRP comparison.

**Why it matters**: Right now we have 16 BTC windows of evidence and nothing
else. The strategy could be BTC-overfit. The paper's finding that ETH and XRP
share BTC's 7-step memory is testable: run our own framework on those assets
and see if the Sharpe-capture ratio holds up.

**How to validate**: If ETH/SOL Sharpe-capture is within ±5pp of BTC's 16%,
the strategy is asset-general. If it drops to <5%, BTC has unique structure
the strategy is exploiting and the production scope should stay BTC-only.

**Where**: `scripts/random_window_eval.py` — add a `--symbols` flag and a
loop. Need to fetch the data first via `signals data fetch` for each.

---

## Tier 2 — Methodological (each ~1 day)

### 8. Walk-forward purged k-fold cross-validation

**What**: Replace the current contiguous train/test slicing in
`BacktestEngine` with López de Prado–style purged k-fold: split the data into
N folds, hold one out, train on the others, and apply a *purge buffer*
between train and test to eliminate any feature/label overlap. Aggregate
metrics across folds.

**Why it matters**: The current walk-forward is correct as-is (lookahead
verified), but it produces a single equity curve. Purged k-fold gives N
independent estimates of the strategy's Sharpe, which lets you compute
confidence intervals and do paired statistical tests across configurations.
This is the methodology gap that the holdout flag only partly addresses.

**How to validate**: Run on BTC with N=5 folds. The reported Sharpe should
have a confidence interval narrow enough to distinguish the composite default
from a 0.5-Sharpe baseline at p < 0.05. If the CI is wider than that, the
sample size is the limit, not the model.

**Where**: New `signals/backtest/cv.py` next to `engine.py`. Engine refactor
to be optional (current `run()` stays as-is for fast backtests).

### 9. Per-asset model selection

**What**: A CLI command `signals model select BTC-USD` that fits all three
models (composite, HMM, HOMC) at a small grid of hyperparameters, computes
deflated Sharpe and AIC/BIC, and picks the best per asset. Saves the choice
to a per-symbol config file so future runs don't re-fit.

**Why it matters**: Right now the CLI defaults are tuned for BTC. ETH may
prefer HMM, SOL may prefer composite, etc. The Nascimento paper found
different memory profiles per asset (BTC=7, LTC=9). Hard-coding one model is
the wrong default for a multi-asset platform.

**How to validate**: After running on 3-4 assets, check whether the auto-
selected model matches the manually-tuned one. If yes, the auto-selector is
trustworthy; if no, debug the criterion.

**Where**: New `signals/cli.py` command `model_select`, plus a tiny config
file under `data/` per symbol.

### 10. Feature stacking: HMM regime as composite input

**What**: Extend the composite encoder to take a third axis: the HMM regime.
A 3×3×3 = 27-state grid where the HMM regime tag is one of the dimensions.
Trained as a 1st-order Markov chain over 27 states.

**Why it matters**: The composite captures per-bar return × volatility but
can't see multi-day regimes. The HMM captures regime but loses the per-bar
detail. Stacking gives both. The paper doesn't address this — it's a
synthesis the signals codebase is uniquely positioned to test because it has
all three model classes.

**How to validate**: Compare the stacked composite against the plain
composite on the 16-window random eval. Expected outcome: at least one of
the regime-specific cells should fire a clearly-different signal than the
plain composite would, and the random-window Sharpe should be ≥ the plain
composite's. If they're equal, the regime tag isn't adding information.

**Where**: `signals/model/states.py` — new `RegimeAwareCompositeEncoder` that
takes a fitted HMM as a constructor argument. Engine plumbing to pass an HMM
in.

### 11. Real risk-free rate plumbed through the CLI

**What**: A `--risk-free-rate` flag on `backtest run` and `backtest sweep`
that defaults to 0 but accepts annualized values (e.g. 0.045 for 4.5%). The
config field already exists; only the CLI surface is missing.

**Why it matters**: All current Sharpe numbers are computed with Rf = 0,
which is the simplest assumption but inflates the apparent Sharpe by ~0.05
for a strategy returning ~20% annualized. Real comparisons against external
strategies require a real Rf. The infrastructure landed in the audit
follow-up; only the CLI surface is missing.

**How to validate**: Sharpe should drop monotonically as Rf increases. A
strategy with Sharpe 1.3 at Rf=0 should be ~1.25 at Rf=0.04.

**Where**: `signals/cli.py` `backtest_run` and `backtest_sweep` — pass
through to `_build_config`.

---

## Tier 3 — Research-grade (each ~1 week)

### 12. Sparse k-tuple representation throughout the engine

**What**: The HOMC stores transitions as a `dict[tuple, ndarray]` already,
which is sparse — but the `predict_next` path materializes a full marginal,
and several derived computations (n-step, steady-state) implicitly densify.
Refactor to keep everything sparse so order-9 with 5 states fits in memory
even when most k-tuples are unobserved.

**Why it matters**: At order=9 the dense tensor is 5^10 = 9.7M cells, which
is still tractable but the *intermediate* dense operations during prediction
blow the cost up. Sparse end-to-end means orders 9-12 become feasible if the
data supports them.

**How to validate**: Memory profiling at order=9 with 5 states. Should stay
under 100MB during a backtest. If it does, you can run the paper's headline
LTC=9 finding directly.

**Where**: `signals/model/homc.py` — refactor `predict_next`,
`expected_next_return`, `n_step`, and `steady_state` to operate on sparse
intermediate forms.

### 13. Hybrid model — **[x] SHIPPED 2026-04-10**

Implemented and promoted to production default. See
`signals/model/hybrid.py` and `scripts/HOMC_TIER0C_HYBRID_RESULTS.md`.
The winning design was not HMM-routed (my initial guess) but **vol-routed**:
when 20-day realized vol is above the training-distribution 75th quantile,
route to composite (bear-defense); otherwise route to HOMC (bull
participation).

Median Sharpe across 16 random 6-month BTC windows: **1.92** (vs HOMC's
1.83, composite's 1.44). Wins 4 of 5 validation tests. Exception: ETH,
where the hybrid catastrophically underperforms composite (-0.40 Sharpe
holdout) — for ETH specifically, composite stays the default.

Three follow-up items opened as replacements (see Tier 2/3 below):
- **Tune vol quantile threshold** (Tier 2 #11 replacement)
- **Fix ETH regime** (new investigation)
- **Continuous blending** between composite and HOMC (Tier 3 replacement)

### 14. Bayesian model averaging

**What**: Instead of picking one model, weight the three (composite, HMM,
HOMC) by their posterior given the data, and combine their predictions as a
weighted average. The weights update as evidence accumulates.

**Why it matters**: Picking one model (Tier-2 #9) commits to it. Bayesian
averaging hedges across model uncertainty, which is the more honest answer
when you have three plausible candidates and not enough data to decisively
prefer one.

**How to validate**: The averaged predictor should never be worse than the
*worst* individual model and rarely worse than the best. If the averaging
collapses to picking a single model with weight ≈ 1, the data is decisive
and you can use that one. If weights stay near 1/3, the data is not decisive
and averaging is the right call.

**Where**: New `signals/model/bma.py`. Requires a likelihood function for
each model class — the HMM has one already, HOMC and composite need
log-likelihood added.

### 15. Live broker integration

**What**: Take the existing `PaperBroker` and wire a real broker
(Alpaca/Coinbase) behind the same interface. Schedule `signal next` to run
daily via cron, post to the broker, log fills. Reconcile the recorded
signals against actual fills the next day.

**Why it matters**: Everything until now is paper. The whole point is to
trade. Once the methodology is locked, this is the long-tail engineering
work to get from "good backtest" to "deployed on capital you care about".

**How to validate**: Run on a small position size for 30 days. Compare
realized fills against the backtest's modeled fills. If realized PnL is
within ±20% of backtest PnL after fees/slippage, the modeling is honest. If
it's worse than -50%, the backtest is hiding execution costs.

**Where**: `signals/broker/` — new files for each broker, plus a scheduler
script in `scripts/`.

---

## Post-hybrid new items (2026-04-10)

### 16. Tune the vol quantile threshold — **[x] DONE 2026-04-11**

Swept q ∈ {0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90} on BTC + ^GSPC.

**BTC result**: Optimum q=**0.70**, median Sharpe 2.15 (up from 1.92 at
the old default q=0.75). +0.23 Sharpe improvement. Default changed in
code.

**^GSPC result**: Every quantile underperforms buy & hold. Best S&P
quantile (q=0.50) produces median Sharpe 0.29 vs B&H's 0.77. No tune
rescues the hybrid on S&P — recommendation is to hold the index rather
than run a hybrid strategy.

See `scripts/HOMC_TIER0E_BTC_SP500.md` for the full per-quantile
breakdown and `scripts/vol_quantile_sweep.py` for the runner.

### 16a. Tune the H-Blend ramp parameters — **[x] DONE 2026-04-11 — BLEND LOSES**

Swept (blend_low, blend_high) over 16 pairs on BTC. Best pair
(low=0.40, high=0.90) scored median Sharpe **2.07**, which is 0.08
below H-Vol @ q=0.70's **2.15**. Every blend pair in the grid was
worse than the hard switch. **H-Vol @ q=0.70 stays the default;
H-Blend is a research option only.**

Interpretation: the underlying BTC regimes are clean enough that
decisive regime switching beats soft averaging. Blending gives you
50% exposure to the wrong model in the bar before the regime flips,
which is a real cost. Hard switching picks the right model at the
right time.

Full writeup in `scripts/HOMC_TIER0F_SIZING_BLEND.md`. Sweep is
reproducible via `python scripts/blend_ramp_sweep.py`.

### 17. Fix ETH regime — **[deleted 2026-04-11 — out of scope]**

ETH and SOL deprioritized. Production scope is BTC + ^GSPC only.
Historical context preserved in `HOMC_TIER0B_COMPREHENSIVE.md` and
`HOMC_TIER0C_HYBRID_RESULTS.md`.

### 17a. Model class for S&P 500 — **[partially x] DONE 2026-04-11 — CANDIDATES TESTED, NONE WORK**

Tested two candidates from the original list:

1. **200-day MA trend filter** (`TrendFilter`) — median Sharpe 0.57 vs
   B&H 0.77. Reduces drawdowns as expected (-9.4% vs -15.3% mean MDD,
   best drawdown profile of any strategy) but pays too much in return
   to compensate. Not worse than B&H catastrophically, just worse.
2. **50/200 golden cross** (`DualMovingAverage`) — median Sharpe 0.54,
   strictly worse than the simple 200-day rule on S&P. The double-MA
   lag compounds during whipsaws. Kill.

ALSO ran a HOMC memory-depth sweep (orders 1-9) on S&P to complete the
picture of Markov-chain behavior at each memory depth:

- Orders 1, 7, 8, 9 produce zero trades (order=1 too crude, 7-9 hit
  the sparsity wall — at order=9 only 0.05% of possible k-tuples are
  observed in training)
- Orders 2-6 are the "sweet spot" with actual signals
- Order=6 at seed 42 beat B&H with median Sharpe 0.90 (vs B&H 0.77)
- **But a 4-seed robustness check killed that result**: order=6 beats
  B&H on 2/4 seeds and catastrophically loses on 1/4 (seed 7: 0.23 vs
  B&H 1.42). Data-mining artifact.

**Final verdict**: no strategy in the current model class beats B&H on
S&P 500 robustly. S&P recommendation remains "hold SPY".

Full analysis: `scripts/SP500_TREND_AND_HOMC_MEMORY.md`.

**Remaining candidates** for a genuine S&P strategy (none implemented):
1. **Multi-asset portfolio** (new #19 below — highest priority)
2. **Long-horizon trend filters** (24-month windows instead of 6)
3. **Macro-feature regime detection** (VIX, yield curve, credit spreads)
4. **Factor rotation** (SPY + GLD + TLT + cash, 12-1 momentum)

### 19. Multi-asset portfolio construction — **NEW TOP PRIORITY for S&P**

**What**: Construct a 2-asset (or 3-asset) portfolio where BTC uses the
H-Vol hybrid (median Sharpe 2.15) and SPX uses buy & hold (median
Sharpe 0.77). Allocate between them via risk parity, min-vol, or a
simple fixed weight. The cross-asset correlation is low, so the
portfolio Sharpe may exceed both individual Sharpes.

**Why it matters**: This is the FIRST candidate for beating B&H on S&P
that doesn't require new model classes. It uses what's already built.
If BTC/SPX correlation is ~0.2 (typical historical value), a 60/40 SPX/
BTC portfolio could produce a portfolio Sharpe meaningfully above
B&H's 0.77 even before the BTC component is optimized.

**How to validate**: Compute the 16-window random-eval Sharpe of a
weighted portfolio (weights fixed, or risk-parity) using the existing
BTC and SPX result series. No new model fitting required — just
linear combinations of existing equity curves.

**Cost**: ~1 hour to write the portfolio combiner + analysis script.
No new tests needed (portfolio math is straightforward linear
algebra).

**Where**: New `signals/backtest/portfolio_blend.py` helper +
`scripts/btc_sp500_portfolio.py` analysis script.

### 18. Continuous blending hybrid — **[x] DONE 2026-04-11**

Shipped as `routing_strategy="blend"` in `signals/model/hybrid.py`.
Linear weight ramp from `blend_low_quantile` (full HOMC) to
`blend_high_quantile` (full composite) of training vol. SignalGenerator
compatibility achieved via a synthetic 1-state interface: `state_returns_
= [blended_expected_return]`, `predict_next = [1.0]`, so `expected =
probs @ state_returns_ = blended_expected_return`.

**Result on BTC**: median Sharpe 2.06 at default ramp (0.50, 0.85), vs
2.15 for H-Vol at the retuned q=0.70. H-Blend beats the old H-Vol
default (1.92) but loses to the new one. Blend is marginally better
than a hard switch on ambiguous windows (notably the 2022 crypto winter
where H-Blend +32.2% vs H-Vol -9.2%) but loses overall because it over-
participates in high-vol bulls.

**Next step**: Tune the blend ramp parameters (see #16a above). A 2D
sweep may find a (low, high) pair that beats 2.15.

5 new tests in `tests/test_hybrid.py` including lookahead regression.

---

## How to use this document

When you ask "what should I work on next?", I'll consult this file and pick
the highest-priority item that:

1. has its prerequisites satisfied (e.g., #5 doesn't need anything; #8
   benefits from #11 being in place);
2. matches your current research focus; and
3. has a clear validation path so we know if it worked.

When an item completes, mark it with `[x]` and append a date + outcome line
("CAGR went up 5pp on the 16-window eval; Sharpe unchanged"). Items that get
disproven by their own validation should stay in the doc with the disproof
recorded — that's how we avoid re-deriving the same negative result in six
months.

The companion file `scripts/HOMC_ORDER7_RESULTS.md` is the saved output of
the Tier-0 experiment and is the first piece of evidence that should
influence which Tier-1 items move ahead.
