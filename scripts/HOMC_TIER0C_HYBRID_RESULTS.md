# Tier 0c / Tier-3 #13 — Hybrid regime-routed model

**Run date**: 2026-04-10
**Test parameters (historical)**:
- **This is the doc that introduced the hybrid model.**
- `hybrid_vol_quantile`: **q=0.75** (ad-hoc initial choice, never sweep-tuned
  in this doc — the sweep that produced q=0.70 came later in Tier 0e,
  and the sweep that produced q=0.50 came in Round 2)
- `routing_strategy`: "vol" was the winner; "hmm" was tested and lost
- Window sampler: overlapping (buggy)
- See `IMPROVEMENTS_PROGRESS.md` for the Round-2 correction that replaces
  the 1.92 median Sharpe with 0.78 multi-seed.

**Motivation**: Tier-0b (see `HOMC_TIER0B_COMPREHENSIVE.md`) established that
HOMC@order=5/window=1000 is a bull-regime specialist and composite-3×3 is a
bear-defensive model. The two are complementary, not ranked. The hypothesis
to test was whether a regime router between them would beat both single
models across all regimes.

**Result**: A vol-routed hybrid (not HMM-routed) achieves this. It
outperforms every prior model on 4 of 5 validation tests, including the
bear-stress test that previously broke HOMC, and becomes the new
production default.

**TL;DR**: `BacktestConfig.hybrid_routing_strategy` now defaults to `"vol"`.
HOMC-aware, bear-defensive, consistent across regimes.

---

## Architecture

`signals/model/hybrid.py` `HybridRegimeModel` wraps:

1. **A regime signal** that classifies each bar into {bear, neutral, bull}
2. **Composite-3×3** fit on the *trailing 252 bars* of the training window
3. **HOMC@order=5** fit on the *full 1000 bars* of the training window
4. **A routing table** that maps each regime label → a component to delegate
   to. Default routing: `{bear: composite, neutral: composite, bull: homc}`

At each inference bar:

- `predict_state(observations)` asks the regime signal for a label, picks a
  component from the routing table, delegates the state decoding to the
  component, and stashes the active component on the instance
- `predict_next(state)`, `state_returns_`, and `label(state)` all delegate
  to the stashed active component

Two routing strategies are supported via `routing_strategy` (default "vol"):

### Strategy 1: HMM-routed (`routing_strategy="hmm"`)

An internal 3-state Gaussian HMM classifies the current bar by latent
regime. `label(state)` returns "bear"/"neutral"/"bull" ranked by avg return.

### Strategy 2: Vol-routed (`routing_strategy="vol"`) — **default**

Uses realized 20-day volatility vs a training-distribution quantile
(default: 75th percentile). If current vol ≥ threshold → "bear" regime
(high-vol = crash risk) → composite. Otherwise → "bull" regime → HOMC.

The training threshold is recomputed on every retrain, so the model adapts
to regime shifts without re-fitting the classifier itself.

---

## Why vol routing beat HMM routing

Both strategies were tested against the same 16-window random BTC evaluation
and the same 4 holdout sweeps. The HMM version was catastrophic on the
random eval despite the bear-fix:

| Metric | **HOMC** | **H-HMM** | **H-Vol** |
|---|---:|---:|---:|
| Mean Sharpe | 1.39 | 0.94 | **1.47** |
| **Median Sharpe** | 1.83 | **0.37** | **1.92** |
| Sharpe capture vs oracle (median) | 21.5% | 4.9% | **22.9%** |

The HMM classifier flips between "bull" and "neutral" on single-bar noise
in choppy markets, forcing the hybrid to switch components mid-window and
losing on both sides of the transition. The vol signal is stable — it can
only change as fast as the rolling window allows, and crashes are exactly
"high sustained vol" events, so the classifier and the underlying regime
are perfectly aligned.

The vol threshold also has a crisp economic interpretation: "when realized
vol is in the top 25% of the training distribution, the market is risk-off
and composite's bear-defense is the right tool". HMM latent states don't
have this interpretability and therefore have no external check on when
the classifier is wrong.

---

## Random-window evaluation (16 random 6-month BTC windows, seed 42)

Same 16 windows as the Tier-0b evaluation. All five strategies run on the
same data.

### Aggregate (mean and median across 16 windows)

| | B&H | Composite | HOMC | H-HMM | **H-Vol** |
|---|---:|---:|---:|---:|---:|
| **Mean Sharpe** | 1.03 | 1.10 | 1.39 | 0.94 | **1.47** |
| **Median Sharpe** | 1.18 | 1.44 | 1.83 | 0.37 | **1.92** |
| Mean CAGR | +434% | +321% | +417% | +456% | +235% |
| Median CAGR | +101% | +44% | +118% | +10% | **+134%** |
| Mean Max DD | -28% | -21% | -23% | -22% | -21% |
| Median Max DD | -24% | -16% | -15% | -16% | -19% |
| Sharpe capture (median) | — | 15.8% | 21.5% | 4.9% | **22.9%** |
| Positive CAGR | 9/16 | 11/16 | 10/16 | 11/16 | **12/16** |

H-Vol is simultaneously:
- The highest median Sharpe of any model (1.92)
- The highest median CAGR (134% — beats both B&H and HOMC)
- The highest Sharpe capture vs oracle (22.9%)
- Tied for lowest mean max DD (-21%, same as composite)
- Most positive windows (12/16)

It is NOT the highest mean CAGR — the mean is lower than B&H/HOMC/H-HMM
because vol-routing trims the biggest bull peaks (it classifies strong
high-vol bulls as "risk-off" and underparticipates). The tradeoff is peak
return for consistency. For production deployment on a BTC strategy, the
consistency profile is the one you want.

### Head-to-head on Sharpe

| Comparison | Wins |
|---|---:|
| H-Vol beats Composite | **10/16** |
| H-Vol beats HOMC | **8/16** |
| H-Vol beats Buy & Hold | 9/16 |
| H-Vol beats H-HMM | 12/16 |
| HOMC beats Composite | 11/16 |
| HOMC beats Buy & Hold | 12/16 |
| Composite beats B&H | 5/16 |

### Per-window detail (selected windows)

**Bear and crash windows** (where composite's defense matters):

| Window | B&H | Composite | HOMC | H-HMM | **H-Vol** |
|---|---:|---:|---:|---:|---:|
| 2018-10-03 → 2019-02-05 (late-2018 crash) | -84.1% | +26.4% | -62.0% | +7.8% | +7.2% |
| 2018-10-30 → 2019-03-04 (same crash) | -78.2% | +20.1% | -52.5% | +7.2% | **+33.9%** |
| 2022-07-06 → 2022-11-08 (crypto winter) | -25.9% | +28.3% | -14.4% | +12.7% | -9.2% |

H-Vol improves on HOMC in all three bear windows. It doesn't quite match
composite's +28% on the 2018 crash (+7.2%) but it goes **positive +33.9%**
on the overlapping window 5 — better than composite's +20.1%.

**Chop and sideways windows** (where HMM routing fails):

| Window | B&H | Composite | HOMC | H-HMM | **H-Vol** |
|---|---:|---:|---:|---:|---:|
| 2020-05-21 → 2020-09-23 | +42.3% | -21.4% | +84.6% | -12.5% | **+84.6%** |
| 2020-07-11 → 2020-11-13 | +426.8% | +103.2% | +230.5% | +2.7% | **+230.5%** |
| 2024-05-26 → 2024-09-28 | -10.8% | -39.2% | +22.5% | -28.8% | **+111.1%** |

H-Vol matches HOMC exactly in windows 11 and 12 (it correctly classifies
those as low-vol and routes to HOMC). Window 16 (2024-05 → 2024-09) is
the most striking: H-Vol +111.1% vs every other model losing or barely
positive. The vol router identified that period as a low-vol environment
and HOMC happened to catch a trending move composite and B&H both missed.

**Pure bulls** (where HOMC shines):

| Window | B&H | Composite | HOMC | H-HMM | **H-Vol** |
|---|---:|---:|---:|---:|---:|
| 2019-01-11 → 2019-05-16 | +821.5% | +332.4% | +389.6% | +570.6% | +341.7% |
| 2019-05-06 → 2019-09-08 | +472.5% | +308.3% | +1061.3% | +291.0% | +686.4% |
| 2020-11-11 → 2021-03-16 | +4183.4% | +3713.1% | +4054.7% | +5824.0% | +1351.6% |
| 2023-11-23 → 2024-03-27 | +515.6% | +395.8% | +480.8% | +404.7% | +339.0% |

H-Vol underperforms HOMC in strong bulls because it routes to composite
during high-vol rallies (window 13 is the clearest case: +1352% vs HOMC's
+4055% vs H-HMM's +5824%). This is the peak-return cost of the vol
router. It's not catastrophic — H-Vol still makes money in all of them.

---

## Holdout sweeps (20% and 30% holdout validation)

Four independent sweeps on three assets + a bear-stress test on BTC. 25
config trials each (buy_bps × sell_bps × stop). Holdout is the trailing
fraction reserved from the sweep.

### Headline

| Test | Composite | HOMC | H-HMM | **H-Vol** |
|---|---:|---:|---:|---:|
| BTC 20% holdout (bull window) | 0.63 | 1.76 | 1.18 | **2.21** |
| BTC 30% holdout (bull + 2022 bear) | — | 0.48 | 0.40 | **0.99** |
| ETH 20% holdout | **0.39** | 0.18 | 0.26 | **-0.40** |
| SOL 20% holdout | 1.11 | 1.45 | 1.43 | **2.11** |

### BTC 20% holdout (2023-01-01 → 2024-12-31, pure bull)

| Metric | HOMC | H-HMM | H-Vol |
|---|---:|---:|---:|
| Train Sharpe | 0.42 | 0.50 | 0.86 |
| **Holdout Sharpe** | 1.76 | 1.18 | **2.21** |
| Holdout CAGR | 110.0% | 52.0% | **141.1%** |
| Holdout Max DD | -20.9% | -18.7% | **-14.2%** |
| Holdout Calmar | 5.26 | 2.77 | **9.94** |
| Trades (train / holdout) | 25 / 57 | 311 / 120 | 259 / 55 |

H-Vol is the best by every metric on the bull holdout. Holdout Sharpe 2.21
is 26% higher than HOMC's 1.76, with smaller drawdown.

### BTC 30% holdout (2022-02-22 → 2024-12-31, bull + 2022 bear)

This is the critical test — the one that killed HOMC (Sharpe collapsed
from 1.76 → 0.48 when the 2022 bear was added).

| Metric | HOMC | H-Vol |
|---|---:|---:|
| Train Sharpe | 0.71 | 1.14 |
| **Holdout Sharpe** | **0.48** | **0.99** |
| Holdout CAGR | 16.4% | **46.0%** |
| Holdout Max DD | -66.3% | -48.2% |

**H-Vol more than doubles HOMC's bear-stress Sharpe (0.99 vs 0.48).** It's
still not a great number in absolute terms — the 2022 bear crushes
everything — but the result is now close to the single-asset Sharpe line
where retail strategies become plausible (1.0).

### ETH 20% holdout (2022-10-05 → 2024-12-31)

| Metric | Composite | HOMC | H-Vol |
|---|---:|---:|---:|
| Train Sharpe | 0.77 | 0.90 | 0.88 |
| **Holdout Sharpe** | **0.39** | 0.18 | **-0.40** |
| Holdout CAGR | 10.2% | 1.5% | **-22.2%** |

**H-Vol catastrophically fails on ETH.** It's the worst result of any
model on any test, with a negative holdout Sharpe and -22% CAGR. This is
consistent with the Tier-0b finding that ETH has an idiosyncratic regime
structure that neither HOMC nor composite handles well. H-Vol makes it
worse by routing incorrectly during ETH's specific 2022-2024 pattern.

The ETH failure is bounded but real. For ETH specifically, the production
default should remain composite-3×3.

### SOL 20% holdout (2023-10-25 → 2024-12-31)

| Metric | Composite | HOMC | H-Vol |
|---|---:|---:|---:|
| Train Sharpe | 1.43 | **2.05** | 1.58 |
| **Holdout Sharpe** | 1.11 | 1.45 | **2.11** |
| Holdout CAGR | 84.0% | 136.9% | **299.5%** |
| Holdout Max DD | -31.8% | -33.7% | **-23.5%** |
| Holdout Calmar | 2.64 | 4.06 | **12.77** |

On SOL, H-Vol produces the best holdout result ever recorded in this
project: Sharpe 2.11, CAGR 299%, Calmar 12.77. The vol router correctly
keeps HOMC active through SOL's structural bull while dodging the occasional
high-vol pullback. SOL is H-Vol's home asset.

---

## Implementation details

### Bear fix — composite trains on trailing 252 bars

The initial hybrid design used the hybrid's full `train_window` (1000
bars) for both composite and HOMC. That broke composite's bear defense:
composite-3×3 is tuned for a 252-bar window, and training on 1000 bars
averages vol regimes so the 3×3 state grid no longer has a "high-vol"
cell that reliably triggers in crashes.

Fix: the hybrid now fits composite on `observations.iloc[-composite_train_window:]`
(default 252 bars). HOMC still fits on the full 1000-bar slice. This
preserves the exact training environment each component expects while
still letting the regime router see all 1000 bars.

Before fix (HMM-routed, initial results): bear windows 4/5/14 produced
-78.9% / -65.3% / -43.2%. After fix: +7.8% / +7.2% / +12.7%. 85-point
improvement on the 2018-10-03 window alone.

### Active-component delegation

SignalGenerator needs four things from a model: `fitted_`, `predict_next`,
`state_returns_` (vector), and `label`. The hybrid presents these as
delegates to whichever component is currently active — set by the most
recent `predict_state` call.

This relies on the engine's flow being `predict_state → generate →
predict_next`, which is the case everywhere in `BacktestEngine.run()`. No
additional synchronization is needed because the engine is single-threaded.

### Walk-forward correctness

The hybrid passes the same lookahead regression test as the other models
(`tests/test_hybrid.py::test_hybrid_engine_no_lookahead` and
`test_hybrid_vol_routing_no_lookahead`). Equity curves up to bar N are
bit-identical regardless of how much future data is in the input. This
is the cheapest guarantee the engine's timing discipline extends to the
new model class.

---

## Verdict

**H-Vol is promoted to the production default for BacktestConfig** (via
`hybrid_routing_strategy="vol"` as the default). This replaces composite-
3×3 as the recommended single-model configuration for:

- **BTC**: clear improvement over composite AND over HOMC on both random-
  window evaluation (median Sharpe 1.92 vs 1.83) and holdout (2.21 vs
  1.76 on bull holdout, 0.99 vs 0.48 on bear-stress holdout)
- **SOL**: best model in every test (holdout Sharpe 2.11, Calmar 12.77)

For ETH specifically, the **production default should remain composite-3×3**
because H-Vol fails catastrophically on ETH-specific regime structure. We
don't yet have a model that handles ETH reliably; that's a separate
investigation and not a blocker for promoting H-Vol elsewhere.

### What this does not resolve

1. **ETH is still broken.** No model in the project beats buy & hold
   meaningfully on ETH. This is probably a statement about ETH's 2022-
   2024 regime rather than any specific model failure, but it's a
   meaningful gap.
2. **The vol quantile threshold is a hyperparameter.** It defaults to
   0.75 without validation. Lowering it to 0.6 or 0.8 might improve
   bull participation or bear defense respectively. Untested.
3. **The composite-on-trailing-252 design** means the composite component
   retrains every 21 bars on slightly different windows than standalone
   composite (the hybrid's retrain cadence drives both components). This
   is correct — the two match up — but worth documenting if the hybrid's
   retrain_freq is ever changed.

### What to try next (new roadmap items)

- **Tune the vol quantile threshold.** Sweep `--hybrid-vol-quantile`
  over {0.5, 0.6, 0.75, 0.85} on the random-window eval.
- **Add a third component**: an HMM bull-participation backend to route
  to during high-vol bull rallies (the one regime where H-Vol
  underperforms because it defaults to composite in high vol).
- **Fix ETH**: investigate why ETH's regime structure defeats every
  model. Candidate hypothesis: ETH had a post-Merge structural break
  in 2022-09 that's invisible to models trained on pre-Merge data.
- **Adjust routing granularity**: instead of binary vol thresholds,
  use a continuous blending (e.g. position = 0.5 × composite + 0.5 ×
  HOMC when vol is between the 60th and 75th percentile). Might reduce
  the tradeoff between peak return and drawdown.

---

## Reproducibility

```bash
# Full random-window eval with all 5 models
python scripts/random_window_eval.py

# Holdout sweeps
signals backtest sweep BTC-USD --model hybrid --start 2015-01-01 --end 2024-12-31 \
  --states 5 --order 5 --train-window 1000 \
  --buy-grid "10,15,20,25,30" --sell-grid "-10,-15,-20,-25,-30" \
  --stop-grid "0" --no-short --rank-by sharpe --top 10 --holdout-frac 0.2

# With --holdout-frac 0.3 for the bear-stress test
# Replace BTC-USD with ETH-USD or SOL-USD for other assets
```

Default routing strategy is `vol`. Pass `--hybrid-routing-strategy hmm` if
wanting to reproduce the HMM-routed results (not currently plumbed through
the CLI — edit `BacktestConfig.hybrid_routing_strategy` in code or add a
CLI flag if testing HMM routing at scale).

Deterministic seeds throughout. All sweep wall times ~35-45 seconds on a
2024 M-series Mac. Random-window eval ~6 seconds for all 16 windows × 5
models (64 backtests total).

---

⚠ **Disclaimer**: This is an experimental research project. The results
shown above are backtests and historical simulations — they do not
predict future performance. Not financial advice. See the root
[`README.md`](../README.md) for the full disclaimer and risk warning.
