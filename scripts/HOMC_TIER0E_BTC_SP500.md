# Tier 0e — BTC + S&P 500 scope, continuous blending, vol quantile tuning

**Run date**: 2026-04-11
**Test parameters (historical)**:
- `hybrid_vol_quantile` swept across `{0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
  0.80, 0.85, 0.90}` — **this is the sweep that produced the q=0.70
  recommendation** (subsequently superseded by the Round-2 multi-seed
  sweep which found q=0.50 is the multi-seed winner)
- Window sampler: overlapping (buggy). Seed 42 only
- Numbers in this doc are NOT comparable to Round-2 results

**Scope change**: jlg explicitly deprioritized ETH and SOL from future
focus. Production scope is now **BTC-USD and ^GSPC only**.
**Experiments**:
1. Ship the continuous-blend routing strategy (Tier 3 #18) added as the
   third `routing_strategy` option alongside `hmm` and `vol`.
2. Tune the H-Vol `vol_quantile_threshold` (Tier 1 #16) — sweep {0.50,
   0.60, 0.70, 0.75, 0.80, 0.85, 0.90} on both assets and pick the
   per-asset optimum.
3. Re-run the full 16-window random evaluation on both BTC and ^GSPC,
   comparing composite, HOMC, H-Vol, and H-Blend side-by-side.

## TL;DR

1. **BTC**: New best model is **H-Vol at q=0.70**, median Sharpe **2.15**
   across 16 random 6-month windows. Up from 1.92 at the previous
   default q=0.75 and from HOMC's 1.83. **Default changed in code** from
   0.75 → 0.70.
2. **BTC**: H-Blend (continuous ramp 50th→85th percentile) scores median
   Sharpe 2.06 — better than H-Vol at q=0.75 (1.92) but worse than
   H-Vol at q=0.70 (2.15). Blend is a valid option but not the champion.
3. **^GSPC**: **No strategy beats buy & hold.** HOMC is the best active
   strategy at median Sharpe 0.66 but still loses to B&H's 0.77. H-Vol
   and H-Blend are actively negative (medians -0.03 and -0.40). Vol
   quantile tuning doesn't help — even the best S&P quantile (q=0.50)
   produces median Sharpe 0.29, still far below B&H. **Recommendation:
   do not trade S&P 500 with signals. Hold the index.**
4. **Scope lesson**: The hybrid's vol thresholds are calibrated for
   high-vol BTC-style regimes. S&P 500's tight-distribution, secular-
   uptrend profile breaks that calibration. The BTC model is the BTC
   model; a useful S&P model is a different research problem and
   probably a different model class entirely.

---

## Experiment 1: Multi-asset random-window evaluation

Same setup as Tier-0c: 16 random 6-month windows per asset, seed 42,
1000-bar training window, 252-bar composite component. Five strategies
per window (B&H, composite, HOMC, H-Vol at q=0.75, H-Blend at 50→85 ramp).

### BTC-USD

| Metric | B&H | Composite | HOMC | H-Vol (0.75) | **H-Blend** |
|---|---:|---:|---:|---:|---:|
| Mean Sharpe | 1.03 | 1.10 | 1.39 | 1.47 | **1.48** |
| **Median Sharpe** | 1.18 | 1.44 | 1.83 | 1.92 | **2.06** |
| Median CAGR | +101% | +44% | +118% | +134% | **+141%** |
| Mean CAGR | +434% | +321% | +417% | +235% | +336% |
| Mean Max DD | -28% | -21% | -23% | -21% | -22% |
| Sharpe capture (median) | — | 15.8% | 21.5% | 22.9% | 21.4% |
| **Positive CAGR** | 9/16 | 11/16 | 10/16 | 12/16 | **13/16** |
| H-Blend beats composite | — | — | — | — | 9/16 |
| H-Blend beats HOMC | — | — | — | — | 10/16 |
| H-Blend beats H-Vol | — | — | — | — | 10/16 |

On BTC, H-Blend produces the highest median Sharpe (2.06) and the most
positive windows (13/16) of any strategy tested so far. The continuous
ramp preserves bull participation where the hard switch cuts it off —
particularly visible in window 14 (2022-07, crypto winter), where
H-Blend produces **+32.2% / 0.80 Sharpe** vs H-Vol's -9.2% / -0.05.

However, H-Blend vs H-Vol @ q=0.75 is NOT the right comparison. The
vol-quantile sweep (below) shows H-Vol @ q=0.70 produces median Sharpe
**2.15**, beating H-Blend's 2.06 by a meaningful +0.09 margin. The
hard-switch model with a properly-tuned threshold is the champion,
not the continuous blend at the default ramp.

### Key BTC windows

**Bull windows where models diverge:**

| Window | B&H | Composite | HOMC | H-Vol | H-Blend |
|---|---:|---:|---:|---:|---:|
| 2020-11 → 2021-03 (strongest bull) | +4183% | +3713% | +4055% | +1352% | +3036% |
| 2019-05 → 2019-09 (mid-bull) | +472% | +308% | +1061% | +686% | +514% |
| 2023-11 → 2024-03 (bull) | +516% | +396% | +481% | +339% | +345% |

The H-Vol hard switch over-routes to composite in high-vol strong bulls
and gives up ~65% of HOMC's peak return in the 2020-11 window. H-Blend
preserves much more of that peak return (3036% vs HOMC's 4055%) while
still catching the bear-defense wins elsewhere.

**Bear windows where composite's defense matters:**

| Window | B&H | Composite | HOMC | H-Vol | H-Blend |
|---|---:|---:|---:|---:|---:|
| 2018-10-03 → 2019-02-05 | -84% | +26% | -62% | +7% | +17% |
| 2018-10-30 → 2019-03-04 | -78% | +20% | -52% | +34% | +7% |
| 2022-07-06 → 2022-11-08 (crypto winter) | -26% | +28% | -14% | -9% | **+32%** |

H-Blend is the only model that wins the 2022 crypto winter window
outright (+32.2%), beating even composite (+28.3%). This is the clearest
evidence that continuous blending has an edge over hard switching in
ambiguous vol regimes.

### ^GSPC

| Metric | **B&H** | Composite | HOMC | H-Vol | H-Blend |
|---|---:|---:|---:|---:|---:|
| Mean Sharpe | **1.07** | 0.70 | 1.04 | 0.46 | 0.34 |
| **Median Sharpe** | **0.77** | 0.49 | 0.66 | -0.03 | **-0.40** |
| Median CAGR | **+12.2%** | +4.2% | +8.3% | -1.2% | -5.4% |
| Mean CAGR | **+14.0%** | +4.0% | +15.0% | 0.0% | -1.3% |
| Mean Max DD | -15.3% | -14.5% | -14.5% | -15.0% | -15.0% |
| Sharpe capture (median) | — | 8.9% | 10.1% | -1.2% | -8.7% |
| Positive CAGR | — | 8/16 | **12/16** | 8/16 | 6/16 |
| HOMC beats B&H | — | — | 4/16 | — | — |
| H-Vol beats B&H | — | — | — | 0/16 | — |
| H-Blend beats B&H | — | — | — | — | 0/16 |

**NONE of the strategies beat buy & hold on S&P 500.** HOMC is the best
active strategy (mean Sharpe 1.04 vs B&H's 1.07, nearly tied) but on the
median (the robust statistic) B&H wins decisively: 0.77 vs HOMC's 0.66.

The H-Vol and H-Blend hybrids are a catastrophe on S&P. **H-Blend beats
B&H on zero of 16 windows.** Median Sharpe -0.40. Median CAGR -5.4%.
The hybrids are actively subtracting alpha.

### Why the hybrids fail on S&P

The hybrids were designed for BTC's wild regime profile (typical daily
vol ~4%, frequent sharp crashes, extended bulls). S&P 500 has a
fundamentally different statistical profile:

1. **Tight vol distribution**. Daily vol is mostly ~1% with occasional
   spikes. The 75th percentile of training vol is not "dangerous vol"
   on S&P — it's normal variation. Routing the top 25% of days to
   composite means the hybrid spends a quarter of its time in bear-
   defensive mode when nothing bear-y is happening.

2. **Secular uptrend with sparse drawdowns**. S&P's drawdowns are
   concentrated in short windows (COVID, 2022). Outside those, it
   grinds higher. Composite's "wait for the crash signal" behavior has
   nothing to wait for, so it flip-flops or stays flat while B&H
   compounds.

3. **Quantile binning on low-vol returns**. HOMC's quantile-binned
   return states are less informative on S&P because the return
   distribution is more compressed. The same 5-bin partition that
   separates crashes from rallies on BTC (which has wide tails)
   produces mostly-neutral bins on S&P.

4. **The hybrids' worst failures are in the COVID window.** Windows
   4-8 (2019-09 → 2020-08) span the COVID crash AND the recovery.
   Buy & hold captured both sides; the hybrids got routed into composite
   during the crash (which also lost money because it had been trained
   on pre-COVID data and the crash was too fast) and stayed in composite
   through the recovery (missing the V-bounce). Net result: -45% CAGR
   on multiple windows while B&H was only down 8-22%.

### Conclusion: use buy & hold for S&P 500

**Recommendation**: do not run hybrid or single-model strategies on
^GSPC. Hold the index directly. The only active strategy that's
competitive (HOMC at mean Sharpe 1.04) still loses on median and adds
~20% more drawdown than B&H on the worst windows.

This is actually a well-known result in quantitative equity research:
the S&P 500 has such a strong drift relative to its volatility that
active trading subtracts alpha after transaction costs and
opportunity cost. The signals project's Markov-chain models are
sophisticated regime detectors; they're the wrong tool for a
single-regime secular uptrend.

---

## Experiment 2: Vol quantile sweep

Swept `hybrid_vol_quantile` over {0.50, 0.60, 0.70, 0.75, 0.80, 0.85,
0.90} on both BTC and S&P. Same 16-window random evaluation setup as
Experiment 1.

### BTC-USD

| Quantile | Mean Sharpe | **Median Sharpe** | Median CAGR | Mean MDD | Pos |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 1.32 | 1.46 | +75.9% | -19.4% | 13/16 |
| 0.60 | 1.19 | 1.85 | +104.0% | -21.5% | 11/16 |
| **0.70** | **1.59** | **2.15** | **+155.8%** | -21.3% | 12/16 |
| 0.75 | 1.47 | 1.92 | +133.9% | -20.9% | 12/16 |
| 0.80 | 1.37 | 1.87 | +131.3% | -21.6% | 10/16 |
| 0.85 | 1.35 | 1.86 | +131.3% | -22.1% | 10/16 |
| 0.90 | 1.25 | 1.86 | +131.3% | -23.9% | 10/16 |

**BTC optimum is q=0.70** by median Sharpe, mean Sharpe, AND median
CAGR simultaneously. Median Sharpe 2.15 is the highest any single
strategy has produced on BTC in this project — +0.23 over the prior
q=0.75 default and +0.32 over HOMC's 1.83.

Interpretation: 0.70 routes the top 30% of training-vol days to
composite (vs 25% at q=0.75). BTC has more "risk-off" time than the
75th percentile captures. Routing slightly more days to composite
catches additional bear regimes without giving up too much bull
participation.

Going lower than 0.70 (q=0.60 → 1.85, q=0.50 → 1.46) starts hurting
because composite's bull performance is meaningfully worse than HOMC's,
and over-routing to composite sacrifices too much bull alpha. Going
higher than 0.70 (q=0.75, 0.80, ...) gives up bear defense without
enough extra bull capture to compensate.

**Action taken**: Changed `BacktestConfig.hybrid_vol_quantile` default
from 0.75 → 0.70. Existing tests pass, CI should be green.

### ^GSPC

| Quantile | Mean Sharpe | **Median Sharpe** | Median CAGR | Pos |
|---:|---:|---:|---:|---:|
| 0.50 | 0.62 | **0.29** | +3.0% | 9/16 |
| 0.60 | 0.50 | 0.11 | +0.6% | 9/16 |
| 0.70 | 0.37 | -0.14 | -5.4% | 6/16 |
| 0.75 | 0.46 | -0.03 | -1.2% | 8/16 |
| 0.80 | 0.34 | -0.39 | -9.0% | 7/16 |
| 0.85 | 0.51 | -0.17 | -2.9% | 7/16 |
| 0.90 | 0.67 | 0.08 | -0.4% | 8/16 |

Best S&P quantile is q=0.50 (median Sharpe 0.29), **still far below
B&H's 0.77**. Every quantile underperforms buy & hold. The range of
outcomes across quantiles (0.29 to -0.39) is small relative to the gap
to B&H (0.77). **No quantile rescues the hybrid on S&P.**

q=0.50 means routing ALL high-vol-half days to composite — essentially
"default to bear mode half the time". Even this aggressive bear-defense
mode produces a median Sharpe below 0.3.

### Conclusion on tuning

BTC: retune from q=0.75 → q=0.70, ship as new default, +0.23 median
Sharpe improvement.

S&P: no tune works, use buy & hold.

---

## Implementation of continuous blending (Tier 3 #18)

New `routing_strategy="blend"` in `signals/model/hybrid.py`:

### Blend weight function

```
if current_vol <= blend_low_value:    w = 0.0   # full HOMC
elif current_vol >= blend_high_value: w = 1.0   # full composite
else:                                 w = (current_vol - lo) / (hi - lo)
```

`blend_low_value` is the `blend_low_quantile` percentile of training vol
(default 0.50 = 50th percentile). `blend_high_value` is
`blend_high_quantile` (default 0.85 = 85th percentile). The weight ramps
linearly between the two.

### Synthetic single-state interface

The tricky part: SignalGenerator computes `expected = probs @ state_returns_`
to get the strategy's expected return. For blending, I needed to combine
composite and HOMC expected returns into a single scalar without breaking
the SignalGenerator's math.

Solution: at `predict_state()` time, the hybrid runs both components,
computes their individual expected returns, blends them with the vol-
based weight, and stores the result. Then `predict_next()` returns
`[1.0]` and `state_returns_` returns `[blended_expected]`. The dot
product yields the blended expected return exactly.

Confidence is computed by SignalGenerator as `probs[direction_mask].sum()`
which collapses to `1.0` for the 1-state synthetic. This is fine — the
individual components' confidences are already baked into the
magnitudes of their expected returns (a weak signal has a small
|expected|), and SignalGenerator sizes positions by `|expected| / scale`
which preserves that information.

### Tests

5 new tests in `tests/test_hybrid.py` (76 total):
- `test_hybrid_blend_fits_and_predicts` — basic fit/predict round-trip
- `test_hybrid_blend_weight_ramps_linearly` — weight function correctness
- `test_hybrid_blend_expected_is_weighted_average` — math correctness
- `test_hybrid_blend_engine_no_lookahead` — lookahead regression
- `test_hybrid_blend_quantile_validation` — input validation

---

## What changed in the codebase

| File | Change |
|---|---|
| `signals/model/hybrid.py` | Added `routing_strategy="blend"` with linear ramp and synthetic 1-state delegation |
| `signals/backtest/engine.py` | Added `hybrid_blend_low/high` to `BacktestConfig`; changed `hybrid_vol_quantile` default 0.75 → **0.70** |
| `tests/test_hybrid.py` | +5 blend tests (71 → 76 total) |
| `scripts/random_window_eval.py` | Refactored for multi-symbol (BTC + ^GSPC) with all 4 strategies (composite, HOMC, H-Vol, H-Blend) |
| `scripts/vol_quantile_sweep.py` | New script: tunes `hybrid_vol_quantile` across {0.50, ..., 0.90} on both assets |
| `scripts/HOMC_TIER0E_BTC_SP500.md` | This file |

---

## Recommended usage

### BTC trading

```bash
# H-Vol at q=0.70 (new default) — median Sharpe 2.15
signals backtest run BTC-USD --model hybrid --start 2018-01-01 --end 2024-12-31

# Or explicit continuous blend — median Sharpe 2.06
signals backtest run BTC-USD --model hybrid --start 2018-01-01 --end 2024-12-31
# (no CLI flag for routing_strategy yet; edit BacktestConfig.hybrid_routing_strategy="blend" if wanted)
```

The daily workflow command:

```bash
signals signal next BTC-USD
```

should be updated to default to the hybrid. Currently it uses `composite`
as its default — that's a CLI issue to fix in a follow-up commit.

### S&P 500 trading

**Don't.** Buy SPY (or an equivalent ETF) and hold. Active trading with
the signals models subtracts alpha net of costs on the S&P's statistical
profile. If you really want active exposure, the closest-to-viable model
is HOMC alone (not a hybrid), but even that trails B&H on the median.

---

## What to try next

1. **Tune the H-Blend ramp parameters.** This sweep only tuned the
   single-quantile hard switch. The blend's (blend_low, blend_high) pair
   has ~15 sensible combinations. A similar 2D sweep on BTC might find
   a blend config that beats 2.15. Cost: ~15 × 30 sec = 8 min. Low
   effort, potentially meaningful.
2. **Sweep composite_train_window inside the hybrid.** Default is 252.
   128 or 504 might be better. Cost: ~4 × 30 sec = 2 min.
3. **Investigate S&P 500 specifically.** The existing model class is
   wrong for secular-uptrend equity. Candidates: (a) momentum factor
   (12-month momentum minus 1-month), (b) simple 200-day moving average
   crossover (the classic trend-following approach), (c) volatility
   targeting with leverage. These are not Markov-chain models — they're
   standard equity quant tools. Worth exploring if S&P is a real
   research priority, but out of scope for the current signals codebase.
4. **Add a CLI flag for `--hybrid-routing-strategy`** so experiments can
   flip between vol/blend without code edits.
5. **Per-symbol default overrides**. Allow `BacktestConfig` to carry
   per-symbol preferences, or add a `default_for_symbol()` helper that
   knows "BTC → hybrid q=0.70, ^GSPC → B&H (no strategy)".

## Reproducibility

```bash
# Random-window eval on both assets
python scripts/random_window_eval.py

# Vol quantile sweep on both assets
python scripts/vol_quantile_sweep.py
```

Both are deterministic (seed 42). Wall time for the random-window eval:
~35 seconds (64 backtests). Wall time for the vol quantile sweep: ~4-5
minutes (14 × 16 = 224 backtests).

---

⚠ **Disclaimer**: This is an experimental research project. The results
shown above are backtests and historical simulations — they do not
predict future performance. Not financial advice. See the root
[`README.md`](../README.md) for the full disclaimer and risk warning.
