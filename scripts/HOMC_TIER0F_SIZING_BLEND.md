# Tier 0f — Sizing sweep and blend ramp sweep

**Run date**: 2026-04-11
**Scope**: BTC-USD only. S&P 500 was excluded because Tier 0e showed no
strategy in the project beats buy & hold on S&P regardless of parameter
tuning.
**Experiments**:
1. **Tier-1 #5 — Asymmetric / aggressive position sizing.** Sweep
   `(target_scale_bps, max_long)` over a 4×4 grid to find the best
   risk/return tradeoff for the retuned H-Vol default.
2. **IMPROVEMENTS.md #16a — Tune H-Blend ramp parameters.** 2D sweep
   over `(blend_low_quantile, blend_high_quantile)` to find a
   continuous-blend config that might beat H-Vol @ q=0.70.

## TL;DR

1. **Sizing sweep**: `target_scale_bps` is **effectively inert** for
   HOMC's expected-return magnitudes. Only `max_long` moves the needle,
   and it's a clean risk/return tradeoff — **Sharpe is essentially flat
   across the tested leverage range** (2.09 to 2.16 median Sharpe from
   max_long 1.0 to 2.0), while CAGR scales roughly linearly with leverage
   (156% → 480%) and drawdown scales proportionally (-21% → -38%). No
   default change recommended; pick your point on the curve based on
   drawdown tolerance.
2. **Blend ramp sweep**: The best H-Blend pair `(low=0.40, high=0.90)`
   scores median Sharpe **2.07**, which is 0.08 below H-Vol @ q=0.70's
   **2.15**. **H-Blend does NOT beat the hard switch.** All 16 tested
   blend pairs fall between 1.82 and 2.07. H-Vol @ q=0.70 remains the
   BTC default. H-Blend stays as an available routing option but is not
   recommended.

**Net outcome**: No default changes. The current BTC configuration
(H-Vol @ q=0.70, max_long=1.0, target_scale_bps=20) is at the Sharpe
plateau. Further Sharpe improvements will require structural changes,
not parameter tuning.

---

## Experiment 1: Sizing sweep

### Setup

Ran 16 (scale, max_long) combinations through the 16-window random BTC
evaluation (seed 42). 256 backtests total. Hyperparameter grid:

- `target_scale_bps` ∈ {5, 10, 15, 20}
- `max_long` ∈ {1.0, 1.25, 1.5, 2.0}

All other params held to the retuned H-Vol defaults:
`model_type=hybrid, train_window=1000, hybrid_routing_strategy=vol,
hybrid_vol_quantile=0.70, n_states=5, order=5, laplace_alpha=0.01`.

### Prerequisite code change

Before this sweep, the SignalGenerator used `min(1.0, |expected|/scale)`
as the magnitude formula, which hardcoded a cap at 1.0 regardless of
`max_long`. That meant raising max_long above 1.0 had **no effect** —
the magnitude was already capped before the max_long clip could fire.

Fix (committed same session): changed the formula to
`magnitude = |expected| / scale` (uncapped), letting `max_long` be the
real leverage ceiling. Backward compatible — with `max_long=1.0` (the
existing default), behavior is unchanged because the final
`min(raw_target, max_long)` clips any magnitude above 1.0 to 1.0.

### Results

| max_long | target_scale | Mean Sharpe | **Median Sharpe** | Median CAGR | Mean MDD | Pos |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 20 (default) | 1.59 | **2.15** | +155.8% | -21.3% | 12/16 |
| 1.25 | 20 | 1.59 | 2.15 | +216.3% | -26.1% | 12/16 |
| **1.50** | 5/10/15 | 1.59 | **2.16** | **+287.8%** | -30.5% | 12/16 |
| 1.50 | 20 | 1.59 | 2.07 | +287.8% | -30.2% | 12/16 |
| 2.00 | 5/10/15 | 1.61 | 2.13 | +480.3% | -38.5% | 12/16 |
| 2.00 | 20 | 1.61 | 2.09 | +384.5% | -35.8% | 12/16 |

Full 16-row table in `/tmp/sizing_sweep.txt` (regeneratable by running
`python scripts/sizing_sweep.py`).

### Key observations

**1. `target_scale_bps` is inert in this regime.**

Rows with the same `max_long` produce identical median Sharpe regardless
of `target_scale_bps`, for scales 5, 10, 15 (and often 20 too). This is
because the hybrid's expected returns are large enough that
`|expected| / scale >> max_long` for every scale in {5, 10, 15, 20} —
the magnitude always saturates at the `max_long` clip, so the scale
denominator drops out of the answer.

For scales below ~5 or expected returns above ~100 bps, scale would
start to matter. Within this regime, it doesn't.

At scale=20 with max_long=1.5 or 2.0, a few windows have
`|expected|/scale < max_long`, and the slightly different sizing
produces a marginally different result (2.07 and 2.09 median Sharpe
respectively). These are noise-level differences.

**2. `max_long` is a clean risk/return dial.**

Fixing scale=5 and varying max_long (rows 1, 2, 3, 4 of the raw output):

| max_long | Mean Sharpe | Median CAGR | Mean MDD | Return/MDD ratio |
|---:|---:|---:|---:|---:|
| 1.00 | 1.59 | +155.8% | -21.3% | 7.3 |
| 1.25 | 1.59 | +216.3% | -26.1% | 8.3 |
| 1.50 | 1.59 | +287.8% | -30.5% | 9.4 |
| 2.00 | 1.61 | +480.3% | -38.5% | 12.5 |

Sharpe is essentially flat (1.59-1.61). CAGR scales proportionally with
leverage — 1.25× leverage gives ~1.4× CAGR, 1.5× gives ~1.9× CAGR,
2.0× gives ~3.1× CAGR (more than linear because of compounding). MDD
scales nearly 1:1 with leverage (expected for a long-only strategy).

The `return/MDD` ratio *improves* with leverage, which is a sign that
the underlying alpha is real — scaling up extracts more of it per unit
of drawdown pain. But Sharpe stays flat because volatility scales with
returns.

**3. Sharpe is at a plateau.**

The best median Sharpe across the entire grid is 2.16, achieved at
`(scale=5/10/15, max_long=1.5)`. Only +0.01 above the baseline 2.15.
This is noise. **The model's Sharpe is at a plateau around 2.15** for
BTC on this 16-window random evaluation. Any further improvement
requires a structural change, not parameter tuning.

### Recommendations

**No default change.** The current `target_scale_bps=20, max_long=1.0`
is fine — it's at the Sharpe plateau with the most conservative
drawdown profile. But the sweep reveals a **pick-your-point-on-the-
curve** choice:

- **Conservative** (`max_long=1.0`): 156% median CAGR, -21% mean MDD, Sharpe 2.15.
- **Moderate** (`max_long=1.5`): **288% median CAGR**, -30% mean MDD, Sharpe 2.16. ~1.85× the return of conservative for a 9pp higher drawdown.
- **Aggressive** (`max_long=2.0`): 480% median CAGR, -38% mean MDD, Sharpe 2.13. ~3× the return of conservative for a 17pp higher drawdown.

For jlg's production use case, `max_long=1.5` looks like the most
attractive single point — it nearly doubles returns at the same Sharpe
and preserves the "not-catastrophic" drawdown floor. But this is a risk
tolerance decision, not a methodology decision. I'm not changing the
default; document and let the user choose.

---

## Experiment 2: H-Blend ramp sweep

### Setup

Ran 16 (blend_low, blend_high) pairs through the 16-window random BTC
evaluation (seed 42). All 16 combinations in the grid had `low < high`
so none were skipped. 256 backtests total. Grid:

- `blend_low_quantile` ∈ {0.30, 0.40, 0.50, 0.60}
- `blend_high_quantile` ∈ {0.70, 0.80, 0.85, 0.90}

### Results (ranked by median Sharpe)

| Rank | Low | High | Mean Sharpe | **Median Sharpe** | Median CAGR | Mean MDD | Pos |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.40 | 0.90 | 1.37 | **2.07** | +138.0% | -22.0% | 11/16 |
| 2 | 0.50 | 0.85 (default) | 1.48 | 2.06 | +141.2% | -21.6% | 13/16 |
| 3 | 0.60 | 0.85 | 1.48 | 2.05 | +143.7% | -22.0% | 11/16 |
| 4 | 0.60 | 0.80 | 1.53 | 2.05 | +143.7% | -21.4% | 13/16 |
| 5 | 0.50 | 0.80 | 1.41 | 2.03 | +143.7% | -20.8% | 11/16 |
| 6 | 0.40 | 0.85 | 1.32 | 2.03 | +143.7% | -21.7% | 11/16 |
| 7 | 0.30 | 0.85 | 1.41 | 1.98 | +130.7% | -21.1% | 12/16 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 16 | 0.60 | 0.90 | 1.42 | 1.82 | +130.5% | -22.3% | 10/16 |

Reference points:
- **H-Vol @ q=0.70 median Sharpe: 2.15**
- H-Blend default (0.50, 0.85) median Sharpe: 2.06

### Interpretation

**No blend pair beats H-Vol @ q=0.70.** The best blend (0.40, 0.90)
scores 2.07, which is 0.08 below the hard-switch baseline of 2.15.

The full range of blend results spans 1.82 to 2.07 — a narrow band
(0.25 Sharpe). Blend is not very sensitive to the specific (low, high)
pair chosen within reasonable bounds. The best pair is NOT the default
(0.50, 0.85) but a very similar (0.40, 0.90). The difference is within
noise.

**Why doesn't the blend beat hard switching?**

Hypothesis: the blend introduces "intermediate" positions that are
neither fully composite nor fully HOMC. In a clear bear regime where
composite is the right model, a 50/50 blend gives you 50% HOMC
exposure — which is bad because HOMC loses in bears. In a clear bull
regime where HOMC is the right model, a 50/50 blend gives you 50%
composite — which is bad because composite under-participates in bulls.

The hard switch at q=0.70 picks the right model at the right time;
the blend averages over it. Averaging is theoretically appealing for
smoothing out regime-detection errors, but the empirical result is
that the underlying regimes are clean enough that decisive selection
beats soft blending.

A SECOND hypothesis (not tested): blend might be best when regime
classification is noisy. On BTC with a 1000-bar training window and
well-separated vol quantiles, the regime signal is reliable, so
averaging doesn't help. On a noisier asset or a shorter training
window, blend might win — but that's hypothetical until tested.

### Recommendation

**Keep H-Vol @ q=0.70 as the BTC default.** Do not switch to H-Blend.

H-Blend stays as an available `routing_strategy` option for
experimentation — a user who wants smoother regime transitions (for
reasons like trading friction, execution, or forward-walked robustness)
can still opt in. But the default should be the hard switch, which has
a clear empirical edge on the median-Sharpe metric.

---

## Combined verdict

Both sweeps produce the same high-level message: **the BTC model is at
a Sharpe plateau around 2.15, and parameter tuning within the current
model structure cannot push it higher.**

| Change tested | Median Sharpe | Delta vs baseline |
|---|---:|---:|
| **Baseline** (H-Vol @ q=0.70, scale=20, max_long=1.0) | **2.15** | — |
| Sizing: max_long=1.5 | 2.16 | +0.01 (noise) |
| Sizing: max_long=2.0 | 2.13 | -0.02 |
| Blend: best pair (0.40, 0.90) | 2.07 | -0.08 |
| Blend: default pair (0.50, 0.85) | 2.06 | -0.09 |

Nothing in the sweep beats the baseline by a meaningful margin. The
only genuine win is the leverage knob — `max_long` is a clean
risk/return dial if you want higher CAGR at the cost of drawdown.

### Where to find improvements beyond 2.15

Parameter tuning within the current model is exhausted. Genuine
Sharpe improvements will require **structural** changes:

1. **Different state encoders.** The composite uses a 3×3 quantile grid
   on (return, volatility). Alternatives:
   - Absolute-width bins (Nascimento et al. style) — tested on HOMC
     and failed, but might work for composite.
   - Added features: volume, VIX, yield-curve slope, BTC dominance.
   - Variable-width bins that adapt per regime.

2. **Different regime signals.** Current H-Vol routes on 20-day
   realized vol. Alternatives:
   - Realized variance of log returns (Welford-style)
   - GARCH(1,1) forecast volatility
   - Implied volatility if available (BVOL index)
   - Correlation structure (BTC-SPX correlation as a "risk-on/risk-off"
     proxy)

3. **Portfolio construction.** Currently single-asset. Multi-asset
   (BTC + SPX + treasuries) with risk parity or min-vol allocation
   might produce a higher portfolio Sharpe than any single strategy.
   This is not about picking a better BTC signal; it's about combining
   the BTC signal with other exposures to improve the aggregate.

4. **Execution quality.** Fees and slippage are modeled at 5 bps each.
   If real execution is cheaper (e.g. limit orders with rebates on
   major exchanges), the strategy's real-world Sharpe could be higher
   than the backtest. Conversely, if slippage is higher at production
   size, it's lower. Test by replaying against real exchange tick data.

5. **Time-varying hyperparameters.** The current defaults are fixed.
   A meta-learner that adjusts `hybrid_vol_quantile` based on realized
   regime stability might extract more — but the added complexity is
   only justified if there's evidence the optimum is time-varying.

None of these are on the current roadmap. They're listed here so
future work has a direction when parameter tuning runs out.

---

## Code changes

| File | Change |
|---|---|
| `signals/model/signals.py` | Removed the hardcoded 1.0 cap on magnitude. Formula is now `magnitude = |expected| / scale`, clipped by `max_long` at the signed-target stage. Backward compatible with `max_long=1.0`. |
| `scripts/sizing_sweep.py` | New — 4×4 sweep over (scale, max_long) for H-Vol @ q=0.70 on BTC. |
| `scripts/blend_ramp_sweep.py` | New — 4×4 sweep over (low, high) for H-Blend on BTC. |
| `scripts/HOMC_TIER0F_SIZING_BLEND.md` | This file. |

Tests: all 76 still passing after the sizing formula change. CI green
on the prior commit; this commit should stay green because the formula
change is exercised by the existing backtest and lookahead tests.

---

## Reproducibility

```bash
python scripts/sizing_sweep.py       # ~4 min, 256 backtests
python scripts/blend_ramp_sweep.py   # ~5 min, 256 backtests
```

Both deterministic (seed 42). Wall time on a 2024 M-series Mac as
noted. Regenerates the raw tables in the per-script output.

## What to try next

Parameter tuning is exhausted. Real follow-up work should target the
structural improvements listed above. The top three in order of
evidence-per-cost:

1. **Multi-asset portfolio** — BTC + SPX (buy & hold) + treasuries at
   risk parity. This directly uses the finding that BTC's active
   strategy is Sharpe 2.15 and SPX is best as B&H at Sharpe ~0.8 — the
   portfolio combination might have a higher Sharpe than either alone
   by diversification.
2. **Time-varying `hybrid_vol_quantile`** — sweep the quantile in
   real-time windows (e.g. every 5 years) and check whether the
   optimum shifts over time. If it does, a time-varying default is
   warranted. If not, the fixed 0.70 is truly optimal.
3. **Add macro features to composite** — VIX, yield curve slope, BTC
   dominance as additional features alongside return and volatility.
   Requires fetching macro data and extending the composite encoder
   to handle more than 2 features.
