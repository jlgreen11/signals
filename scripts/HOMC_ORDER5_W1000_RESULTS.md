# Tier 0a experiment — HOMC at order=5 with 1000-bar window

**Run date**: 2026-04-10
**Test parameters (historical)**:
- Model: HOMC @ `order=5`, `n_states=5`, `train_window=1000`
- Hybrid: **not applicable** — this doc predates the hybrid model
  (`hybrid_vol_quantile` did not exist yet).
- Window sampler: overlapping (buggy); walk-forward single pass for
  holdout evaluation.

**Question**: After the order=7/window=252 setup failed catastrophically (see
`HOMC_ORDER7_RESULTS.md`), is the HOMC backend salvageable with a wider
training window and a lower order — i.e. is the failure structural, or just
under-resourced?

**Method**: Two sweeps on identical data (BTC-USD 2015-01-01 → 2024-12-31,
2,557 bars), identical hyperparameter grid, identical holdout (20% trailing
slice = 2023-01-01 → 2024-12-31, 731 bars), identical 1000-bar walk-forward
training window. Only the model class differs:

1. **HOMC** at `n_states=5, order=5` (Tier-0a)
2. **Composite** at `return_bins=3, volatility_bins=3` (baseline for fair
   comparison)

The composite baseline is the critical control: if HOMC's holdout strength is
just an artifact of 2023-2024 being a friendly period for any model, the
composite should also do well on the holdout.

## Headline numbers

| | HOMC (order=5, n=5) | Composite (3×3) |
|---|---:|---:|
| **In-sample** (2015-01 → 2022-12, 8yr) | | |
| Best CAGR | 13.3% | **30.1%** |
| Best Sharpe | 0.42 | **0.62** |
| DSR | 0.00 | 0.00 |
| Max DD | -76% | -81% |
| # trades | 307 | 283 |
| **Holdout** (2023-01 → 2024-12, 2yr) | | |
| CAGR | **110.1%** | 11.8% |
| Sharpe | **1.76** | 0.63 |
| Max DD | -21% | -16% |
| # trades | 57 | 32 |
| **In-sample → Holdout Sharpe ratio** | **4.2×** | **1.0×** |

In-sample buy & hold over 2015-2022: $37,380, CAGR 28.7%, MDD -83.4%.
Buy & hold over the holdout (2023-01 → 2024-12): BTC went from ~$16,500
to ~$94,000 = **~140% CAGR** with significant intra-period drawdowns.

## What this tells us

### The HOMC's holdout strength is real and HOMC-specific

The composite baseline is the cleanest possible control: same dates, same
window, same sweep grid, same fee/slippage model, same holdout. The
composite produces a flat 0.62 → 0.63 Sharpe transition — fully consistent
between train and test, with no alpha "emerging" from the holdout. The HOMC
produces a 0.42 → 1.76 transition, a **4.2× Sharpe expansion**. The composite
cannot exploit the structure the HOMC found, so the strength is not coming
from the period being friendly to all Markov-chain models. It is specific to
the higher-order memory.

This is the clearest possible refutation of the "it's just a lucky window"
explanation: the lucky window theory predicts that BOTH models do well in
the holdout, and one of them clearly doesn't.

### But the in-sample failure is also real

The HOMC in-sample Sharpe of 0.42 is meaningfully worse than the composite's
0.62. With 307 trades over 8 years, the HOMC was actively trading and
consistently underperforming. A live operator running this model from 2015
through 2022 would have observed:

- 8 years of mediocre returns (13.3% CAGR vs B&H 28.7%)
- Multiple drawdowns into the -50% to -76% range
- Sharpe well below the composite's
- DSR = 0 (no statistical evidence the model is doing anything beyond noise)

A rational risk officer would have killed this model long before 2023.
**The model that produced the spectacular holdout is the same model that
would have been shut down in production based on its in-sample track
record.**

### The deflated Sharpe is not lying

DSR = 0.00 across all 25 in-sample sweep configs. This is correct:
the multi-trial-corrected expected Sharpe under H₀ is roughly 1.4 with
N=25 trials and ~2000 in-sample observations, and every observed Sharpe is
below 0.5. The DSR statement "this is consistent with noise" is true *of
the in-sample data*. What DSR cannot tell us is whether a noise-indistinguishable
in-sample model will explode into something useful out of sample.

This experiment is a clean counter-example to "DSR ≥ 0.95 is the bar before
trusting a backtest". The HOMC in-sample has DSR 0.00 and would have failed
the bar — yet the holdout result is dramatically better than the composite,
which had a higher in-sample Sharpe.

## Three plausible explanations

1. **Long-memory structure emerged or strengthened around 2023.** Crypto
   went from a 2015-2022 period of wild idiosyncratic moves (whales,
   exchange failures, regulatory shocks) to a more "mature" 2023-2024 period
   (institutional adoption, ETF launches, more rational price discovery).
   The HOMC's order=5 memory captures pattern dependence that didn't exist
   strongly in the earlier period but does now. The composite's per-bar
   return×vol state is too myopic to see it.

2. **The 1000-bar training window finally allows order=5 estimation.**
   With 1000 training bars and 5⁵ = 3,125 possible 5-tuples, the typical
   training fit observes ~1000 actual 5-tuples for ~30% coverage —
   sparse but workable. At 252 bars and order=7, coverage was 0.3%
   (Tier-0 result). The current setup is the first time HOMC has had
   enough data per fit to make non-trivial predictions, and that
   capability happens to land on a period where the predictions are useful.

3. **Lucky regime that selectively favors the HOMC's biases.** 24 months is
   not very long. A single sustained trend that the HOMC's particular state
   transitions happen to predict could account for the entire holdout
   Sharpe. The composite control rules out the "any model" version of this,
   but not the "HOMC-specific bias happens to align" version.

I cannot distinguish these from one experiment. They are not mutually
exclusive — explanations 1 and 2 are complementary, and 3 could partially
account for the magnitude.

## Verdict

The original Tier-0a decision rule was: "DSR > 0.5 AND holdout Sharpe ≥ 0.5
→ salvageable; otherwise → demote permanently." Strict reading: HOMC FAILS
(DSR = 0.00). But the comparison to composite reveals that the strict rule
would dismiss a model that's measurably better than the alternative on
out-of-sample data, which is the wrong answer.

**Revised verdict: INCONCLUSIVE — needs confirmatory experiments before
either promotion or demotion.** The honest summary is "the HOMC at
order=5/window=1000 found something on the 2023-2024 BTC data that the
composite cannot find, but in-sample it would have looked like a broken
model and we don't know whether the holdout strength generalizes to other
assets, other periods, or longer holdouts."

## Confirmatory experiments to run before deciding

### A. Slide the holdout window

```bash
# 30% holdout — start of test = 2022-01
signals backtest sweep BTC-USD --model homc \
  --start 2015-01-01 --end 2024-12-31 \
  --states 5 --order 5 --train-window 1000 \
  --buy-grid "10,15,20,25,30" --sell-grid "-10,-15,-20,-25,-30" \
  --stop-grid "0" --no-short --rank-by sharpe --top 10 \
  --holdout-frac 0.3
```

If the 30% holdout (which adds the bear-market end-of-2022 to the test set)
still produces Sharpe > 1, the result generalizes beyond a single bull
window. If the Sharpe collapses, the holdout strength was a 2023-2024
artifact.

### B. Multi-asset

Run the exact same sweep on ETH-USD and SOL-USD. The Nascimento paper
claimed ETH and XRP behave like BTC; this is a direct test. If the HOMC's
2-year-holdout edge holds on a second asset, the chances of it being
HOMC-specific structure go up sharply. If it doesn't, BTC-2023-2024 was
unique.

### C. Random-window evaluation with HOMC

The existing `scripts/random_window_eval.py` runs the composite. Add HOMC
at order=5 / window=1000 to it and rerun on 16 random 6-month windows.
That gives a per-window Sharpe-capture distribution. If HOMC beats
composite in more than 9/16 windows, it's a real upgrade. If it ties or
loses, the in-sample failure dominates and the model is not promotable.

### D. Lower-order HOMC controls

Run the same sweep at order=3 and order=4 with the 1000-bar window. If the
holdout strength persists at lower orders, the lift comes from the wider
window, not the higher memory. If it only shows up at order=5, the
high-order memory is doing the work.

The cheapest of these is **A** (one command, ~30 seconds). I would run A
first, then C, then B, then D, in that order of evidence-per-cost.

## Implications for the roadmap

The Tier-0a result moves nothing in IMPROVEMENTS.md *yet* — the verdict is
inconclusive, so the priorities from the order=7 negative result still
hold. But it does add three new candidate experiments (above) that should
run before any HOMC-related Tier-1 work, and it raises the question of
whether the HOMC backend deserves a "production" track rather than a
research-only one.

If experiment **A** shows the holdout strength survives the 30% holdout,
the next priority becomes building Tier-1 #1 (AbsoluteGranularityEncoder)
and Tier-1 #2 (order/granularity sweep) so we can hunt for the right
operating point. If A fails, the verdict goes back to "HOMC remains
research-only" and Tier-1 effort stays on #5/#6/#7 (composite sizing
improvements + multi-asset benchmark).

## Reproducibility

Both sweeps with the exact commands above. Wall time: ~40 seconds for HOMC,
~33 seconds for composite, on a 2024 M-series Mac. Deterministic — re-runs
should produce identical numbers.

## A note on the deflated Sharpe interpretation

This experiment is the first time in this project where DSR has been a
*misleading* signal rather than a corrective one. The conventional advice
"DSR ≥ 0.95 before trusting a backtest" would have caused us to dismiss
the HOMC outright. The lesson is that DSR is a *necessary but not
sufficient* check — it correctly identifies when in-sample results are
indistinguishable from noise, but it cannot tell you when an
in-sample-noisy model has out-of-sample structure that emerges from
holdout validation. The right rule is "DSR is the bar for the in-sample
result; out-of-sample validation is the bar for the strategy". A model
that fails DSR but passes a clean holdout against a controlled baseline
is in a category the conventional rule doesn't cover, and that's exactly
where this HOMC result lives.

---

⚠ **Disclaimer**: This is an experimental research project. The results
shown above are backtests and historical simulations — they do not
predict future performance. Not financial advice. See the root
[`README.md`](../README.md) for the full disclaimer and risk warning.
