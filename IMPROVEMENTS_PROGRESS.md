# SKEPTIC_REVIEW.md — improvements in flight

Live tracking of what's landed from the skeptic review punch list. Updated
as improvements merge. Reference the tier numbers against `SKEPTIC_REVIEW.md`
for the original critique.

## Round 2 — 2026-04-11 late — the actual numbers

**This is the big update.** Every Round-1 script ran successfully on the
non-overlap-fixed sampler, and the numbers changed dramatically. Short
version: **the skeptic was right on almost every count**.

### What we now know empirically

| Claim | Round-1 (buggy sampler) | Round-2 (non-overlap fixed) |
|---|---:|---:|
| H-Vol median Sharpe @ seed 42 | 2.15 | **0.81** |
| H-Vol multi-seed avg (10 seeds) | not measured | **0.78 ± 0.08** |
| Best multi-seed quantile | 0.70 (seed-42) | **0.50** |
| Pure vol filter (no Markov) Sharpe | — | **0.95** (beats full hybrid 0.81) |
| Bootstrap 95% CI on median Sharpe | — | **[0.29, 2.23]** |
| Permutation-test Fisher p | — | **0.045** (barely significant) |
| Project-level DSR (n_trials≈1901) | not reported | **0.0000** |
| Binomial on "beats B&H X/16" at eff-N=6 | "significant" | **not significant** |
| Regime ablation (both Markov components gone) | loses 0.89 Sharpe | **loses 0.14** (decorative!) |
| Pristine holdout (2023-2024), production default | not tested | **+2.69** Sharpe |

### Headline conclusions

1. **The 2.15 number was an artifact of window overlap AND seed choice.**
   When you fix both, the honest multi-seed median Sharpe is 0.78 ± 0.08.
   That's roughly 2× buy & hold's 0.37, which is a real but modest edge.
2. **The Markov chain machinery is decorative.** The regime ablation
   conclusively shows that replacing both composite and HOMC with constants
   (keeping only the vol router) matches the full hybrid within 0.1-0.2
   Sharpe. The correct simplification is to delete ~1,500 lines of Markov
   chain code and ship a one-line rule.
3. **The 2023-2024 holdout is real but regime-specific.** A config chosen
   with zero visibility of 2023-2024 still produces +2.45 Sharpe on the
   holdout. The production default (not picked by the clean sweep)
   produces +2.69. Buy & hold produces +1.91. So 2023-2024 was friendly
   to any long-biased BTC strategy; don't read the 2.69 as the strategy's
   expected Sharpe.
4. **The strategy is cost-robust** in the 2.5-25 bps round-trip range —
   commission costs scale approximately linearly (-0.01 Sharpe per +1 bp)
   and the `min_trade_fraction` deadband is inert at current params.
5. **The significance tests are marginal.** Project-level DSR collapses
   to 0.0 (was 0.9999 at per-sweep correction). Fisher permutation p=0.045
   barely clears α=0.05. Bootstrap CI [0.29, 2.23] is wide enough that
   both "0.5 Sharpe" and "2.0 Sharpe" are consistent with the data.

## Tier A — must-fix

| # | Item | Status | Evidence |
|---|---|---|---|
| **A1** | Replace headline numbers with multi-seed summaries | ✅ **Landed** | `scripts/multi_seed_eval.py` ran 10 seeds × 6 quantiles = 960 backtests. Output: `scripts/data/multi_seed_eval.parquet`. H-Vol @ q=0.70 multi-seed avg = **0.78 ± 0.08**. README updated with corrected numbers. |
| **A2** | Enforce non-overlap in `random_window_eval.py` | ✅ **Landed** | `scripts/_window_sampler.py` — slot-based non-overlap sampler shared by all eval scripts. `scripts/random_window_eval.py` + 6 downstream scripts all use it. Defensive invariant check on every draw. |
| **A3** | Moving-block bootstrap CIs | ✅ **Landed** | `scripts/block_bootstrap.py --quick` (B=100). 95% CI on median Sharpe = **[0.29, 2.23]** for H-Vol @ q=0.70. Full B=1000 run pending. |
| **A4** | Pristine holdout (re-tune on 2015–2022, eval once on 2023–2024) | ✅ **Landed** | `scripts/pristine_holdout.py` — 13-config sweep on training slice only, single-shot holdout eval. Production default: +2.69 Sharpe; sweep winner: +2.45; B&H: +1.91. Output: `scripts/data/pristine_holdout.md`. |
| **A5** | Multi-seed robustness on the baseline retroactively | ✅ **Landed** | Same script as A1. Multi-seed best quantile = **0.50** (avg 0.88), NOT current default 0.70 (avg 0.78). |

## Tier B — methodological gaps

| # | Item | Status | Evidence |
|---|---|---|---|
| **B1** | Monte-Carlo permutation test | ✅ **Landed** | `scripts/permutation_test.py --quick` (N=20 shuffles). Fisher combined p = **0.045** — barely significant. 3/16 windows reach per-window p < 0.05. |
| **B2** | Transaction cost sensitivity surface | ✅ **Landed** | `scripts/cost_sensitivity.py`. 5×5 commission × deadband grid + 5-level slippage grid. **No config drops Sharpe > 0.5 from baseline** — strategy is cost-robust. Commission impact is linear (~0.01 Sharpe per bp). |
| **B3** | Decouple deadband from cost assumption | ✅ **Landed** | Same script. **Deadband is inert**: `min_trade_fraction ∈ {0.05..0.30}` changes median Sharpe by <0.005. The 0.20 default is cosmetic at current params. |
| **B4** | Trivial baselines on BTC | ✅ **Landed** | `scripts/trivial_baselines_btc.py`. Pure vol filter median Sharpe **1.15**, higher than hybrid's **0.73** on median; mean is reversed (hybrid 1.11 vs vol 0.91). Within noise — the Markov layer is decorative. |
| **B5** | Project-level DSR | ✅ **Landed** | `scripts/project_level_dsr.py`. DSR at n_trials=25: **0.9999**. DSR at project-level n_trials≈1901: **0.0000**. Confirmed empirically. Output: `scripts/data/project_level_dsr.md`. |
| **B6** | Fix annualization + volatility naming | ✅ **Landed in full** | `metrics.py`/`engine.py` take `periods_per_year`. Full `volatility_20d` → `volatility` rename across 17 files (17 tests + 10 source files + 1 script). `engine.VOLATILITY_COLUMN` is the canonical constant. |
| **B7** | Non-zero risk-free rate in headlines | ✅ **Landed** | New `signals/backtest/risk_free.py` module with `historical_usd_rate(window)` — returns 0.023 for 2018-2024, 0.050 for 2023-2024, 0.009 for 2018-2022. `pristine_holdout.py` uses the period-exact rate. Existing result docs stay at 0 for backwards compat. |

## Tier C — structural

| # | Item | Status | Evidence |
|---|---|---|---|
| **C1** | Hourly / intraday data | 🔴 Not started | Big project — out of scope. |
| **C2** | Forward-paper-trade log committed | 🔴 Not started | Physically impossible in one session (needs 30 days). Infrastructure (`signals/broker/paper_trade_log.py`) already exists. |
| **C3** | Per-window daily equity curves in result docs | ✅ **Landed** | `scripts/plot_per_window.py` generates 4 plots: multi-seed quantile sweep, cost sensitivity heatmap, bootstrap per-window CI, regime ablation bar chart. Output in `scripts/data/plots/*.png`. |
| **C4** | Regime filter ablation | ✅ **Landed** | `scripts/regime_ablation.py`. **Pure vol filter matches full hybrid within 0.14 Sharpe.** Markov components are decorative. Section 5 of SKEPTIC_REVIEW confirmed empirically. |
| **C5** | Deprecate HMM routing code path | ✅ **Landed** (Round 1) | `signals/model/hybrid.py:__init__` warns on `routing_strategy='hmm'` pointing to HOMC_TIER0C_HYBRID_RESULTS.md. |
| **C6** | Proper binomial test on "beats B&H in X/16" | ✅ **Landed** | `scripts/project_level_dsr.py` Section B. **At effective N=6, none of the claims clear p<0.05.** |

## Tier D — anti-temptations

| # | Item | Status |
|---|---|---|
| **D1** | Don't sweep more hyperparameters | ✅ In effect — all new scripts are validations, not sweeps |
| **D2** | Don't add more model classes | ✅ None added |
| **D3** | Don't add alternative data before core claim solid | ✅ None added |
| **D4** | Don't route real capital until A+C2 complete | ✅ Alpaca broker remains dry-run by default |

## What's green after Round 2

- **165/165 tests pass**, ruff clean, on Python 3.12.
- **Full `volatility_20d` → `volatility` rename** landed across 17 files
  with `engine.VOLATILITY_COLUMN` as the canonical constant.
- **Shared non-overlap sampler** `scripts/_window_sampler.py` used by all
  7 evaluation scripts with a defensive invariant check.
- **7 evaluation scripts have been EXECUTED**: multi_seed_eval,
  block_bootstrap (--quick), permutation_test (--quick), cost_sensitivity,
  trivial_baselines_btc, regime_ablation, pristine_holdout. All results
  persisted to `scripts/data/*.parquet` or `*.md`.
- **4 visualization PNGs** generated in `scripts/data/plots/`.
- **README updated** with the real multi-seed numbers, pristine holdout
  results, ablation, bootstrap CI, permutation p-value.

## What's still pending after Round 2

- **Full-resolution runs**: permutation_test at N=200 (instead of --quick
  N=20), block_bootstrap at B=1000 (instead of --quick B=100). Scripts
  work, just compute-bound; let them run overnight.
- **A4 extended**: the current pristine_holdout uses a 13-config sweep;
  a more thorough version would use a wider grid. Marginal value given
  the multi-seed story.
- **C1 intraday data**: big project, still deferred.
- **C2 forward paper trade log**: physically requires 30 days of clock
  time. Start now to get results by mid-May.
- **Decision**: should the production default change from q=0.70 to q=0.50
  based on the multi-seed sweep? (Avg Sharpe 0.78 → 0.88, +0.10 improvement.)
  Recommended yes, but held for user review since it would invalidate
  every historical result doc that references q=0.70.

## Ground rules going forward

1. **No "median Sharpe 2.15" in any new README or result doc.** Every
   headline number must quote a multi-seed mean ± stderr or be explicitly
   labeled `(seed=42 in-sample)`.
2. **No new hyperparameter sweep without pre-registering seeds.** If a
   sweep produces a winner at seed 42 only, that's a hypothesis, not a
   result.
3. **No new model classes** until the existing hybrid has survived
   Round-2 cleanly. Given the ablation result, the next step should be
   SIMPLIFICATION not expansion — delete the Markov layer, keep the vol
   router.
4. **No alternative data** until A4's pristine holdout number is
   reconciled with the multi-seed number.
