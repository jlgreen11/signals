# SKEPTIC_REVIEW.md — improvements in flight

Live tracking of what's landed from the skeptic review punch list. Updated
as improvements merge. Reference the tier numbers against `SKEPTIC_REVIEW.md`
for the original critique.

## Tier A — must-fix

| # | Item | Status | Evidence |
|---|---|---|---|
| **A1** | Replace headline numbers with multi-seed summaries | 🟡 Script written, not yet run | `scripts/multi_seed_eval.py` — runs 10 seeds × 6 vol quantiles on BTC H-Vol. Output pending execution. |
| **A2** | Enforce non-overlap in `random_window_eval.py` | ✅ Landed | `scripts/random_window_eval.py:208-237` now uses rejection sampling with `min_spacing=six_months`. Clamps requested `n_windows` down to `max_fit` with a WARN if too many are asked for. |
| **A3** | Moving-block bootstrap CIs | 🟡 Script written, not yet run | `scripts/block_bootstrap.py` — B=1000 moving-block bootstrap on per-window return series with block length 21. |
| **A4** | Pristine holdout (re-tune on 2015–2022, eval once on 2023–2024) | 🔴 Not started | Requires a meaningful re-tune cycle, scheduled post-paralllel-batch. |
| **A5** | Multi-seed robustness on the baseline retroactively | 🟡 Covered by A1 script | `scripts/multi_seed_eval.py` sweeps `vol_quantile ∈ {0.50…0.90}` at 10 seeds. |

## Tier B — methodological gaps

| # | Item | Status | Evidence |
|---|---|---|---|
| **B1** | Monte-Carlo permutation test | 🟡 Script written, not yet run | `scripts/permutation_test.py` — N=200 return-shuffle permutations per window, one-sided p-value. |
| **B2** | Transaction cost sensitivity surface | 🟡 Script written, not yet run | `scripts/cost_sensitivity.py` — 5×5 grid over `commission_bps × min_trade_fraction` plus a 1×5 slippage slice. |
| **B3** | Decouple deadband from cost assumption | 🟡 Covered by B2 | Same script sweeps `min_trade_fraction ∈ {0.05…0.30}` independently. |
| **B4** | Trivial baselines on BTC | 🟡 Script written, not yet run | `scripts/trivial_baselines_btc.py` — B&H, Trend(200), Golden Cross, **pure vol filter (no Markov)**, H-Vol. The pure vol filter is the decisive test of § 5 of the skeptic review. |
| **B5** | Project-level DSR | 🟡 Script written, fast to run | `scripts/project_level_dsr.py` — recomputes DSR at per-sweep (25), per-tier (200), and project-level (~1900) trial counts. |
| **B6** | Fix annualization + volatility naming | ✅ Partial | `signals/backtest/metrics.py`: `_annualization_factor` and `compute_metrics` now take an explicit `periods_per_year` override. `BacktestConfig.periods_per_year` plumbed through to `compute_metrics`. Set it to 365 for crypto. The `volatility_20d` column name is preserved (rename is a separate blast-radius change) with a prominent NOTE comment at `engine.py:_prepare_features` explaining it's historical and the actual window is `config.vol_window`. Full rename tracked as future work. |
| **B7** | Non-zero risk-free rate in headlines | 🔴 Not started | Simple plumbing once we decide which series (3M T-bill). |

## Tier C — structural

| # | Item | Status | Evidence |
|---|---|---|---|
| **C1** | Hourly / intraday data | 🔴 Not started | Big project. Out of scope for this batch. |
| **C2** | Forward-paper-trade log committed | 🔴 Not started | Infrastructure exists (`signals/broker/paper_trade_log.py`); needs 30-day forward run. |
| **C3** | Per-window daily equity curves in result docs | 🔴 Not started | Easy plotting follow-up once A1 results land. |
| **C4** | Regime filter ablation | 🟡 Script written, not yet run | `scripts/regime_ablation.py` — composite-only / HOMC-only / both-ablated variants of the hybrid. Tests whether the Markov components matter at all. |
| **C5** | Deprecate HMM routing code path | ✅ Landed | `signals/model/hybrid.py:__init__` — warns on `routing_strategy='hmm'` pointing to HOMC_TIER0C_HYBRID_RESULTS.md. |
| **C6** | Proper binomial test on "beats B&H in X/16" | 🟡 Covered by B5 script | Section B of `scripts/project_level_dsr.py` runs exact binomial tests on 12/16, 11/16, 10/16, 13/16 at both nominal (16) and effective (6) window counts. |

## Tier D — anti-temptations

| # | Item | Status |
|---|---|---|
| **D1** | Don't sweep more hyperparameters | ✅ Discipline in effect — no new sweeps added in this session |
| **D2** | Don't add more model classes | ✅ None added |
| **D3** | Don't add alternative data before core claim solid | ✅ None added |
| **D4** | Don't route real capital until A+C2 complete | ✅ Alpaca broker remains dry-run by default |

## What's green after this batch

- 165/165 tests pass (was 140+) — the annualization change and all other core
  edits are regression-clean.
- `SKEPTIC_REVIEW.md` published at repo root with full critique and tier list.
- README updated with seed-variance caveats, methodology-caveat section, and
  footnoted headline numbers (pending the README-update agent's worktree merge).
- Six new evaluation scripts scaffolded in `scripts/` (pending agent worktree
  merges). Each is runnable standalone and saves to `scripts/data/*.parquet`
  (or `.md` for the DSR summary).
- Core code paths corrected for annualization (B6), HMM routing deprecation
  (C5), and non-overlap sampling (A2).

## What's still pending after this batch

- Actually **running** the evaluation scripts A1/A3/B1/B2/B4/C4 (compute takes
  hours; scripts are deterministic and well-commented so you can launch them
  in sequence or parallel at your convenience).
- A4 pristine holdout re-tune (requires a full walk-forward pass on 2015–2022
  with zero visibility into 2023–2024).
- B7 risk-free rate plumbing (small).
- C1 / C2 / C3 structural work.
- Full `volatility_20d` → `volatility` rename (deferred — low severity,
  high blast radius; comment-documented for now).

## Ground rules going forward (from Tier D)

1. **No "median Sharpe 2.15" in any new README or result doc.** Every headline
   number must quote a multi-seed mean ± stderr or be explicitly labeled
   `(seed=42 in-sample)`.
2. **No new hyperparameter sweep without pre-registering seeds.** If a sweep
   produces a winner at seed 42 only, that's a hypothesis, not a result.
3. **No new model classes** until the existing hybrid has survived A1 + B1 +
   B4 cleanly.
4. **No alternative data** until A4 delivers a clean OOS number.
