# Experiment 1 — AbsoluteGranularityEncoder HOMC

**Run date**: 2026-04-11 (Round-3 follow-up)
**Script**: `scripts/absolute_encoder_eval.py`
**Test parameters**:

- Model: HOMC with `AbsoluteGranularityEncoder(bin_width)` at `order ∈ [3, 5, 7]` × `bin_width ∈ [0.005, 0.01, 0.02]`
- Training window: 2015-01-01 → 2022-12-31, HOMC train_window=1000, retrain_freq=21
- Seeds: 10 pre-registered ([42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23])
- Windows: 10 non-overlapping 6-month windows per seed
- Pristine holdout: 2023-01-01 → 2024-12-31, single-shot walk-forward
- Baseline for comparison: pure vol filter median Sharpe ~1.15 from `scripts/trivial_baselines_btc.py`
- Materiality threshold: in-sample avg Sharpe ≥ 1.30 AND holdout > 0

## Pre-registered grid (per D1 — do not expand)

- `bin_width ∈ [0.005, 0.01, 0.02]`  (0.5%, 1%, 2% absolute return bins)
- `order ∈ [3, 5, 7]`  (Markov memory depth)
- 9 total configs

## In-sample ranking

| label | bin_width | order | avg Sharpe | stderr | min seed | max seed |
|---|---:|---:|---:|---:|---:|---:|
| `abs_bw0.005_o3` | 0.005 | 3 | +0.000 | 0.000 | +0.000 | +0.000 |
| `abs_bw0.005_o5` | 0.005 | 5 | +0.000 | 0.000 | +0.000 | +0.000 |
| `abs_bw0.005_o7` | 0.005 | 7 | +0.000 | 0.000 | +0.000 | +0.000 |
| `abs_bw0.010_o3` | 0.010 | 3 | +0.000 | 0.000 | +0.000 | +0.000 |
| `abs_bw0.010_o5` | 0.010 | 5 | +0.000 | 0.000 | +0.000 | +0.000 |
| `abs_bw0.010_o7` | 0.010 | 7 | +0.000 | 0.000 | +0.000 | +0.000 |
| `abs_bw0.020_o3` | 0.020 | 3 | +0.000 | 0.000 | +0.000 | +0.000 |
| `abs_bw0.020_o5` | 0.020 | 5 | +0.000 | 0.000 | +0.000 | +0.000 |
| `abs_bw0.020_o7` | 0.020 | 7 | +0.000 | 0.000 | +0.000 | +0.000 |

## Pristine holdout — winner only

Winner: `abs_bw0.005_o3`  (in-sample +0.000)

| metric | value |
|---|---:|
| Holdout Sharpe | +0.000 |
| Holdout CAGR | +0.0% |
| Holdout Max DD | +0.0% |
| Holdout trades | 0 |

## Verdict

- In-sample criterion (≥ 1.30): **FAIL** (+0.000)
- Holdout criterion (> 0): **FAIL** (+0.000)
- **Overall: FAILURE**

**Negative result.** Per epistemic guardrail D1, the grid is NOT expanded. The `AbsoluteGranularityEncoder` branch of the Markov-layer question is closed as a fail.
