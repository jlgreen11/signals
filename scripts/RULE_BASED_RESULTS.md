# Experiment 2 — RuleBasedSignalGenerator on HOMC

**Run date**: 2026-04-11 (Round-3 follow-up)
**Script**: `scripts/rule_based_eval.py`
**Test parameters**:

- Model: HOMC (quantile encoder) + `RuleBasedSignalGenerator`
- Grid: `top_k ∈ [10, 20]` × `p_threshold ∈ [0.6, 0.7]` × `order ∈ [3, 5]` × `n_states ∈ [5, 7]` = 16 configs
- Training: 2015-01-01 → 2022-12-31, train_window=1000, retrain_freq=21
- Seeds: 10 pre-registered ([42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23])
- Windows: 10 non-overlapping 6-month per seed
- Pristine holdout: 2023-01-01 → 2024-12-31, single-shot
- Baseline: pure vol filter ~1.15 Sharpe median
- Materiality: avg Sharpe ≥ 1.30 AND holdout > 0

## In-sample ranking

| label | top_k | p_thr | order | n_states | avg Sharpe | stderr | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rule_k10_p0.60_o3_s7` | 10 | 0.60 | 3 | 7 | +0.567 | 0.072 | +0.426 | +0.927 |
| `rule_k10_p0.70_o3_s5` | 10 | 0.70 | 3 | 5 | +0.352 | 0.172 | -0.338 | +0.971 |
| `rule_k20_p0.70_o3_s5` | 20 | 0.70 | 3 | 5 | +0.339 | 0.176 | -0.338 | +0.971 |
| `rule_k10_p0.60_o3_s5` | 10 | 0.60 | 3 | 5 | +0.321 | 0.209 | -0.583 | +1.101 |
| `rule_k20_p0.60_o5_s5` | 20 | 0.60 | 5 | 5 | +0.285 | 0.059 | +0.088 | +0.625 |
| `rule_k10_p0.70_o3_s7` | 10 | 0.70 | 3 | 7 | +0.227 | 0.074 | +0.000 | +0.710 |
| `rule_k20_p0.60_o3_s7` | 20 | 0.60 | 3 | 7 | +0.214 | 0.085 | -0.063 | +0.788 |
| `rule_k20_p0.60_o3_s5` | 20 | 0.60 | 3 | 5 | +0.197 | 0.193 | -0.887 | +0.893 |
| `rule_k20_p0.70_o5_s5` | 20 | 0.70 | 5 | 5 | +0.151 | 0.027 | +0.088 | +0.307 |
| `rule_k20_p0.70_o3_s7` | 20 | 0.70 | 3 | 7 | +0.070 | 0.128 | -0.424 | +0.946 |
| `rule_k10_p0.60_o5_s5` | 10 | 0.60 | 5 | 5 | +0.042 | 0.014 | +0.000 | +0.110 |
| `rule_k10_p0.70_o5_s5` | 10 | 0.70 | 5 | 5 | +0.042 | 0.014 | +0.000 | +0.110 |
| `rule_k10_p0.60_o5_s7` | 10 | 0.60 | 5 | 7 | +0.000 | 0.000 | +0.000 | +0.000 |
| `rule_k10_p0.70_o5_s7` | 10 | 0.70 | 5 | 7 | +0.000 | 0.000 | +0.000 | +0.000 |
| `rule_k20_p0.60_o5_s7` | 20 | 0.60 | 5 | 7 | +0.000 | 0.000 | +0.000 | +0.000 |
| `rule_k20_p0.70_o5_s7` | 20 | 0.70 | 5 | 7 | +0.000 | 0.000 | +0.000 | +0.000 |

## Pristine holdout — winner only

Winner: `rule_k10_p0.60_o3_s7`  in-sample +0.567

| metric | value |
|---|---:|
| Holdout Sharpe | +0.618 |
| Holdout CAGR | +18.9% |
| Holdout Max DD | -21.2% |
| Holdout trades | 2 |

## Verdict

- In-sample (≥ 1.30): **FAIL** (+0.567)
- Holdout (> 0): **PASS** (+0.618)
- **Overall: FAILURE**

**Negative result.** Per D1, the grid is NOT expanded. Rule-extraction arm of the Markov-layer question is closed as a fail.
