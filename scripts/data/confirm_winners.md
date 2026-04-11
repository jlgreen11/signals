# 10-seed confirmation — Tier 1 vol filter vs production hybrid

Confirms the Tier-1 winner from `explore_improvements.py` at 10 seeds,
side-by-side with the old q=0.70 legacy default and the new q=0.50
Round-2 default.

## Multi-seed summary

| strategy | multi-seed mean Sharpe | stderr | min seed | max seed |
|---|---:|---:|---:|---:|
| `hvol_q0.70_legacy` | +1.175 | 0.083 | +0.784 | +1.447 |
| `hvol_q0.50_new_default` | +1.081 | 0.074 | +0.800 | +1.457 |
| `vf_vw14_q0.60_rf7` | +0.890 | 0.102 | +0.200 | +1.343 |

**Winner**: `hvol_q0.70_legacy`  multi-seed mean Sharpe +1.175 ± 0.083

## DSR correction

- n_trials at sweep level (explore_improvements.py): 144
- n_trials at project level (legacy + round 3): 2044
- DSR at sweep: 0.0000
- DSR at project: 0.0000

## Interpretation

If the Tier-1 vol filter winner survives DSR at project-level n_trials (DSR >= 0.95), it should be promoted to the new production default. Otherwise, document as 'winner by point estimate but not distinguishable from noise at the project-level multi-trial correction'.
