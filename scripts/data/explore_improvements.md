# explore_improvements.py — large-grid search for material improvements

**Objective**: find a configuration that materially beats the H-Vol @ q=0.70 production baseline (multi-seed avg Sharpe ≈ 0.78 ± 0.08).

**Discipline**: 5 seeds during exploration, 10 seeds during final candidate confirmation, all seeds pre-registered. Multi-testing correction applied via project-level DSR.

**Sweep trial count**: 144  (75 Tier-1 + 15 Tier-2 + 54 Tier-3)
**Project-level trial count (including legacy)**: 2044

## Tier 1 — Pure vol filter (5 seeds, 75 configs)

Top 10:

| label | vol_window | q | rf | avg Sh | stderr | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| vf_vw14_q0.60_rf7 | 14 | 0.60 | 7 | +0.835 | 0.195 | +0.200 | +1.343 |
| vf_vw14_q0.60_rf21 | 14 | 0.60 | 21 | +0.738 | 0.181 | +0.200 | +1.202 |
| vf_vw10_q0.70_rf42 | 10 | 0.70 | 42 | +0.722 | 0.314 | +0.006 | +1.526 |
| vf_vw14_q0.60_rf42 | 14 | 0.60 | 42 | +0.697 | 0.202 | +0.121 | +1.229 |
| vf_vw10_q0.70_rf21 | 10 | 0.70 | 21 | +0.692 | 0.302 | +0.006 | +1.526 |
| vf_vw10_q0.40_rf21 | 10 | 0.40 | 21 | +0.632 | 0.166 | +0.154 | +1.121 |
| vf_vw10_q0.40_rf7 | 10 | 0.40 | 7 | +0.562 | 0.182 | +0.049 | +1.069 |
| vf_vw10_q0.70_rf7 | 10 | 0.70 | 7 | +0.542 | 0.268 | -0.043 | +1.289 |
| vf_vw14_q0.70_rf7 | 14 | 0.70 | 7 | +0.538 | 0.250 | +0.075 | +1.310 |
| vf_vw14_q0.70_rf42 | 14 | 0.70 | 42 | +0.538 | 0.250 | +0.075 | +1.310 |

## Tier 2 — Vol filter + vol target overlay (5 seeds, 15 configs)

| label | vt_annual | vt_max | avg Sh | stderr |
|---|---:|---:|---:|---:|
| vf_vt_vw14_q0.60_rf7_vt0.40_mx1.5 | 0.40 | 1.5 | +0.664 | 0.227 |
| vf_vt_vw14_q0.60_rf7_vt0.40_mx2.0 | 0.40 | 2.0 | +0.601 | 0.249 |
| vf_vt_vw14_q0.60_rf7_vt0.30_mx1.5 | 0.30 | 1.5 | +0.592 | 0.248 |
| vf_vt_vw14_q0.60_rf7_vt0.25_mx1.5 | 0.25 | 1.5 | +0.584 | 0.230 |
| vf_vt_vw14_q0.60_rf7_vt0.25_mx2.0 | 0.25 | 2.0 | +0.584 | 0.230 |
| vf_vt_vw14_q0.60_rf7_vt0.25_mx3.0 | 0.25 | 3.0 | +0.584 | 0.230 |
| vf_vt_vw14_q0.60_rf7_vt0.40_mx3.0 | 0.40 | 3.0 | +0.581 | 0.256 |
| vf_vt_vw14_q0.60_rf7_vt0.30_mx2.0 | 0.30 | 2.0 | +0.573 | 0.255 |
| vf_vt_vw14_q0.60_rf7_vt0.30_mx3.0 | 0.30 | 3.0 | +0.573 | 0.255 |
| vf_vt_vw14_q0.60_rf7_vt0.20_mx1.5 | 0.20 | 1.5 | +0.548 | 0.246 |

## Tier 3 — Hybrid wider grid (5 seeds, 54 configs)

Top 10:

| label | vol_window | q | rf | tw | avg Sh | stderr |
|---|---:|---:|---:|---:|---:|---:|
| hyb_vw10_q0.50_rf14_tw750 | 10 | 0.50 | 14 | 750 | +1.434 | 0.152 |
| hyb_vw10_q0.55_rf14_tw1000 | 10 | 0.55 | 14 | 1000 | +1.333 | 0.157 |
| hyb_vw10_q0.40_rf14_tw750 | 10 | 0.40 | 14 | 750 | +1.212 | 0.211 |
| hyb_vw10_q0.55_rf21_tw1000 | 10 | 0.55 | 21 | 1000 | +1.210 | 0.209 |
| hyb_vw10_q0.55_rf14_tw750 | 10 | 0.55 | 14 | 750 | +1.166 | 0.314 |
| hyb_vw10_q0.50_rf14_tw1000 | 10 | 0.50 | 14 | 1000 | +1.109 | 0.118 |
| hyb_vw10_q0.40_rf14_tw1000 | 10 | 0.40 | 14 | 1000 | +1.108 | 0.130 |
| hyb_vw14_q0.40_rf21_tw1000 | 14 | 0.40 | 21 | 1000 | +1.095 | 0.122 |
| hyb_vw10_q0.40_rf21_tw750 | 10 | 0.40 | 21 | 750 | +1.075 | 0.092 |
| hyb_vw10_q0.50_rf21_tw750 | 10 | 0.50 | 21 | 750 | +1.061 | 0.206 |

## Tier 4 — 10-seed confirmation (top candidates + production baseline)

| label | avg Sh | stderr | min seed | max seed | mean MDD |
|---|---:|---:|---:|---:|---:|
| hyb_vw10_q0.50_rf14_tw750 | +1.551 | 0.099 | +1.010 | +1.949 | -26.0% |
| hyb_vw10_q0.55_rf14_tw750 | +1.410 | 0.185 | +0.311 | +2.172 | -26.3% |
| hyb_vw10_q0.55_rf14_tw1000 | +1.368 | 0.088 | +0.853 | +1.715 | -24.6% |
| hyb_vw10_q0.40_rf14_tw750 | +1.329 | 0.123 | +0.723 | +1.852 | -24.6% |
| hyb_vw10_q0.55_rf21_tw1000 | +1.249 | 0.118 | +0.498 | +1.726 | -23.5% |
| hybrid_prod_q0.70_w10_r21_tw1000 | +0.893 | 0.100 | +0.345 | +1.403 | -24.8% |

## DSR correction

- Sweep n_trials = 144
- Project n_trials = 2044
- Baseline Sharpe = +0.893
- Winner Sharpe = +1.551
- Winner label = `hyb_vw10_q0.50_rf14_tw750`
- DSR at sweep n_trials: 0.0000
- DSR at project n_trials: 0.0000

