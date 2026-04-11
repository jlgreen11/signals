# Round-4 #1 — vol_target overlay sweep

**Run date**: 2026-04-11
**Script**: `scripts/vol_target_sweep.py`
**Test parameters**:

- Base config: `BTC_HYBRID_PRODUCTION` (q=0.50, rf=14, tw=750, vw=10)
- Overlay: `vol_target_enabled` ∈ {False, True} with
  `vol_target_annual` ∈ [0.15, 0.2, 0.25] when enabled
- Seeds: [42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23]
- Windows: 16 non-overlapping 6-month per seed
- Annualization: 365/yr, rf = historical_usd_rate('2018-2024')

## Multi-seed ranking

| label | vt_annual | avg Sharpe | stderr | min seed | max seed |
|---|---:|---:|---:|---:|---:|
| `baseline_no_vol_target` | 0.00 | +1.188 | 0.025 | +1.039 | +1.239 |
| `vt_enabled_annual0.25` | 0.25 | +0.664 | 0.074 | +0.174 | +1.066 |
| `vt_enabled_annual0.20` | 0.20 | +0.584 | 0.135 | -0.392 | +0.834 |
| `vt_enabled_annual0.15` | 0.15 | +0.580 | 0.036 | +0.328 | +0.731 |

## Verdict

- Materiality (winner − baseline ≥ 0.10): **FAIL** (+0.000)
- Min-seed dominance: **FAIL** (+1.039 vs baseline +1.039)
- **NO CHANGE**

Vol target overlay did not materially improve the production bundle. Baseline (no overlay) retained.
