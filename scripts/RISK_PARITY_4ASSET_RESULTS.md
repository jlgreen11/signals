# Round-4 #2 — 4-asset risk-parity portfolio (BTC/SP/TLT/GLD)

**Run date**: 2026-04-11
**Script**: `scripts/risk_parity_4asset.py`
**Test parameters**:

- Legs: BTC (H-Vol hybrid `BTC_HYBRID_PRODUCTION`), ^GSPC B&H, TLT B&H, GLD B&H
- Weighting schemes: ['equal', 'inverse_vol_21d', 'inverse_vol_63d']
- Seeds: [42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23]
- Windows: 16 non-overlapping 6-month per seed
- Rebalance: daily (returns-based blend)
- Annualization: 365/yr, rf = historical_usd_rate('2018-2024')

## Multi-seed ranking

| label | weighting | avg Sharpe | stderr | min seed | max seed |
|---|---|---:|---:|---:|---:|
| `rp4_equal` | equal | +1.366 | 0.126 | +0.646 | +1.677 |
| `rp4_inverse_vol_21d` | inverse_vol_21d | +1.118 | 0.036 | +0.837 | +1.284 |
| `rp4_inverse_vol_63d` | inverse_vol_63d | +1.040 | 0.063 | +0.762 | +1.362 |

## Comparison to single-asset baseline

- BTC-alone Round-3 hybrid baseline: +1.188
- Best 4-asset portfolio: +1.366
- Delta: +0.178
- Materiality (Δ ≥ 0.10): **PASS**

Recommend shipping `rp4_equal` as the risk-balanced multi-asset production path.
