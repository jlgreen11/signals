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
| `rp4_equal` | equal | +1.695 | 0.147 | +0.852 | +2.065 |
| `rp4_inverse_vol_21d` | inverse_vol_21d | +1.452 | 0.048 | +1.092 | +1.677 |
| `rp4_inverse_vol_63d` | inverse_vol_63d | +1.357 | 0.078 | +0.998 | +1.754 |

## Comparison to single-asset baseline

- BTC-alone Round-3 hybrid baseline: +1.551
- Best 4-asset portfolio: +1.695
- Delta: +0.144
- Materiality (Δ ≥ 0.10): **PASS**

Recommend shipping `rp4_equal` as the risk-balanced multi-asset production path.
