# Round-4 #4 — signal hysteresis widening

**Run date**: 2026-04-11
**Script**: `scripts/hysteresis_sweep.py`
**Test parameters**:

- Base config: `BTC_HYBRID_PRODUCTION`
- Threshold pairs (buy_bps, sell_bps): [(25.0, -35.0), (30.0, -50.0), (40.0, -60.0)]
- Seeds: [42, 7, 100, 999, 1337, 2024, 3, 11, 17, 23]
- Windows: 16 non-overlapping 6-month per seed

## Multi-seed ranking

| label | buy_bps | sell_bps | avg Sharpe | stderr | min seed | max seed |
|---|---:|---:|---:|---:|---:|---:|
| `hyst_buy25_sell-35` | +25 | -35 | +1.188 | 0.025 | +1.039 | +1.239 |
| `hyst_buy30_sell-50` | +30 | -50 | +1.176 | 0.075 | +0.795 | +1.593 |
| `hyst_buy40_sell-60` | +40 | -60 | +0.931 | 0.121 | +0.305 | +1.481 |

## Verdict

- Delta vs production: +0.000
- Materiality (Δ ≥ 0.05): **FAIL**
- Min-seed dominance: **FAIL**
- **Overall: NO CHANGE**

