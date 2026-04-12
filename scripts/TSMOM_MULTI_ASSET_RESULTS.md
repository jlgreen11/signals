# Multi-Asset Time-Series Momentum (TSMOM) Results

Moskowitz-Ooi-Pedersen (2012) trend-following signal across a diversified
macro asset-class basket. Long-only: go long if trailing return > 0,
flat otherwise. Risk-parity weighting (inverse realized vol).
5 bps commission + 5 bps slippage per rebalance.

Universe: ^GSPC, TLT, IEF, GLD, USO, UUP, EFA, BTC-USD
Risk-free rate: 2.26% (historical 2018-2024 average)
Annualization: 252 days/year

## Trailing 7-year evaluation (2019-04-01 -> 2026-04-01)

| Strategy | End Value | Total Return | CAGR | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| `TSMOM-63d` | $21,341 | +113.4% | +11.4% | +0.954 | -10.2% | +1.13 |
| `TSMOM-Combined` | $19,194 | +91.9% | +9.8% | +0.947 | -8.9% | +1.10 |
| `EqWt B&H (multi-asset)` | $24,473 | +144.7% | +13.6% | +0.881 | -22.8% | +0.60 |
| `TSMOM-21d` | $18,701 | +87.0% | +9.4% | +0.736 | -17.3% | +0.54 |
| `TSMOM-252d` | $17,260 | +72.6% | +8.1% | +0.698 | -11.2% | +0.72 |
| `SP500 B&H` | $22,933 | +129.3% | +12.6% | +0.583 | -33.9% | +0.37 |

## Multi-seed windowed evaluation (5 seeds x 12 non-overlapping 6-month windows)

| Strategy | Mean Sharpe | Median Sharpe | Std Sharpe | Mean CAGR | Mean MDD | BH Sharpe | N | % > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `TSMOM-21d` | +0.970 | +0.868 | 1.591 | +11.4% | -4.7% | +0.961 | 60 | 75% |
| `TSMOM-63d` | +0.793 | +0.753 | 1.319 | +8.9% | -5.1% | +0.961 | 60 | 75% |
| `TSMOM-252d` | +0.583 | +0.415 | 1.471 | +6.9% | -4.9% | +0.961 | 60 | 68% |
| `TSMOM-Combined` | +0.863 | +0.314 | 1.519 | +8.9% | -4.2% | +0.961 | 60 | 70% |

## Reproduce

```bash
python scripts/tsmom_multi_asset_eval.py
```

