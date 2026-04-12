# Trailing 3-year view (2022-01-01 → 2024-12-31)

Single-window, end-anchored comparison of the project's top strategies against SP500 buy-and-hold over the most recent 3 calendar years of available data. Everything starts at $10,000 on 2022-01-03 (first equity trading day ≥ 2022-01-01) and ends on 2024-12-31.

Annualization: 252/yr (equity shared calendar). Risk-free rate ≈ 2.3% (2018-2024 average).

## Summary

| strategy | end value | total return | CAGR | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| `BTC_HYBRID_PRODUCTION` | $23,416 | +134.2% | +32.9% | +0.845 | -64.9% | +0.51 |
| `GLD buy-and-hold` | $14,384 | +43.8% | +12.9% | +0.754 | -21.0% | +0.61 |
| `4-asset equal-weight portfolio` | $13,393 | +33.9% | +10.3% | +0.584 | -33.6% | +0.31 |
| `SP500 buy-and-hold` | $12,262 | +22.6% | +7.1% | +0.349 | -25.4% | +0.28 |
| `60/40 SP/TLT (classic)` | $9,434 | -5.7% | -1.9% | -0.252 | -27.4% | -0.07 |
| `TLT buy-and-hold` | $6,052 | -39.5% | -15.4% | -0.986 | -42.6% | -0.36 |

## Quarterly portfolio value

$10,000 initial → quarter-end value for each strategy.

| quarter end | 4-asset portfolio | SP B&H | BTC hybrid |
|---|---:|---:|---:|
| 2022-03-31 | $9,149 | $9,445 | $7,308 |
| 2022-06-30 | $7,101 | $7,892 | $3,736 |
| 2022-09-30 | $6,798 | $7,475 | $3,997 |
| 2022-12-30 | $7,225 | $8,005 | $4,414 |
| 2023-03-31 | $8,139 | $8,567 | $5,633 |
| 2023-06-30 | $8,627 | $9,278 | $6,901 |
| 2023-09-29 | $8,100 | $8,940 | $6,667 |
| 2023-12-29 | $9,634 | $9,944 | $9,457 |
| 2024-03-28 | $11,232 | $10,954 | $14,954 |
| 2024-06-28 | $11,327 | $11,384 | $14,426 |
| 2024-09-30 | $12,701 | $12,014 | $17,486 |
| 2024-12-31 | $13,393 | $12,262 | $23,416 |

## Interpretation

Read the Sharpe column as the risk-adjusted winner and CAGR as the absolute-return winner. A single 3-year window is NOT statistically robust (N=1), so treat this as a narrative snapshot rather than evidence. For multi-seed averages see `scripts/PORTFOLIO_VS_SP_BH.md` and `scripts/BROAD_COMPARISON.md`.

## Raw data

- `scripts/data/trailing_3y_view.parquet` — daily portfolio values for all 3 headline strategies
- Reproduce: `python scripts/trailing_3y_view.py`

