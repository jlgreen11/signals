# Pristine holdout — SKEPTIC_REVIEW.md Tier A4

**Training slice**: 2015-01-01 → 2022-12-31 (sweep + in-sample eval)
**Holdout slice**: 2023-01-01 → 2024-12-31 (single-shot, never seen during tuning)

## Sweep on training slice (13 configs, 12 non-overlapping windows, seed=42)

| label | q | buy | sell | mtf | IS med Sh | IS mean Sh | IS med CAGR | IS mean MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q=0.6_sell=-25.0_mtf=0.1 | 0.60 | +25.0 | -25.0 | 0.10 | +0.396 | +0.590 | +10.1% | -31.3% |
| q=0.6_sell=-25.0_mtf=0.2 | 0.60 | +25.0 | -25.0 | 0.20 | +0.396 | +0.590 | +10.1% | -31.3% |
| q=0.6_sell=-35.0_mtf=0.1 | 0.60 | +25.0 | -35.0 | 0.10 | +0.319 | +0.595 | +8.5% | -31.2% |
| q=0.6_sell=-35.0_mtf=0.2 | 0.60 | +25.0 | -35.0 | 0.20 | +0.319 | +0.595 | +8.5% | -31.2% |
| q=0.7_sell=-35.0_mtf=0.1 | 0.70 | +25.0 | -35.0 | 0.10 | +0.277 | +0.622 | +5.0% | -30.2% |
| q=0.7_sell=-35.0_mtf=0.2 | 0.70 | +25.0 | -35.0 | 0.20 | +0.277 | +0.622 | +5.0% | -30.2% |
| production_hvol_default | 0.70 | +25.0 | -35.0 | 0.20 | +0.277 | +0.622 | +5.0% | -30.2% |
| q=0.7_sell=-25.0_mtf=0.1 | 0.70 | +25.0 | -25.0 | 0.10 | +0.238 | +0.586 | +1.7% | -30.6% |
| q=0.7_sell=-25.0_mtf=0.2 | 0.70 | +25.0 | -25.0 | 0.20 | +0.238 | +0.586 | +1.7% | -30.6% |
| q=0.8_sell=-25.0_mtf=0.1 | 0.80 | +25.0 | -25.0 | 0.10 | +0.111 | +0.527 | -7.3% | -31.6% |
| q=0.8_sell=-25.0_mtf=0.2 | 0.80 | +25.0 | -25.0 | 0.20 | +0.111 | +0.527 | -7.3% | -31.6% |
| q=0.8_sell=-35.0_mtf=0.1 | 0.80 | +25.0 | -35.0 | 0.10 | +0.111 | +0.551 | -7.3% | -31.2% |
| q=0.8_sell=-35.0_mtf=0.2 | 0.80 | +25.0 | -35.0 | 0.20 | +0.111 | +0.551 | -7.3% | -31.2% |

## Single-shot holdout evaluation

| config | Sharpe | CAGR | Max DD |
|---|---:|---:|---:|
| Buy & hold (2023-2024) | +1.906 | +137.2% | -26.2% |
| Sweep winner (q=0.6_sell=-25.0_mtf=0.1) | +2.453 | +158.0% | -19.2% |
| Production H-Vol default | +2.690 | +181.9% | -19.2% |

Risk-free rate for Sharpe uses `historical_usd_rate` from `signals.backtest.risk_free`: 0.009 for the 2018-2022 in-sample average, 0.050 for the 2023-2024 holdout average.

**Interpretation**: this is the project's only clean OOS number. The 'Production H-Vol holdout' row is the one to compare against the README's 2.21 headline — note that the README number was measured on the same period but with the tuning process having visibility into it, while this number was produced by a config chosen with zero visibility of the 2023-2024 slice.
