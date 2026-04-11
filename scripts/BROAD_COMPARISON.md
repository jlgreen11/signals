# Broad Sharpe comparison

Every strategy the project has evaluated, aggregated from the persisted parquets in `scripts/data/` after the Round-5 sweep-fix pass (commits 732111e + 603b96c). Every row uses the correct per-asset annualization: **365/yr for BTC-only**, **252/yr for equity-only and equity-calendar portfolios**.

Rows sort by `avg_sharpe` descending within each family.

⚠ **Read the `N` column carefully**. Some sources are **single-seed** (N=1) — for those, the "avg Sharpe" is the cross-window median at seed 42 only. Multi-seed sources (N=10) report the mean of per-seed medians. The two are NOT directly comparable: a single-seed reading can swing ±0.3 Sharpe from its 10-seed average (see the pure vol filter at 1.348 single-seed vs 0.890 ± 0.102 10-seed), so trust multi-seed readings over single-seed readings when ranking strategies.

| strategy | family | annualization | avg Sharpe | stderr | min seed | max seed | N |
|---|---|---|---:|---:|---:|---:|---:|
| `Rule-based HOMC winner (rule_k10_p0.60_o3_s7)` | BTC / Markov sunset | 365/yr | +0.567 | +0.072 | +0.426 | +0.927 | 10 |
| `HOMC absolute-encoder winner (abs_bw0.005_o3)` | BTC / Markov sunset | 365/yr | +0.000 | +0.000 | +0.000 | +0.000 | 10 |
| `VolFilterOnly` | BTC / single-asset | 365/yr | +1.348 | +0.000 | -1.588 | +4.080 | 1 |
| `Ablation: composite_only` | BTC / single-asset | 365/yr | +1.201 | +0.000 | +1.201 | +1.201 | 1 |
| `BTC_HYBRID_PRODUCTION (q=0.50 rf=14 tw=750)` | BTC / single-asset | 365/yr | +1.188 | +0.025 | +1.039 | +1.239 | 10 |
| `hvol_q0.70_legacy (10-seed confirm)` | BTC / single-asset | 365/yr | +1.175 | +0.083 | +0.784 | +1.447 | 10 |
| `Ablation: both_constants` | BTC / single-asset | 365/yr | +1.106 | +0.000 | +1.106 | +1.106 | 1 |
| `hvol_q0.50_new_default (10-seed confirm)` | BTC / single-asset | 365/yr | +1.081 | +0.074 | +0.800 | +1.457 | 10 |
| `Ablation: homc_only` | BTC / single-asset | 365/yr | +1.031 | +0.000 | +1.031 | +1.031 | 1 |
| `H-Vol q=0.50 (legacy defaults)` | BTC / single-asset | 365/yr | +1.009 | +0.037 | +0.866 | +1.303 | 10 |
| `Ablation: full` | BTC / single-asset | 365/yr | +0.916 | +0.000 | +0.916 | +0.916 | 1 |
| `B&H` | BTC / single-asset | 365/yr | +0.912 | +0.000 | -1.802 | +4.085 | 1 |
| `H-Vol q=0.70 (legacy defaults)` | BTC / single-asset | 365/yr | +0.893 | +0.100 | +0.345 | +1.403 | 10 |
| `vf_vw14_q0.60_rf7 (10-seed confirm)` | BTC / single-asset | 365/yr | +0.890 | +0.102 | +0.200 | +1.343 | 10 |
| `H-Vol q=0.60 (legacy defaults)` | BTC / single-asset | 365/yr | +0.880 | +0.110 | +0.295 | +1.376 | 10 |
| `H-Vol q=0.75 (legacy defaults)` | BTC / single-asset | 365/yr | +0.830 | +0.071 | +0.475 | +1.215 | 10 |
| `H-Vol hybrid (legacy defaults)` | BTC / single-asset | 365/yr | +0.819 | +0.000 | -2.068 | +4.682 | 1 |
| `BTC hybrid + vt_enabled_annual0.25` | BTC / single-asset | 365/yr | +0.664 | +0.074 | +0.174 | +1.066 | 10 |
| `H-Vol q=0.80 (legacy defaults)` | BTC / single-asset | 365/yr | +0.604 | +0.133 | +0.086 | +1.299 | 10 |
| `H-Vol q=0.90 (legacy defaults)` | BTC / single-asset | 365/yr | +0.587 | +0.169 | -0.157 | +1.452 | 10 |
| `DualMA(50,200)` | BTC / single-asset | 365/yr | +0.584 | +0.000 | -2.525 | +4.080 | 1 |
| `BTC hybrid + vt_enabled_annual0.20` | BTC / single-asset | 365/yr | +0.584 | +0.135 | -0.392 | +0.834 | 10 |
| `BTC hybrid + vt_enabled_annual0.15` | BTC / single-asset | 365/yr | +0.580 | +0.036 | +0.328 | +0.731 | 10 |
| `TrendFilter(200)` | BTC / single-asset | 365/yr | +0.000 | +0.000 | -2.818 | +4.080 | 1 |
| `4-asset equal (BTC+SP+TLT+GLD)` | Multi-asset portfolio | 252/yr | +1.366 | +0.126 | +0.646 | +1.677 | 10 |
| `4-asset inverse_vol_21d (BTC+SP+TLT+GLD)` | Multi-asset portfolio | 252/yr | +1.118 | +0.036 | +0.837 | +1.284 | 10 |
| `4-asset inverse_vol_63d (BTC+SP+TLT+GLD)` | Multi-asset portfolio | 252/yr | +1.040 | +0.063 | +0.762 | +1.362 | 10 |

## How to read

- **BTC / single-asset**: evaluated on BTC-USD at 365/yr annualization.
- **BTC / Markov sunset**: the closure experiments for the Markov class — retained in the code, shown here only for reference.
- **Multi-asset portfolio**: 4-asset equal-weight and risk-parity variants on the equity shared calendar at 252/yr.
- **N**: number of pre-registered seeds the aggregation covers. Some sources are single-seed (N=1) — those are flagged in the source-parquet comments in `scripts/broad_comparison.py`.
- **Legacy H-Vol q=0.70** measured on default rf=21/tw=1000 differs from `BTC_HYBRID_PRODUCTION (q=0.50 rf=14 tw=750)` because the Round-3 winner tunes all three parameters together.

**Production recommendations** (per README.md):

1. 4-asset equal-weight risk-parity basket — highest Sharpe on the equity-calendar number, strongest diversification behavior in stress windows.
2. BTC alone via `BTC_HYBRID_PRODUCTION` — simpler operationally, stable multi-seed stderr, 365/yr annualization.
3. ^GSPC alone — buy & hold. No active strategy in the project beats B&H on S&P across 4 model classes tested.

