# Multi-stock algo evaluation — top SP500 + NASDAQ names

**Question**: can ANY of the project's model variants beat buy-and-hold on major US equities, across multiple random seeds?

**Universe**: 20 tickers — AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, BRK-B, UNH, JNJ, JPM, V, PG, XOM, LLY, AVGO, COST, NFLX, AMD, ADBE
**Models**: composite, homc_o5, hybrid_hvol, trend_200, golden_cross
**Seeds**: [42, 7, 100, 999, 1337] (5 pre-registered)
**Windows**: 12 non-overlapping 6-month per seed
**Annualization**: 252/yr (equity calendar), rf ≈ 2.3%

## Model-level summary

| model | tickers | Sharpe wins | Sharpe win% | CAGR wins | CAGR win% | avg Δ Sharpe | avg Δ CAGR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hybrid_hvol` | 20 | 3 | 15% | 1 | 5% | -0.178 | -6.0% |
| `composite` | 20 | 1 | 5% | 1 | 5% | -0.171 | -5.6% |
| `homc_o5` | 20 | 1 | 5% | 0 | 0% | -0.198 | -9.1% |
| `golden_cross` | 20 | 0 | 0% | 0 | 0% | -0.134 | -1.2% |
| `trend_200` | 20 | 0 | 0% | 0 | 0% | -0.236 | -5.7% |

## Per-ticker best model (by Sharpe delta vs B&H)

| ticker | best model | algo Sharpe | B&H Sharpe | Δ Sharpe | algo CAGR | B&H CAGR | Δ CAGR | wins? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `ADBE` | `hybrid_hvol` | +0.921 | +0.612 | +0.099 | +25.7% | +16.5% | +8.5% | ✓ |
| `NVDA` | `hybrid_hvol` | +1.128 | +1.528 | +0.052 | +51.3% | +82.9% | -0.6% | ✓ |
| `UNH` | `composite` | +0.322 | +0.464 | +0.014 | +6.8% | +9.4% | +0.2% | ✓ |
| `COST` | `homc_o5` | +1.394 | +1.294 | +0.009 | +27.2% | +27.5% | -0.5% | ✓ |
| `TSLA` | `golden_cross` | +0.366 | +0.928 | -0.003 | +4.9% | +71.4% | -0.6% | ✗ |
| `AMD` | `golden_cross` | +0.586 | +1.052 | -0.004 | +24.6% | +57.1% | -0.3% | ✗ |
| `XOM` | `golden_cross` | +0.000 | +0.311 | -0.006 | +0.0% | +6.0% | -0.3% | ✗ |
| `NFLX` | `golden_cross` | +0.634 | +1.029 | -0.006 | +17.0% | +39.8% | -0.3% | ✗ |
| `AVGO` | `golden_cross` | +0.774 | +0.992 | -0.007 | +24.7% | +39.8% | -0.3% | ✗ |
| `LLY` | `golden_cross` | +0.966 | +1.070 | -0.007 | +25.9% | +40.4% | -0.3% | ✗ |
| `META` | `golden_cross` | +0.853 | +0.926 | -0.008 | +23.3% | +31.6% | -0.3% | ✗ |
| `GOOGL` | `golden_cross` | +0.935 | +0.943 | -0.008 | +26.1% | +26.3% | -0.3% | ✗ |
| `JPM` | `golden_cross` | +0.956 | +1.112 | -0.009 | +21.1% | +24.7% | -0.2% | ✗ |
| `MSFT` | `golden_cross` | +1.102 | +1.164 | -0.010 | +26.0% | +33.7% | -0.2% | ✗ |
| `V` | `golden_cross` | +0.842 | +0.763 | -0.012 | +17.3% | +17.5% | -0.2% | ✗ |
| `BRK-B` | `golden_cross` | +1.015 | +1.149 | -0.016 | +16.1% | +19.2% | -0.2% | ✗ |
| `AAPL` | `hybrid_hvol` | +0.836 | +1.207 | -0.045 | +22.9% | +40.6% | -5.3% | ✗ |
| `AMZN` | `composite` | +0.719 | +1.015 | -0.105 | +18.6% | +28.9% | -2.9% | ✗ |
| `PG` | `composite` | +0.367 | +0.257 | -0.115 | +4.3% | +4.6% | -0.9% | ✗ |
| `JNJ` | `homc_o5` | +0.000 | +0.182 | -0.153 | +0.0% | +3.0% | -2.8% | ✗ |

## Verdict

- Total (ticker × model) pairs: **100**
- Algo beats B&H on Sharpe: **5/100 (5%)**
- Algo beats B&H on CAGR: **2/100 (2%)**
- Tickers where best algo beats B&H: **4/20 (20%)**

