# $50k SP-complement allocation — 5-year trailing backtest

**Window**: 2021-04-01 → 2026-04-01  (~5 years, 1826 calendar days)
**Initial**: $50,000

Context: investor holds SP500 as their core and has $50k additional capital for a 5-year horizon. The $50k should go into SP-complementary assets — i.e. low correlation with the existing core, positive expected forward return.

The `ρ(SP)` column shows daily-return correlation vs the existing SP core. **Lower correlation = better diversification** for an SP-heavy portfolio.

## Ranking by end value on $50,000

| strategy | end $ | total | CAGR | Sharpe | MDD | Calmar | ρ(SP) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100% GLD buy-and-hold` | $135,146 | +170.3% | +22.0% | +1.087 | -21.0% | +1.05 | +0.125 |
| `60/40 GLD/BTC (inverted)` | $117,410 | +134.8% | +18.6% | +0.705 | -43.4% | +0.43 | +0.387 |
| `50/50 BTC/GLD` | $108,731 | +117.5% | +16.8% | +0.591 | -50.6% | +0.33 | +0.393 |
| `60/40 SP/GLD (gold-bond)` | $103,186 | +106.4% | +15.6% | +1.004 | -19.2% | +0.81 | +0.843 |
| `60/40 BTC/GLD` | $99,026 | +98.1% | +14.6% | +0.501 | -57.0% | +0.26 | +0.394 |
| `70/30 GLD/TLT (defensive)` | $89,296 | +78.6% | +12.3% | +0.732 | -23.9% | +0.51 | +0.132 |
| `70/30 BTC/GLD` | $88,690 | +77.4% | +12.1% | +0.431 | -62.8% | +0.19 | +0.392 |
| `50/30/20 BTC/GLD/TLT` | $81,950 | +63.9% | +10.4% | +0.407 | -54.1% | +0.19 | +0.397 |
| `100% SP500 B&H (redundant w/ core)` | $81,785 | +63.6% | +10.3% | +0.535 | -25.4% | +0.41 | +1.000 |
| `60/25/15 BTC/GLD/TLT` | $80,065 | +60.1% | +9.9% | +0.381 | -59.3% | +0.17 | +0.395 |
| `25/25/25/25 BTC/SP/TLT/GLD (ref)` | $79,398 | +58.8% | +9.7% | +0.480 | -39.7% | +0.24 | +0.580 |
| `80/20 BTC/GLD` | $78,112 | +56.2% | +9.3% | +0.376 | -67.9% | +0.14 | +0.389 |
| `40/30/30 BTC/GLD/TLT` | $76,765 | +53.5% | +9.0% | +0.382 | -49.4% | +0.18 | +0.395 |
| `50/25/25 BTC/GLD/TLT` | $76,248 | +52.5% | +8.8% | +0.358 | -54.9% | +0.16 | +0.396 |
| `100% BTC buy-and-hold` | $57,600 | +15.2% | +2.9% | +0.294 | -76.6% | +0.04 | +0.383 |
| `60/40 SP/TLT (classic pension)` | $57,444 | +14.9% | +2.8% | +0.104 | -28.2% | +0.10 | +0.857 |
| `100% TLT buy-and-hold` | $31,365 | -37.3% | -8.9% | -0.653 | -46.4% | -0.19 | +0.066 |
