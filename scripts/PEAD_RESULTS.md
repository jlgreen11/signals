# PEAD Evaluation Results

**Post-Earnings Announcement Drift** — long-only strategy on 20 major US equities.

- **Window**: 2021-04-01 to 2026-04-01
- **Tickers with data**: 20/20
- **Earnings events**: 400 with surprise data
- **Risk-free rate**: 0.0226 (historical USD 3M T-bill, 2018-2024)
- **Cost**: 5.0 bps entry + 5.0 bps exit
- **Max positions**: 5 concurrent

## Performance Comparison

| Config | Sharpe | CAGR | Max DD | Calmar | Final$ | #Trades | Win% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SP500 B&H` | +0.535 | +10.3% | -25.4% | +0.41 | 16,357 | nan | nan% |
| `EW-20 B&H` | +0.929 | +20.5% | -29.4% | +0.70 | 25,373 | nan | nan% |
| `PEAD_t3_h30` | +0.960 | +23.4% | -26.6% | +0.88 | 28,579 | 234.0 | 59.4% |
| `PEAD_t3_h60` | +0.308 | +6.8% | -32.8% | +0.21 | 13,899 | 234.0 | 60.7% |
| `PEAD_t3_h90` | +0.073 | +0.2% | -38.5% | +0.01 | 10,113 | 234.0 | 62.0% |
| `PEAD_t5_h30` | +0.773 | +19.4% | -32.5% | +0.60 | 24,231 | 186.0 | 60.2% |
| `PEAD_t5_h60` | +0.178 | +3.5% | -40.3% | +0.09 | 11,864 | 186.0 | 61.8% |
| `PEAD_t5_h90` | +0.790 | +20.6% | -28.6% | +0.72 | 25,456 | 186.0 | 63.4% |
| `PEAD_t10_h30` | +0.679 | +17.5% | -47.8% | +0.37 | 22,393 | 104.0 | 62.5% |
| `PEAD_t10_h60` | +0.330 | +8.0% | -44.0% | +0.18 | 14,692 | 104.0 | 64.4% |
| `PEAD_t10_h90` | +0.581 | +16.7% | -48.0% | +0.35 | 21,634 | 104.0 | 66.3% |

## Trade-Level Statistics

| Config | Trades | Winners | Losers | Win% | Avg Net | Med Net |
|---|---:|---:|---:|---:|---:|---:|
| `PEAD_t3_h30` | 234 | 139 | 95 | 59.4% | +2.49% | +1.79% |
| `PEAD_t3_h60` | 234 | 142 | 92 | 60.7% | +4.75% | +3.02% |
| `PEAD_t3_h90` | 234 | 145 | 89 | 62.0% | +6.73% | +4.27% |
| `PEAD_t5_h30` | 186 | 112 | 74 | 60.2% | +2.94% | +2.37% |
| `PEAD_t5_h60` | 186 | 115 | 71 | 61.8% | +4.65% | +3.37% |
| `PEAD_t5_h90` | 186 | 118 | 68 | 63.4% | +6.78% | +4.71% |
| `PEAD_t10_h30` | 104 | 65 | 39 | 62.5% | +3.11% | +2.37% |
| `PEAD_t10_h60` | 104 | 67 | 37 | 64.4% | +5.73% | +4.19% |
| `PEAD_t10_h90` | 104 | 69 | 35 | 66.3% | +8.03% | +4.96% |

## Interpretation

PEAD exploits the market's systematic underreaction to earnings surprises. Unlike price/vol pattern models, this uses FUNDAMENTAL information. The academic literature shows the drift persists for 60+ trading days after the announcement.

Key questions:
- Does PEAD beat equal-weight B&H on risk-adjusted basis (Sharpe)?
- Does lower surprise threshold (more trades) help or hurt?
- Is 60-day hold optimal, or does 30/90 work better?

