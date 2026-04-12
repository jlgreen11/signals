# Universe analysis — optimal stock universe for momentum

**Window**: 2022-04-01 → 2026-04-01 (4 years)
**Strategy**: 12-1 month cross-sectional momentum, monthly rebalance

## Results by universe size and number of positions held

| Config | N stocks | N long | Sharpe | CAGR | MDD | End $ |
|---|---:|---:|---:|---:|---:|---:|
| `Top-20 (current), top-20` | 20 | 20 | +0.906 | +20.6% | -26.4% | $21,155 |
| `Top-20 (current), top-10` | 20 | 10 | +0.840 | +20.8% | -24.9% | $21,295 |
| `Top-20 (current), top-5` | 20 | 5 | +0.839 | +24.5% | -28.9% | $23,993 |
| `Top-50, top-5` | 24 | 5 | +0.809 | +23.2% | -28.9% | $23,043 |
| `Top-100, top-5` | 24 | 5 | +0.809 | +23.2% | -28.9% | $23,043 |
| `Top-200, top-5` | 24 | 5 | +0.809 | +23.2% | -28.9% | $23,043 |
| `Full SP500, top-5` | 24 | 5 | +0.809 | +23.2% | -28.9% | $23,043 |
| `Top-50, top-10` | 24 | 10 | +0.788 | +19.1% | -24.9% | $20,075 |
| `Top-100, top-10` | 24 | 10 | +0.788 | +19.1% | -24.9% | $20,075 |
| `Top-200, top-10` | 24 | 10 | +0.788 | +19.1% | -24.9% | $20,075 |
| `Full SP500, top-10` | 24 | 10 | +0.788 | +19.1% | -24.9% | $20,075 |
| `Top-50, top-20` | 24 | 20 | +0.744 | +15.6% | -25.1% | $17,871 |
| `Top-100, top-20` | 24 | 20 | +0.744 | +15.6% | -25.1% | $17,871 |
| `Top-200, top-20` | 24 | 20 | +0.744 | +15.6% | -25.1% | $17,871 |
| `Full SP500, top-20` | 24 | 20 | +0.744 | +15.6% | -25.1% | $17,871 |
| `SP500 B&H` | 1 | 1 | +0.491 | +9.7% | -21.9% | $14,464 |

## Sector concentration (quarterly snapshots, best config)

| Sector | Picks | % |
|---|---:|---:|
| Information Technology | 15 | 30% |
| Communication Services | 11 | 22% |
| Financials | 11 | 22% |
| Consumer Discretionary | 6 | 12% |
| Health Care | 4 | 8% |
| Consumer Staples | 2 | 4% |
| Energy | 1 | 2% |

## Current picks with sectors

| Ticker | Weight | Sector |
|---|---:|---|
| `TSLA` | 10% | Consumer Discretionary |
| `NVDA` | 10% | Information Technology |
| `AAPL` | 10% | Information Technology |
| `GOOGL` | 10% | Communication Services |
| `AMD` | 10% | Information Technology |
| `AVGO` | 10% | Information Technology |
| `LLY` | 10% | Health Care |
| `JPM` | 10% | Financials |
| `XOM` | 10% | Energy |
| `JNJ` | 10% | Health Care |
